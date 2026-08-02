import io
import json
import shutil
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* or chardet.*doesn't match a supported version!",
)

from safetensors import safe_open
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from chat import answer
from convert_tokenql import convert, convert_matrix, convert_matrix_q4
from optimize_tokenql import optimize
from streaming_qwen import DiskKVCache, StreamedQwenModel, StreamingBackend, WeightStore
from tokenql import Session, SessionRepository, TokenQLEngine, parse_options, split_statements


class ParsingTests(unittest.TestCase):
    def test_plain_chat_passes_role_history(self):
        class FakeEngine:
            def generate_text(self, prompt, options, messages=None):
                self.prompt = prompt
                self.messages = messages
                return "reply"

        engine = FakeEngine()
        history = [
            {"role": "system", "content": "help"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
        ]
        result = answer(engine, "second", history, {"max_tokens": 3})
        self.assertEqual(result, "reply")
        self.assertEqual(engine.messages[-1], {"role": "user", "content": "second"})
        self.assertEqual(len(history), 3)

    def test_streamed_chat_template_switches_qwen3_thinking(self):
        class FakeTokenizer:
            def __init__(self):
                self.calls = []

            def apply_chat_template(self, messages, **kwargs):
                self.calls.append(kwargs)
                return torch.tensor([[1, 2]])

        backend = StreamingBackend.__new__(StreamingBackend)
        backend.tokenizer = FakeTokenizer()
        backend.thinking_enabled = False
        backend._format_messages([{"role": "user", "content": "hello"}])
        self.assertFalse(backend.tokenizer.calls[-1]["enable_thinking"])

        backend.set_thinking(True)
        backend._format_messages([{"role": "user", "content": "hello"}])
        self.assertTrue(backend.tokenizer.calls[-1]["enable_thinking"])

    def test_options(self):
        self.assertEqual(
            parse_options("strategy='greedy', temperature=0.7, seed=4"),
            {"strategy": "greedy", "temperature": 0.7, "seed": 4},
        )

    def test_semicolon_inside_string(self):
        statements = split_statements("CREATE SESSION 'x' WITH PROMPT 'a;b'; SHOW SESSIONS;")
        self.assertEqual(len(statements), 2)

    def test_one_shot_command_shape(self):
        class FakeBackend:
            model_name = "fake"

        engine = TokenQLEngine(FakeBackend(), object())
        engine._generate = lambda session, options, **kwargs: {"text": "Plain reply"}
        result = engine.execute(
            "SELECT RESPONSE FROM MODEL WHERE PROMPT = 'Hello' WITH (max_tokens=10)"
        )
        self.assertEqual(result, "Plain reply")

    def test_generation_does_not_advance_after_last_token(self):
        class FakeBackend:
            model_name = "fake"

            def __init__(self):
                self.advances = []

            def choose(self, session, **options):
                return {"token_id": 7}

            def commit(self, session, token_id, *, advance=True):
                self.advances.append(advance)
                session.generated_ids.append(token_id)

            def text(self, session):
                return "x" * len(session.generated_ids)

        backend = FakeBackend()
        engine = TokenQLEngine(backend, repository=None)
        result = engine._generate(Session("x", "", "fake"), {"max_tokens": 3})
        self.assertEqual(result["new_tokens"], 3)
        self.assertEqual(backend.advances, [True, True, False])

    def test_repository_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = SessionRepository(Path(directory) / "sessions.db")
            repository.insert(Session("test", "hello", "model", [1, 2]))
            loaded = repository.get("test")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.generated_ids, [1, 2])

    def test_replace_and_drop_session(self):
        class FakeBackend:
            model_name = "model"

        with tempfile.TemporaryDirectory() as directory:
            repository = SessionRepository(Path(directory) / "sessions.db")
            engine = TokenQLEngine(FakeBackend(), repository)
            engine.execute("CREATE SESSION 'same' WITH PROMPT 'first'")
            replaced = engine.execute("CREATE OR REPLACE SESSION 'same' WITH PROMPT 'second'")
            self.assertTrue(replaced["created"])
            self.assertEqual(repository.get("same").prompt, "second")
            dropped = engine.execute("DROP SESSION 'same'")
            self.assertTrue(dropped["dropped"])
            self.assertIsNone(repository.get("same"))

    def test_q8_matrix_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            matrix = torch.tensor([[1.0, -2.0, 0.5], [0.25, 0.75, -1.0]])
            record = convert_matrix(matrix, [2, 3], model_dir, "test.weight", 1)
            manifest = {
                "format": "tokenql-qwen-stream",
                "format_version": 1,
                "tensors": {"test.weight": record},
            }
            (model_dir / "tokenql_manifest.json").write_text(json.dumps(manifest))
            store = WeightStore(model_dir, buffer_mb=1)
            inputs = torch.tensor([[2.0, 1.0, -1.0]])
            actual = store.linear(inputs, "test.weight")
            expected = inputs @ matrix.t()
            self.assertTrue(torch.allclose(actual, expected, atol=0.03))
            store.close()

    def test_q4_matrix_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            matrix = torch.tensor([[1.0, -2.0, 0.5], [0.25, 0.75, -1.0]])
            record = convert_matrix_q4(matrix, [2, 3], model_dir, "test.weight", 1, 4)
            manifest = {
                "format": "tokenql-qwen-stream",
                "format_version": 1,
                "tensors": {"test.weight": record},
            }
            (model_dir / "tokenql_manifest.json").write_text(json.dumps(manifest))
            store = WeightStore(model_dir, buffer_mb=1)
            inputs = torch.tensor([[2.0, 1.0, -1.0]])
            actual = store.linear(inputs, "test.weight")
            expected = inputs @ matrix.t()
            self.assertTrue(torch.allclose(actual, expected, atol=0.4))
            store.close()

    def test_contiguous_q4_expert_uses_one_shared_read_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            tensors = {}
            packed = bytearray()
            expert_offsets = []
            for expert_id in (0, 1, 2):
                expert_offsets.append(len(packed))
                for index, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
                    name = f"model.layers.0.mlp.experts.{expert_id}.{projection}.weight"
                    matrix = (
                        torch.arange(8, dtype=torch.float32).reshape(2, 4) + index + expert_id * 10
                    )
                    record = convert_matrix_q4(
                        matrix, [2, 4], model_dir, name, expert_id * 3 + index + 1, 4
                    )
                    data = (model_dir / record["data_file"]).read_bytes()
                    scales = (model_dir / record["scale_file"]).read_bytes()
                    record["data_file"] = "expert.pack"
                    record["scale_file"] = "expert.pack"
                    record["data_offset"] = len(packed)
                    packed.extend(data)
                    record["scale_offset"] = len(packed)
                    packed.extend(scales)
                    tensors[name] = record
            (model_dir / "expert.pack").write_bytes(packed)
            manifest = {
                "format": "tokenql-qwen-stream",
                "format_version": 1,
                "config": {"num_hidden_layers": 1},
                "tensors": tensors,
                "moe": {
                    "packs": [
                        {
                            "layer": 0,
                            "file": "expert.pack",
                            "bytes": len(packed),
                            "expert_offsets": expert_offsets,
                        }
                    ]
                },
            }
            (model_dir / "tokenql_manifest.json").write_text(json.dumps(manifest))

            store = WeightStore(model_dir, buffer_mb=1, prefetch=False)
            # This test covers the portable contiguous layout, independent of
            # which native kernel is available on the test machine.
            store.native_avx512_q4_available = True
            before = store.bytes_read
            expert = store.load_resident_expert(0, 0)
            self.assertIsNotNone(expert)
            self.assertEqual(store.bytes_read - before, expert_offsets[1])
            self.assertEqual(len(store._file_handles), 1)

            def root_owner(values):
                while isinstance(values.base, np.ndarray):
                    values = values.base
                return values

            self.assertIs(
                root_owner(expert["gate_proj"].data),
                root_owner(expert["up_proj"].data),
            )
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.layers.0.mlp.experts.0.{projection}.weight"
                separately = store.load_resident_matrix(name)
                self.assertTrue(np.array_equal(expert[projection].data, separately.data))
                self.assertTrue(np.array_equal(expert[projection].scales, separately.scales))
            self.assertEqual(len(store._file_handles), 1)
            before = store.bytes_read
            group = store.load_resident_expert_group(0, (0, 1))
            self.assertEqual(store.bytes_read - before, expert_offsets[2])
            self.assertFalse(
                np.shares_memory(
                    root_owner(group[0]["gate_proj"].data),
                    root_owner(group[1]["gate_proj"].data),
                )
            )
            for expert_id in (0, 1):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    name = f"model.layers.0.mlp.experts.{expert_id}.{projection}.weight"
                    separately = store.load_resident_matrix(name)
                    self.assertTrue(
                        np.array_equal(group[expert_id][projection].data, separately.data)
                    )
                    self.assertTrue(
                        np.array_equal(
                            group[expert_id][projection].scales,
                            separately.scales,
                        )
                    )
            before = store.bytes_read
            gapped = store.load_resident_expert_group(0, (0, 2))
            self.assertEqual(store.bytes_read - before, len(packed))
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.layers.0.mlp.experts.2.{projection}.weight"
                separately = store.load_resident_matrix(name)
                self.assertTrue(np.array_equal(gapped[2][projection].data, separately.data))
                self.assertTrue(np.array_equal(gapped[2][projection].scales, separately.scales))
            store.close()

    def test_native_q4_prepack_round_trip(self):
        if not hasattr(torch, "_weight_int4pack_mm_for_cpu"):
            self.skipTest("PyTorch native CPU INT4 kernel is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            torch.manual_seed(3)
            matrix = torch.randn(32, 128)
            record = convert_matrix_q4(matrix, [32, 128], model_dir, "test.weight", 1, 128)
            manifest = {
                "format": "tokenql-qwen-stream",
                "format_version": 1,
                "tensors": {"test.weight": record},
            }
            (model_dir / "tokenql_manifest.json").write_text(json.dumps(manifest))
            with redirect_stdout(io.StringIO()):
                result = optimize(model_dir, chunk_mb=1)
            self.assertEqual(result["tensors"], 1)

            store = WeightStore(model_dir, buffer_mb=1, matmul="int8", prefetch=False)
            inputs = torch.randn(3, 128)
            native_actual = (
                store.linear(inputs, "test.weight") if store.native_avx512_q4_available else None
            )
            native_pair = (
                store.linear_pair(inputs, "test.weight", "test.weight")
                if store.native_avx512_q4_available
                else None
            )
            # This test specifically covers the PyTorch int4pack sidecar. The
            # normal aligned-Q4 path uses AVX-512 or the packed Numba kernel.
            store.native_avx512_q4_available = False
            store.numba_q4_available = False
            actual = store.linear(inputs, "test.weight")
            portable = WeightStore._unpack_q4(
                np.fromfile(model_dir / record["data_file"], dtype=np.uint8).reshape(32, 64),
                128,
            )
            scales = np.fromfile(model_dir / record["scale_file"], dtype=np.float32).reshape(32, 1)
            expected = inputs @ torch.from_numpy(portable * scales).t().float()
            self.assertTrue(torch.allclose(actual, expected, atol=1e-4))
            if native_actual is not None:
                # The native FP32-activation kernel should preserve Q4 output.
                similarity = torch.nn.functional.cosine_similarity(
                    native_actual.flatten(), expected.flatten(), dim=0
                )
                self.assertGreater(float(similarity), 0.999999)
                self.assertTrue(torch.allclose(native_actual, expected, atol=1e-4))
                self.assertTrue(
                    torch.equal(native_pair[0], native_actual)
                    and torch.equal(native_pair[1], native_actual)
                )
            store.close()

    def test_disk_kv_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = DiskKVCache(Path(directory) / "cache.bin", 2, 1, 8, 4)
            keys = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
            values = keys + 10
            cache.write(1, 0, keys, values)
            loaded_keys, loaded_values = cache.read(1, 3)
            self.assertTrue(torch.equal(loaded_keys, keys))
            self.assertTrue(torch.equal(loaded_values, values))
            cache.close()

    def test_qwen3_moe_matches_reference_and_bounds_expert_cache(self):
        config = Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
            attention_bias=False,
            rope_theta=10000.0,
        )
        torch.manual_seed(7)
        reference = Qwen3MoeForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        with torch.inference_mode():
            expected = reference(input_ids).logits[0, -1].float()
            expected_next = (
                reference(torch.tensor([[1, 2, 3, 4]], dtype=torch.long)).logits[0, -1].float()
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "converted"
            reference.save_pretrained(source, safe_serialization=True)
            with redirect_stdout(io.StringIO()):
                manifest = convert(str(source), output, True, 1, False, "q8", 128)
            self.assertEqual(manifest["architecture"], "qwen3_moe")
            self.assertGreater(manifest["expert_weight_bytes"], 0)
            self.assertEqual(len(manifest["moe"]["packs"]), 2)
            self.assertTrue((output / "weights/experts/layer-000.experts.pack").exists())

            store = WeightStore(
                output,
                buffer_mb=1,
                matmul="int8",
                prefetch=False,
                ram_budget_mb=2,
            )
            runtime = StreamedQwenModel(store)
            cache = runtime.create_cache(root / "kv.bin", 64)
            actual = runtime.forward(input_ids[0].tolist(), cache).float()
            similarity = torch.nn.functional.cosine_similarity(actual, expected, dim=0)
            self.assertGreater(float(similarity), 0.999)
            self.assertEqual(int(actual.argmax()), int(expected.argmax()))
            actual_next = runtime.forward([4], cache).float()
            next_similarity = torch.nn.functional.cosine_similarity(
                actual_next, expected_next, dim=0
            )
            self.assertGreater(float(next_similarity), 0.999)
            self.assertEqual(cache.length, 4)

            # Whole-expert slot allocation leaves less than one expert of
            # unusable capacity instead of one fragment per layer.
            allocated = sum(store.expert_cache._layer_capacities.values())
            smallest_expert = min(
                store.expert_cache._entry_size(layer, 0)
                for layer in range(config.num_hidden_layers)
            )
            self.assertLessEqual(allocated, store.expert_cache.capacity_bytes)
            self.assertLess(store.expert_cache.capacity_bytes - allocated, smallest_expert)

            # A correct previous-route prediction is consumed through the
            # asynchronous path and reported as useful prefetch I/O.
            store.expert_cache.clear()
            store.prefetch = True
            store.expert_cache.prefetch(0, [0], predicted=True)
            store.expert_cache.cancel_layer_predictions_except(0, [0])
            self.assertIsNotNone(store.expert_cache.get(0, 0))
            prediction_stats = store.expert_cache.stats()
            self.assertEqual(prediction_stats["prediction"]["correct"], 1)
            self.assertEqual(prediction_stats["prediction"]["io_hits"], 1)
            self.assertGreaterEqual(prediction_stats["prefetch"]["useful"], 1)

            # A repeated lookup must hit RAM without additional logical disk I/O.
            store.expert_cache.get(0, 0)
            reads = store.bytes_read
            hits = store.expert_cache.hits
            store.expert_cache.get(0, 0)
            self.assertEqual(store.bytes_read, reads)
            self.assertEqual(store.expert_cache.hits, hits + 1)

            # With room for one expert, a second expert is transiently bypassed
            # instead of evicting the protected entry and causing scan thrash.
            one_expert = store.expert_cache._entry_size(0, 0)
            store.expert_cache.set_capacity(one_expert)
            store.expert_cache.get(0, 1)
            self.assertLessEqual(store.expert_cache.resident_bytes, one_expert)
            self.assertGreater(store.expert_cache.bypasses, 0)
            cache.close()
            store.close()

            # The intended large-model path is Q4, whose packed expert offsets
            # must be executable as well as Q8.
            q4_output = root / "converted-q4"
            with redirect_stdout(io.StringIO()):
                convert(str(source), q4_output, True, 1, False, "q4", 4)
            q4_store = WeightStore(
                q4_output,
                buffer_mb=1,
                matmul="int8",
                prefetch=False,
                ram_budget_mb=2,
            )
            q4_runtime = StreamedQwenModel(q4_store)
            q4_cache = q4_runtime.create_cache(root / "kv-q4.bin", 64)
            q4_actual = q4_runtime.forward(input_ids[0].tolist(), q4_cache).float()
            q4_similarity = torch.nn.functional.cosine_similarity(q4_actual, expected, dim=0)
            self.assertGreater(float(q4_similarity), 0.99)
            self.assertEqual(int(q4_actual.argmax()), int(expected.argmax()))
            q4_cache.close()
            q4_store.close()

            # An interrupted conversion has no manifest. Complete tensor files
            # must be recovered by size and reused on the next invocation.
            partial = root / "partial"
            partial.mkdir()
            shutil.copy2(source / "config.json", partial / "config.json")
            with safe_open(source / "model.safetensors", framework="pt", device="cpu") as tensors:
                tensor_slice = tensors.get_slice("model.embed_tokens.weight")
                shape = list(tensor_slice.get_shape())
                convert_matrix(
                    tensor_slice,
                    shape,
                    partial,
                    "model.embed_tokens.weight",
                    1,
                )
            with redirect_stdout(io.StringIO()):
                resumed = convert(str(source), partial, True, 1, False, "q8", 128)
            self.assertEqual(resumed["resumed_tensors"], 1)
            self.assertTrue((partial / "tokenql_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
