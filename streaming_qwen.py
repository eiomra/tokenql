"""Bounded-memory dense Qwen and expert-paged Qwen3-MoE inference.

This module implements the decoder directly. It does not instantiate a Hugging
Face model and it does not ask Ollama or llama.cpp to generate tokens. Numba
compiles the packed-Q4 decode/prefill kernels and PyTorch provides tensor
operations and portable fallbacks. Shared matrices are mapped in bounded chunks;
selected MoE experts can remain compressed in a fixed-budget LRU while the rest
stay in offset-addressable SSD packs.
"""

from __future__ import annotations

import atexit
import ctypes
import json
import logging
import math
import re
import shutil
import tempfile
import threading
import time
import uuid
import warnings
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_NATIVE_Q4_GEMM = None
_NATIVE_Q4_GEMM_PAIR = None
NATIVE_AVX512_Q4_AVAILABLE = False
try:
    cpu_features = np._core._multiarray_umath.__cpu_features__
    native_q4_path = Path(__file__).resolve().with_name("_tokenql_q4_avx512.dll")
    if native_q4_path.exists() and all(
        cpu_features.get(feature, False)
        for feature in ("AVX512F", "AVX512BW", "AVX512VBMI", "AVX512VNNI")
    ):
        native_q4_library = ctypes.CDLL(str(native_q4_path))
        _NATIVE_Q4_GEMM = native_q4_library.tokenql_q4_gemm_f32
        _NATIVE_Q4_GEMM.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        _NATIVE_Q4_GEMM.restype = ctypes.c_int
        _NATIVE_Q4_GEMM_PAIR = native_q4_library.tokenql_q4_gemm_f32_pair
        _NATIVE_Q4_GEMM_PAIR.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        _NATIVE_Q4_GEMM_PAIR.restype = ctypes.c_int
        NATIVE_AVX512_Q4_AVAILABLE = True
except (AttributeError, KeyError, OSError):
    _NATIVE_Q4_GEMM = None
    _NATIVE_Q4_GEMM_PAIR = None
    NATIVE_AVX512_Q4_AVAILABLE = False

try:
    from numba import config as numba_config
    from numba import njit, prange
    from numba import set_num_threads as set_numba_threads

    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_q4_gemv(
        inputs: np.ndarray,
        packed: np.ndarray,
        scales: np.ndarray,
        block_size: int,
        columns: int,
    ) -> np.ndarray:
        """Direct signed-Q4 GEMV; never expands the packed weight matrix."""
        rows = packed.shape[0]
        blocks = scales.shape[1]
        half_block = block_size // 2
        output = np.empty(rows, dtype=np.float32)
        for row in prange(rows):
            total = np.float32(0.0)
            for block in range(blocks):
                dot = np.float32(0.0)
                packed_base = block * half_block
                input_base = block * block_size
                for index in range(half_block):
                    value = int(packed[row, packed_base + index])
                    low = value & 15
                    high = value >> 4
                    if low >= 8:
                        low -= 16
                    if high >= 8:
                        high -= 16
                    input_index = input_base + index * 2
                    if input_index < columns:
                        dot += low * inputs[input_index]
                    if input_index + 1 < columns:
                        dot += high * inputs[input_index + 1]
                total += dot * scales[row, block]
            output[row] = total
        return output

    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_q4_gemv_aligned(
        inputs: np.ndarray,
        packed: np.ndarray,
        scales: np.ndarray,
        block_size: int,
    ) -> np.ndarray:
        """Fast Q4 GEMV for matrices whose logical width needs no padding."""
        rows = packed.shape[0]
        blocks = scales.shape[1]
        half_block = block_size // 2
        output = np.empty(rows, dtype=np.float32)
        for row in prange(rows):
            total = np.float32(0.0)
            for block in range(blocks):
                dot = np.float32(0.0)
                packed_base = block * half_block
                input_base = block * block_size
                for index in range(half_block):
                    value = int(packed[row, packed_base + index])
                    low = value & 15
                    high = value >> 4
                    if low >= 8:
                        low -= 16
                    if high >= 8:
                        high -= 16
                    input_index = input_base + index * 2
                    dot += low * inputs[input_index] + high * inputs[input_index + 1]
                total += dot * scales[row, block]
            output[row] = total
        return output

    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_q4_gemv_aligned_pair(
        inputs: np.ndarray,
        packed_first: np.ndarray,
        scales_first: np.ndarray,
        packed_second: np.ndarray,
        scales_second: np.ndarray,
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate equal-shaped gate/up matrices in one worker launch."""
        rows = packed_first.shape[0]
        blocks = scales_first.shape[1]
        half_block = block_size // 2
        first = np.empty(rows, dtype=np.float32)
        second = np.empty(rows, dtype=np.float32)
        for row in prange(rows):
            total_first = np.float32(0.0)
            total_second = np.float32(0.0)
            for block in range(blocks):
                dot_first = np.float32(0.0)
                dot_second = np.float32(0.0)
                packed_base = block * half_block
                input_base = block * block_size
                for index in range(half_block):
                    first_value = int(packed_first[row, packed_base + index])
                    second_value = int(packed_second[row, packed_base + index])
                    first_low = first_value & 15
                    first_high = first_value >> 4
                    second_low = second_value & 15
                    second_high = second_value >> 4
                    if first_low >= 8:
                        first_low -= 16
                    if first_high >= 8:
                        first_high -= 16
                    if second_low >= 8:
                        second_low -= 16
                    if second_high >= 8:
                        second_high -= 16
                    input_index = input_base + index * 2
                    low_input = inputs[input_index]
                    high_input = inputs[input_index + 1]
                    dot_first += first_low * low_input + first_high * high_input
                    dot_second += second_low * low_input + second_high * high_input
                total_first += dot_first * scales_first[row, block]
                total_second += dot_second * scales_second[row, block]
            first[row] = total_first
            second[row] = total_second
        return first, second

    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_q4_gemm_aligned(
        inputs: np.ndarray,
        packed: np.ndarray,
        scales: np.ndarray,
        block_size: int,
    ) -> np.ndarray:
        """Batched packed-Q4 projection for prompt prefill."""
        tokens = inputs.shape[0]
        rows = packed.shape[0]
        blocks = scales.shape[1]
        half_block = block_size // 2
        output = np.empty((tokens, rows), dtype=np.float32)
        for task in prange(tokens * rows):
            token = task // rows
            row = task - token * rows
            total = np.float32(0.0)
            for block in range(blocks):
                dot = np.float32(0.0)
                packed_base = block * half_block
                input_base = block * block_size
                for index in range(half_block):
                    value = int(packed[row, packed_base + index])
                    low = value & 15
                    high = value >> 4
                    if low >= 8:
                        low -= 16
                    if high >= 8:
                        high -= 16
                    input_index = input_base + index * 2
                    dot += low * inputs[token, input_index] + high * inputs[token, input_index + 1]
                total += dot * scales[row, block]
            output[token, row] = total
        return output

    NUMBA_Q4_AVAILABLE = True
except (ImportError, OSError):
    NUMBA_Q4_AVAILABLE = False
    set_numba_threads = None

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* or chardet.*doesn't match a supported version!",
)
logging.getLogger("torchao.kernel.intmm").setLevel(logging.ERROR)

from transformers import AutoTokenizer


class StreamingModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResidentMatrix:
    """Compressed matrix bytes kept inside TokenQL's managed RAM budget."""

    data: np.ndarray
    scales: np.ndarray
    native_q4: bool = False

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes + self.scales.nbytes)


class ExpertCache:
    """Thread-safe segmented cache with bounded speculative expert prefetch.

    An expert is admitted atomically: its gate, up, and down matrices are all
    resident or all streamed.  This prevents a partially cached expert from
    consuming memory without avoiding the SSD stall on its critical path.
    """

    PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

    def __init__(self, store: "WeightStore", capacity_bytes: int):
        self.store = store
        self.capacity_bytes = max(0, int(capacity_bytes))
        self.layer_count = max(1, int(store.manifest.get("config", {}).get("num_hidden_layers", 1)))
        self.resident_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.bypasses = 0
        self.loaded_bytes = 0
        self.admission_enabled = True
        self.prefetch_submitted = 0
        self.prefetch_useful = 0
        self.prefetch_ready = 0
        self.prefetch_waits = 0
        self.prefetch_wait_seconds = 0.0
        self.prefetch_wasted = 0
        self.coalesced_reads = 0
        self.coalesced_experts = 0
        self.prediction_candidates = 0
        self.prediction_correct = 0
        self.prediction_misses = 0
        self.prediction_cache_hits = 0
        self.prediction_io_hits = 0
        self._entries: OrderedDict[tuple[int, int], dict[str, ResidentMatrix]] = OrderedDict()
        self._layer_bytes: dict[int, int] = defaultdict(int)
        self._frequency: dict[tuple[int, int], int] = defaultdict(int)
        self._layer_accesses: dict[int, int] = defaultdict(int)
        self._layer_capacities: dict[int, int] = {}
        self._pending: dict[tuple[int, int], tuple[Future[Any], bool, int | None]] = {}
        self._predictions: dict[int, set[tuple[int, int]]] = defaultdict(set)
        self._entry_intervals: dict[tuple[int, int], tuple[Path, int, int]] = {}
        for pack in store.manifest.get("moe", {}).get("packs", []):
            layer = int(pack["layer"])
            offsets = list(map(int, pack.get("expert_offsets", [])))
            path = store._resolved_paths.get(str(pack.get("file", "")))
            if path is None:
                continue
            pack_bytes = int(pack.get("bytes", 0))
            for expert, start in enumerate(offsets):
                end = offsets[expert + 1] if expert + 1 < len(offsets) else pack_bytes
                self._entry_intervals[(layer, expert)] = (path, start, end)
        self._lock = threading.RLock()
        self._recompute_layer_capacities()

    @staticmethod
    def tensor_name(layer: int, expert: int, projection: str) -> str:
        return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"

    def _entry_size(self, layer: int, expert: int) -> int:
        return sum(
            int(self.store.record(self.tensor_name(layer, expert, projection))["bytes"])
            for projection in self.PROJECTIONS
        )

    def entry_interval(self, layer: int, expert: int) -> tuple[Path, int, int] | None:
        """Return the exact contiguous byte interval occupied by one Q4 expert."""
        cached = self._entry_intervals.get((int(layer), int(expert)))
        if cached is not None:
            return cached
        segments: list[tuple[int, int]] = []
        common_path: Path | None = None
        for projection in self.PROJECTIONS:
            record = self.store.record(self.tensor_name(layer, expert, projection))
            if record.get("storage") != "q4_block":
                return None
            data_path = self.store._path(record)
            scale_path = self.store._path(record, "scale_file")
            if data_path != scale_path:
                return None
            if common_path is None:
                common_path = data_path
            elif common_path != data_path:
                return None
            rows, _ = map(int, record["shape"])
            padded = int(record["padded_columns"])
            blocks = padded // int(record["block_size"])
            segments.append((int(record.get("data_offset", 0)), rows * (padded // 2)))
            segments.append(
                (
                    int(record.get("scale_offset", 0)),
                    rows * blocks * np.dtype(np.float32).itemsize,
                )
            )
        ordered = sorted(segments)
        if common_path is None or any(
            ordered[index][0] + ordered[index][1] != ordered[index + 1][0]
            for index in range(len(ordered) - 1)
        ):
            return None
        result = common_path, ordered[0][0], ordered[-1][0] + ordered[-1][1]
        self._entry_intervals[(int(layer), int(expert))] = result
        return result

    def uncached_experts(self, layer: int, experts: list[int] | tuple[int, ...]) -> list[int]:
        with self._lock:
            return [
                int(expert)
                for expert in experts
                if (int(layer), int(expert)) not in self._entries
                and (int(layer), int(expert)) not in self._pending
            ]

    def _recompute_layer_capacities(self) -> None:
        """Allocate complete expert slots instead of stranding quota fragments."""
        capacities = {layer: 0 for layer in range(self.layer_count)}
        if self.tensor_name(0, 0, self.PROJECTIONS[0]) not in self.store.tensors:
            self._layer_capacities = capacities
            return
        sizes = {layer: self._entry_size(layer, 0) for layer in range(self.layer_count)}
        round_bytes = sum(sizes.values())
        complete_rounds = self.capacity_bytes // round_bytes if round_bytes else 0
        for layer, required in sizes.items():
            capacities[layer] = complete_rounds * required
        remaining = self.capacity_bytes - complete_rounds * round_bytes
        for layer in range(self.layer_count):
            required = sizes[layer]
            if required <= remaining:
                capacities[layer] += required
                remaining -= required
        self._layer_capacities = capacities

    def _layer_capacity(self, layer: int) -> int:
        return self._layer_capacities.get(int(layer), 0)

    def set_capacity(self, capacity_bytes: int) -> None:
        with self._lock:
            self.capacity_bytes = max(0, int(capacity_bytes))
            self._recompute_layer_capacities()
            for layer in range(self.layer_count):
                while self._layer_bytes[layer] > self._layer_capacity(layer):
                    key = min(
                        (candidate for candidate in self._entries if candidate[0] == layer),
                        key=lambda candidate: self._frequency[candidate],
                        default=None,
                    )
                    if key is None:
                        break
                    self._remove(key)
            self._evict_until(0)

    def _remove(self, key: tuple[int, int]) -> None:
        entry = self._entries.pop(key)
        removed = sum(matrix.nbytes for matrix in entry.values())
        self.resident_bytes -= removed
        self._layer_bytes[key[0]] -= removed
        self.evictions += 1

    def _touch(self, key: tuple[int, int], frequency: int = 1) -> None:
        self._frequency[key] += max(1, int(frequency))
        layer = key[0]
        self._layer_accesses[layer] += 1
        if self._layer_accesses[layer] % 256 == 0:
            for candidate in list(self._frequency):
                if candidate[0] == layer:
                    self._frequency[candidate] = max(1, self._frequency[candidate] // 2)

    def _evict_until(self, incoming_bytes: int) -> None:
        while self._entries and self.resident_bytes + incoming_bytes > self.capacity_bytes:
            key = next(iter(self._entries))
            self._remove(key)

    def _load_entry(self, key: tuple[int, int]) -> dict[str, ResidentMatrix]:
        packed = self.store.load_resident_expert(key[0], key[1])
        if packed is not None:
            return packed
        return {
            projection: self.store.load_resident_matrix(
                self.tensor_name(key[0], key[1], projection)
            )
            for projection in self.PROJECTIONS
        }

    def prefetch(
        self, layer: int, experts: list[int] | tuple[int, ...], *, predicted: bool
    ) -> None:
        """Schedule compressed experts without exceeding the bounded decode lookahead."""
        if not self.store.prefetch or not self.admission_enabled:
            return
        with self._lock:
            for expert in experts:
                key = (int(layer), int(expert))
                if predicted:
                    self.prediction_candidates += 1
                    self._predictions[int(layer)].add(key)
                if key in self._entries:
                    continue
                if key in self._pending:
                    continue
                executor = self.store._prediction_executor if predicted else self.store._executor
                future = executor.submit(self._load_entry, key)
                self._pending[key] = (future, predicted, None)
                self.prefetch_submitted += 1

    def prefetch_coalesced(
        self,
        layer: int,
        experts: list[int] | tuple[int, ...],
        *,
        gap_experts: int = 0,
    ) -> None:
        """Schedule exact prefill experts with a bounded skipped-expert gap."""
        if not self.store.prefetch or not self.admission_enabled:
            return
        with self._lock:
            candidates = []
            for expert in sorted(set(map(int, experts))):
                key = (int(layer), expert)
                if key in self._entries or key in self._pending:
                    continue
                interval = self.entry_interval(layer, expert)
                if interval is None:
                    self.prefetch(layer, [expert], predicted=False)
                    continue
                candidates.append((expert, *interval))
            candidates.sort(key=lambda item: item[2])
            runs: list[list[tuple[int, Path, int, int]]] = []
            for candidate in candidates:
                if (
                    runs
                    and runs[-1][-1][1] == candidate[1]
                    and candidate[0] - runs[-1][-1][0] - 1 <= max(0, int(gap_experts))
                ):
                    runs[-1].append(candidate)
                else:
                    runs.append([candidate])
            for run in runs:
                run_experts = tuple(item[0] for item in run)
                if len(run_experts) == 1:
                    key = (int(layer), run_experts[0])
                    future = self.store._executor.submit(self._load_entry, key)
                    self._pending[key] = (future, False, None)
                else:
                    future = self.store._executor.submit(
                        self.store.load_resident_expert_group,
                        int(layer),
                        run_experts,
                    )
                    for expert in run_experts:
                        self._pending[(int(layer), expert)] = (
                            future,
                            False,
                            expert,
                        )
                    self.coalesced_reads += 1
                    self.coalesced_experts += len(run_experts)
                self.prefetch_submitted += len(run_experts)

    def cancel_layer_predictions_except(
        self, layer: int, experts: list[int] | tuple[int, ...]
    ) -> None:
        keep = {(int(layer), int(expert)) for expert in experts}
        with self._lock:
            predicted_keys = self._predictions.pop(int(layer), set())
            correct = predicted_keys & keep
            self.prediction_correct += len(correct)
            self.prediction_misses += len(predicted_keys - keep)
            self.prediction_cache_hits += sum(key in self._entries for key in correct)
            for key, (future, predicted, _) in list(self._pending.items()):
                if key[0] != int(layer) or key in keep or not predicted:
                    continue
                self._pending.pop(key, None)
                future.cancel()
                self.prefetch_wasted += 1

    def get(
        self, layer: int, expert: int, *, frequency: int = 1
    ) -> dict[str, ResidentMatrix] | None:
        key = (int(layer), int(expert))
        with self._lock:
            self._touch(key, frequency)
            cached = self._entries.pop(key, None)
            if cached is not None:
                self._entries[key] = cached
                self.hits += 1
                return cached

            self.misses += 1
            if not self.admission_enabled:
                self.bypasses += 1
                return None
            required = self._entry_size(*key)
            layer_capacity = self._layer_capacity(key[0])
            if required > layer_capacity:
                self.bypasses += 1
                return None
            pending = self._pending.pop(key, None)

        if pending is not None:
            future, predicted, group_expert = pending
            ready = future.done()
            wait_started = time.perf_counter()
            loaded = future.result()
            entry = loaded if group_expert is None else loaded[group_expert]
            wait_seconds = time.perf_counter() - wait_started
            with self._lock:
                self.prefetch_useful += 1
                self.prefetch_ready += int(ready)
                self.prefetch_waits += int(not ready)
                self.prefetch_wait_seconds += wait_seconds
                self.prediction_io_hits += int(predicted)
        else:
            entry = self._load_entry(key)

        with self._lock:
            actual = sum(matrix.nbytes for matrix in entry.values())
            if actual > self.capacity_bytes:
                self.bypasses += 1
                return entry
            if self._layer_bytes[key[0]] + actual > layer_capacity:
                victims = [candidate for candidate in self._entries if candidate[0] == key[0]]
                victim = min(
                    victims,
                    key=lambda candidate: self._frequency[candidate],
                    default=None,
                )
                # Strictly greater frequency is required. Equal-frequency scan
                # traffic remains transient and therefore cannot churn a layer.
                if victim is None or self._frequency[key] <= self._frequency[victim]:
                    self.bypasses += 1
                    return entry
                self._remove(victim)
            self._entries[key] = entry
            self.resident_bytes += actual
            self._layer_bytes[key[0]] += actual
            self.loaded_bytes += actual
            return entry

    def clear(self) -> None:
        with self._lock:
            for future in {pending[0] for pending in self._pending.values()}:
                future.cancel()
            self._pending.clear()
            self._predictions.clear()
            self._entries.clear()
            self._layer_bytes.clear()
            self._frequency.clear()
            self._layer_accesses.clear()
            self.resident_bytes = 0

    def stats(self) -> dict[str, Any]:
        attempts = self.hits + self.misses
        return {
            "capacity_bytes": self.capacity_bytes,
            "resident_bytes": self.resident_bytes,
            "resident_experts": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / attempts if attempts else 0.0,
            "evictions": self.evictions,
            "bypasses": self.bypasses,
            "loaded_bytes": self.loaded_bytes,
            "coalesced_reads": self.coalesced_reads,
            "coalesced_experts": self.coalesced_experts,
            "policy": "whole-slot-segmented-frequency",
            "layers": self.layer_count,
            "capacity_slack_bytes": self.capacity_bytes - sum(self._layer_capacities.values()),
            "prefetch": {
                "submitted": self.prefetch_submitted,
                "useful": self.prefetch_useful,
                "ready": self.prefetch_ready,
                "waits": self.prefetch_waits,
                "wait_seconds": self.prefetch_wait_seconds,
                "wasted": self.prefetch_wasted,
                "useful_rate": (
                    self.prefetch_useful / self.prefetch_submitted
                    if self.prefetch_submitted
                    else 0.0
                ),
            },
            "prediction": {
                "candidates": self.prediction_candidates,
                "cache_hits": self.prediction_cache_hits,
                "io_hits": self.prediction_io_hits,
                "correct": self.prediction_correct,
                "misses": self.prediction_misses,
                "accuracy": (
                    self.prediction_correct / self.prediction_candidates
                    if self.prediction_candidates
                    else 0.0
                ),
            },
        }


class WeightStore:
    """Read TokenQL matrices a bounded number of rows at a time."""

    def __init__(
        self,
        model_dir: str | Path,
        buffer_mb: int = 64,
        matmul: str = "auto",
        prefetch: bool = True,
        ram_budget_mb: int | None = None,
        io_workers: int = 1,
    ):
        self.model_dir = Path(model_dir).resolve()
        manifest_path = self.model_dir / "tokenql_manifest.json"
        if not manifest_path.exists():
            raise StreamingModelError(f"No tokenql_manifest.json under {self.model_dir}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "tokenql-qwen-stream":
            raise StreamingModelError("Directory is not a TokenQL streamed model")
        if self.manifest.get("format_version") != 1:
            raise StreamingModelError(
                f"Unsupported TokenQL model version {self.manifest.get('format_version')}"
            )
        self.tensors: dict[str, dict[str, Any]] = self.manifest["tensors"]
        # Resolve and validate each distinct pack once. Calling Path.resolve()
        # for every matrix access is particularly expensive on Windows because
        # it performs filesystem metadata/final-path queries; decode touches
        # thousands of matrices per response while only a small set of pack
        # files actually exists.
        self._resolved_paths: dict[str, Path] = {}
        for record in self.tensors.values():
            for key in ("data_file", "scale_file", "native_data_file"):
                relative = record.get(key)
                if relative is None or relative in self._resolved_paths:
                    continue
                path = (self.model_dir / str(relative)).resolve()
                try:
                    path.relative_to(self.model_dir)
                except ValueError as exc:
                    raise StreamingModelError(
                        f"Weight path escapes model directory: {path}"
                    ) from exc
                self._resolved_paths[str(relative)] = path
        self._thread_files = threading.local()
        self._file_handles: list[Any] = []
        self._file_handles_lock = threading.Lock()
        self.buffer_bytes = max(1, int(buffer_mb)) * 1024 * 1024
        requested_budget = (
            self.buffer_bytes if ram_budget_mb is None else max(1, int(ram_budget_mb)) * 1024 * 1024
        )
        if requested_budget < self.buffer_bytes:
            raise StreamingModelError("RAM budget cannot be smaller than the weight buffer")
        self.ram_budget_bytes = requested_budget
        self.bytes_read = 0
        self._vectors: dict[str, torch.Tensor] = {}
        self._vector_bytes = 0
        self._shared_matrices: dict[str, ResidentMatrix] = {}
        self._shared_matrix_bytes = 0
        self._shared_lock = threading.RLock()
        if matmul not in {"auto", "int8", "float"}:
            raise StreamingModelError("matmul must be auto, int8, or float")
        has_integer_mm = hasattr(torch, "_int_mm")
        if matmul == "int8" and not has_integer_mm:
            raise StreamingModelError("This PyTorch build has no CPU integer matmul primitive")
        self.matmul_mode = (
            "int8" if matmul == "int8" or (matmul == "auto" and has_integer_mm) else "float"
        )
        self.native_q4_available = all(
            hasattr(torch, name)
            for name in (
                "_convert_weight_to_int4pack_for_cpu",
                "_weight_int4pack_mm_for_cpu",
            )
        )
        self.native_avx512_q4_available = NATIVE_AVX512_Q4_AVAILABLE
        self.q4_threads = max(1, torch.get_num_threads())
        self.numba_q4_available = NUMBA_Q4_AVAILABLE
        if self.numba_q4_available and set_numba_threads is not None:
            set_numba_threads(
                max(1, min(torch.get_num_threads(), int(numba_config.NUMBA_NUM_THREADS)))
            )
        self.prefetch = bool(prefetch)
        self.io_workers = max(1, int(io_workers))
        self._executor = ThreadPoolExecutor(
            max_workers=self.io_workers, thread_name_prefix="tokenql-io"
        )
        # Keep speculative reads out of the exact-read queue. A single worker
        # bounds their physical SSD contention while exact current-token reads
        # retain all configured I/O workers.
        self._prediction_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tokenql-predict-io"
        )
        self.expert_cache = ExpertCache(self, self.ram_budget_bytes - self.buffer_bytes)
        self._shared_reserve_bytes = 0
        if self.manifest.get("architecture") == "qwen3_moe":
            # Retain room for at least one complete expert in every layer after
            # guaranteed-reuse attention/router/vocabulary matrices are cached.
            self._shared_reserve_bytes = sum(
                self.expert_cache._entry_size(layer, 0)
                for layer in range(self.expert_cache.layer_count)
            )

    def _path(self, record: dict[str, Any], key: str = "data_file") -> Path:
        try:
            return self._resolved_paths[str(record[key])]
        except KeyError as exc:
            raise StreamingModelError(f"Manifest record has no validated {key!r} path") from exc

    def record(self, name: str) -> dict[str, Any]:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise StreamingModelError(f"Missing weight tensor {name}") from exc

    def _read_array(self, path: Path, dtype: Any, count: int, offset: int) -> np.ndarray:
        """Read an exact array using a persistent handle local to this thread."""
        handles = getattr(self._thread_files, "handles", None)
        if handles is None:
            handles = {}
            self._thread_files.handles = handles
        handle = handles.get(path)
        if handle is None:
            handle = path.open("rb", buffering=0)
            handles[path] = handle
            with self._file_handles_lock:
                self._file_handles.append(handle)
        values = np.empty(int(count), dtype=dtype)
        target = memoryview(values).cast("B")
        handle.seek(int(offset))
        received = handle.readinto(target)
        if received != target.nbytes:
            raise StreamingModelError(
                f"Short weight read from {path}: expected {target.nbytes} bytes, "
                f"received {received}"
            )
        return values

    def vector(self, name: str) -> torch.Tensor:
        if name not in self._vectors:
            record = self.record(name)
            if record["storage"] != "f32" or len(record["shape"]) != 1:
                raise StreamingModelError(f"{name} is not an FP32 vector")
            values = np.fromfile(self._path(record), dtype=np.float32)
            if values.size != record["shape"][0]:
                raise StreamingModelError(f"Invalid byte length for {name}")
            self.bytes_read += values.nbytes
            self._vectors[name] = torch.from_numpy(values)
            self._vector_bytes += values.nbytes
            self._rebalance_expert_capacity()
        return self._vectors[name]

    def _rebalance_expert_capacity(self) -> None:
        self.expert_cache.set_capacity(
            self.ram_budget_bytes
            - self.buffer_bytes
            - self._vector_bytes
            - self._shared_matrix_bytes
        )

    @staticmethod
    def _resident_matrix_bytes(record: dict[str, Any]) -> int:
        rows, columns = map(int, record["shape"])
        if record["storage"] == "q8_row":
            return rows * columns + rows * np.dtype(np.float32).itemsize
        if record["storage"] == "q4_block":
            padded = int(record["padded_columns"])
            blocks = padded // int(record["block_size"])
            return rows * (padded // 2) + rows * blocks * np.dtype(np.float32).itemsize
        return 0

    def shared_matrix(self, name: str) -> ResidentMatrix | None:
        """Admit a guaranteed-reuse non-expert matrix within the RAM budget."""
        if ".experts." in name:
            return None
        with self._shared_lock:
            cached = self._shared_matrices.get(name)
            if cached is not None:
                return cached
            record = self.record(name)
            required = self._resident_matrix_bytes(record)
            shared_limit = max(
                0,
                self.ram_budget_bytes
                - self.buffer_bytes
                - self._vector_bytes
                - self._shared_reserve_bytes,
            )
            if required <= 0 or self._shared_matrix_bytes + required > shared_limit:
                return None
            # Shrink/evict expert slots before allocating the shared matrix so
            # the configured managed-memory ceiling is never exceeded.
            self._shared_matrix_bytes += required
            self._rebalance_expert_capacity()
            try:
                matrix = self.load_resident_matrix(name)
            except Exception:
                self._shared_matrix_bytes -= required
                self._rebalance_expert_capacity()
                raise
            self._shared_matrix_bytes += matrix.nbytes - required
            self._rebalance_expert_capacity()
            self._shared_matrices[name] = matrix
            return matrix

    def _rows_per_chunk(self, columns: int, storage: str) -> int:
        # Reserve half of the configured budget for the next asynchronously
        # prefetched chunk. The divisor includes packed, unpacked, and kernel
        # input representations; output activations are accounted separately.
        bytes_per_value = 3 if storage == "q4_block" else (3 if self.matmul_mode == "int8" else 5)
        chunk_budget = max(1, self.buffer_bytes // (2 if self.prefetch else 1))
        return max(1, chunk_budget // max(1, columns * bytes_per_value))

    def _pipeline(self, ranges: list[tuple[int, int]], loader: Any):
        if not ranges:
            return
        if not self.prefetch or len(ranges) == 1:
            for start, end in ranges:
                yield start, end, loader(start, end)
            return
        future = self._executor.submit(loader, *ranges[0])
        for index, (start, end) in enumerate(ranges):
            data = future.result()
            if index + 1 < len(ranges):
                future = self._executor.submit(loader, *ranges[index + 1])
            yield start, end, data

    @staticmethod
    def _quantize_activations(
        values: torch.Tensor, limit: float = 127.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scales = values.abs().amax(dim=1) / limit
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        quantized = torch.round(values / scales[:, None]).clamp_(-limit, limit).to(torch.int8)
        return quantized.contiguous(), scales

    @staticmethod
    def _unpack_q4(packed: np.ndarray, padded_columns: int) -> np.ndarray:
        unpacked = np.empty((packed.shape[0], padded_columns), dtype=np.int8)
        # Sign-extend each nibble without allocating a full boolean mask.
        unpacked[:, 0::2] = (packed.astype(np.int8) << 4) >> 4
        unpacked[:, 1::2] = packed.view(np.int8) >> 4
        return unpacked

    def load_resident_matrix(self, name: str) -> ResidentMatrix:
        """Read one compressed matrix completely for admission to the expert cache."""
        record = self.record(name)
        rows, columns = map(int, record["shape"])
        storage = record["storage"]
        if storage == "q8_row":
            data = self._read_array(
                self._path(record),
                np.int8,
                rows * columns,
                int(record.get("data_offset", 0)),
            ).reshape(rows, columns)
            scales = self._read_array(
                self._path(record, "scale_file"),
                np.float32,
                rows,
                int(record.get("scale_offset", 0)),
            ).reshape(rows)
        elif storage == "q4_block":
            padded_columns = int(record["padded_columns"])
            blocks = padded_columns // int(record["block_size"])
            # Cached experts serve single-token decoding, where the direct
            # packed GEMV is substantially faster than tinygemm. Preserve the
            # portable row-major nibbles for that path.
            native_q4 = (
                not self.native_avx512_q4_available
                and not self.numba_q4_available
                and self.native_q4_available
                and "native_data_file" in record
            )
            data = self._read_array(
                self._path(record, "native_data_file" if native_q4 else "data_file"),
                np.uint8,
                rows * (padded_columns // 2),
                int(
                    record.get("native_data_offset", 0)
                    if native_q4
                    else record.get("data_offset", 0)
                ),
            ).reshape(rows, padded_columns // 2)
            scales = self._read_array(
                self._path(record, "scale_file"),
                np.float32,
                rows * blocks,
                int(record.get("scale_offset", 0)),
            ).reshape(rows, blocks)
        else:
            raise StreamingModelError(f"Cannot cache matrix storage {storage!r} for {name}")
        self.bytes_read += data.nbytes + scales.nbytes
        return ResidentMatrix(
            data=data,
            scales=scales,
            native_q4=storage == "q4_block" and native_q4,
        )

    def load_resident_expert(self, layer: int, expert: int) -> dict[str, ResidentMatrix] | None:
        """Load one contiguous row-major Q4 expert with a single file read.

        The converter places gate data/scales, up data/scales, and down
        data/scales consecutively in each layer pack.  Reading each component
        separately costs six file opens and six reads on a cache miss.  When
        that layout is present, retain one byte buffer and expose zero-copy
        NumPy views for all three matrices.  Returning ``None`` selects the
        general per-matrix loader for older or non-contiguous manifests.
        """
        names = {
            projection: ExpertCache.tensor_name(layer, expert, projection)
            for projection in ExpertCache.PROJECTIONS
        }
        records = {projection: self.record(name) for projection, name in names.items()}
        if any(record["storage"] != "q4_block" for record in records.values()):
            return None

        # The portable packed rows are used by AVX-512 and Numba.  PyTorch's
        # native int4pack has a separate file while sharing row-major scales,
        # so it cannot use this single-span representation.
        use_native = any(
            not self.native_avx512_q4_available
            and not self.numba_q4_available
            and self.native_q4_available
            and "native_data_file" in record
            for record in records.values()
        )
        if use_native:
            return None

        segments: list[tuple[int, int, str, str]] = []
        common_path: Path | None = None
        for projection, record in records.items():
            rows, _ = map(int, record["shape"])
            padded_columns = int(record["padded_columns"])
            blocks = padded_columns // int(record["block_size"])
            data_path = self._path(record)
            scale_path = self._path(record, "scale_file")
            if data_path != scale_path:
                return None
            if common_path is None:
                common_path = data_path
            elif data_path != common_path:
                return None
            segments.append(
                (
                    int(record.get("data_offset", 0)),
                    rows * (padded_columns // 2),
                    projection,
                    "data",
                )
            )
            segments.append(
                (
                    int(record.get("scale_offset", 0)),
                    rows * blocks * np.dtype(np.float32).itemsize,
                    projection,
                    "scales",
                )
            )

        ordered = sorted(segments)
        if common_path is None or any(
            ordered[index][0] + ordered[index][1] != ordered[index + 1][0]
            for index in range(len(ordered) - 1)
        ):
            return None
        base = ordered[0][0]
        span = ordered[-1][0] + ordered[-1][1] - base
        blob = self._read_array(common_path, np.uint8, span, base)
        if blob.nbytes != span:
            raise StreamingModelError(
                f"Short expert read for layer {layer}, expert {expert}: "
                f"expected {span} bytes, received {blob.nbytes}"
            )
        self.bytes_read += blob.nbytes

        entry = self._expert_views_from_blob(layer, expert, blob, base)
        if entry is None:
            raise StreamingModelError(
                f"Cannot construct contiguous expert views for layer {layer}, expert {expert}"
            )
        return entry

    def _expert_views_from_blob(
        self, layer: int, expert: int, blob: np.ndarray, base: int
    ) -> dict[str, ResidentMatrix] | None:
        """Expose gate/up/down views backed by an already-read pack span."""
        records = {
            projection: self.record(ExpertCache.tensor_name(layer, expert, projection))
            for projection in ExpertCache.PROJECTIONS
        }
        views: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        for projection, record in records.items():
            if record.get("storage") != "q4_block":
                return None
            rows, _ = map(int, record["shape"])
            padded_columns = int(record["padded_columns"])
            blocks = padded_columns // int(record["block_size"])
            segments = (
                (
                    "data",
                    int(record.get("data_offset", 0)),
                    rows * (padded_columns // 2),
                ),
                (
                    "scales",
                    int(record.get("scale_offset", 0)),
                    rows * blocks * np.dtype(np.float32).itemsize,
                ),
            )
            for kind, offset, length in segments:
                relative = offset - int(base)
                if relative < 0 or relative + length > blob.nbytes:
                    return None
                raw = blob[relative : relative + length]
                if kind == "data":
                    views[projection][kind] = raw.reshape(rows, padded_columns // 2)
                else:
                    views[projection][kind] = raw.view(np.float32).reshape(rows, blocks)

        return {
            projection: ResidentMatrix(
                data=views[projection]["data"],
                scales=views[projection]["scales"],
                native_q4=False,
            )
            for projection in ExpertCache.PROJECTIONS
        }

    def load_resident_expert_group(
        self, layer: int, experts: tuple[int, ...]
    ) -> dict[int, dict[str, ResidentMatrix]]:
        """Read one bounded expert span and retain only selected experts."""
        if not experts:
            return {}
        intervals = []
        for expert in experts:
            interval = self.expert_cache.entry_interval(layer, expert)
            if interval is None:
                raise StreamingModelError(
                    f"No contiguous interval for layer {layer}, expert {expert}"
                )
            intervals.append((int(expert), *interval))
        intervals.sort(key=lambda item: item[2])
        path = intervals[0][1]
        if any(item[1] != path for item in intervals):
            raise StreamingModelError("Expert group spans multiple pack files")
        base = intervals[0][2]
        span = intervals[-1][3] - base
        blob = self._read_array(path, np.uint8, span, base)
        self.bytes_read += blob.nbytes
        result = {}
        for expert, _, start, end in intervals:
            # Give every cache entry its own backing allocation. Otherwise one
            # surviving expert view would retain the entire coalesced span and
            # make managed-memory accounting understate real residency.
            expert_blob = blob[start - base : end - base].copy()
            entry = self._expert_views_from_blob(layer, expert, expert_blob, start)
            if entry is None:
                raise StreamingModelError(
                    f"Cannot construct group view for layer {layer}, expert {expert}"
                )
            result[expert] = entry
        return result

    def linear(
        self,
        inputs: torch.Tensor,
        name: str,
        bias_name: str | None = None,
        resident: ResidentMatrix | None = None,
    ) -> torch.Tensor:
        record = self.record(name)
        if resident is None and record["storage"] in {"q8_row", "q4_block"}:
            resident = self.shared_matrix(name)
        rows, columns = map(int, record["shape"])
        if inputs.shape[-1] != columns:
            raise StreamingModelError(
                f"{name} expects {columns} input features, received {inputs.shape[-1]}"
            )
        original_shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, columns).float().contiguous()
        output = torch.empty((flat.shape[0], rows), dtype=torch.float32)
        storage = record["storage"]
        chunk_rows = self._rows_per_chunk(columns, storage)
        ranges = [(start, min(rows, start + chunk_rows)) for start in range(0, rows, chunk_rows)]

        if storage == "q8_row":
            if resident is None:
                q_values = np.memmap(
                    self._path(record),
                    mode="r",
                    dtype=np.int8,
                    shape=(rows, columns),
                    offset=int(record.get("data_offset", 0)),
                )
                scales = np.memmap(
                    self._path(record, "scale_file"),
                    mode="r",
                    dtype=np.float32,
                    shape=(rows,),
                    offset=int(record.get("scale_offset", 0)),
                )
            else:
                q_values, scales = resident.data, resident.scales

            def load_q8(start: int, end: int):
                if resident is not None:
                    return q_values[start:end], scales[start:end]
                return (
                    np.array(q_values[start:end], dtype=np.int8, copy=True),
                    np.array(scales[start:end], dtype=np.float32, copy=True),
                )

            input_q, input_scales = self._quantize_activations(flat)
            for start, end, (q_np, scale_np) in self._pipeline(ranges, load_q8):
                if resident is None:
                    self.bytes_read += q_np.nbytes + scale_np.nbytes
                weight_q = torch.from_numpy(q_np)
                weight_scales = torch.from_numpy(scale_np)
                if self.matmul_mode == "int8":
                    accumulator = torch._int_mm(input_q, weight_q.t().contiguous())
                    output[:, start:end] = (
                        accumulator.float() * input_scales[:, None] * weight_scales[None, :]
                    )
                else:
                    weights = weight_q.float() * weight_scales[:, None]
                    output[:, start:end] = torch.mm(flat, weights.t())
            del q_values, scales
        elif storage == "q4_block":
            block_size = int(record["block_size"])
            padded_columns = int(record["padded_columns"])
            blocks = padded_columns // block_size
            packed_columns = padded_columns // 2
            if (
                self.native_avx512_q4_available and block_size == 128 and columns == padded_columns
            ) or (self.numba_q4_available and (flat.shape[0] == 1 or columns == padded_columns)):
                if resident is not None and not resident.native_q4:
                    packed_gemv = resident.data
                    scales_gemv = resident.scales
                else:
                    packed_gemv = np.memmap(
                        self._path(record),
                        mode="r",
                        dtype=np.uint8,
                        shape=(rows, packed_columns),
                        offset=int(record.get("data_offset", 0)),
                    )
                    scales_gemv = np.memmap(
                        self._path(record, "scale_file"),
                        mode="r",
                        dtype=np.float32,
                        shape=(rows, blocks),
                        offset=int(record.get("scale_offset", 0)),
                    )
                    self.bytes_read += packed_gemv.nbytes + scales_gemv.nbytes
                if (
                    self.native_avx512_q4_available
                    and block_size == 128
                    and columns == padded_columns
                ):
                    native_inputs = flat.numpy()
                    native_output = np.empty((flat.shape[0], rows), dtype=np.float32)
                    status = _NATIVE_Q4_GEMM(
                        native_inputs.ctypes.data,
                        flat.shape[0],
                        packed_gemv.ctypes.data,
                        scales_gemv.ctypes.data,
                        rows,
                        columns,
                        block_size,
                        native_output.ctypes.data,
                        self.q4_threads,
                    )
                    if status != 0:
                        raise StreamingModelError(
                            f"Native AVX-512 Q4 kernel failed with status {status}"
                        )
                    output.copy_(torch.from_numpy(native_output))
                elif flat.shape[0] > 1:
                    values = _numba_q4_gemm_aligned(
                        flat.numpy(), packed_gemv, scales_gemv, block_size
                    )
                    output.copy_(torch.from_numpy(values))
                elif columns == padded_columns:
                    values = _numba_q4_gemv_aligned(
                        flat[0].numpy(), packed_gemv, scales_gemv, block_size
                    )
                    output[0] = torch.from_numpy(values)
                else:
                    values = _numba_q4_gemv(
                        flat[0].numpy(),
                        packed_gemv,
                        scales_gemv,
                        block_size,
                        columns,
                    )
                    output[0] = torch.from_numpy(values)
                if bias_name is not None:
                    output += self.vector(bias_name)
                return output.reshape(*original_shape, rows)
            native_q4 = self.native_q4_available and (
                (resident is not None and resident.native_q4)
                or (resident is None and "native_data_file" in record)
            )
            if resident is None:
                packed = np.memmap(
                    self._path(record, "native_data_file" if native_q4 else "data_file"),
                    mode="r",
                    dtype=np.uint8,
                    shape=(rows, packed_columns),
                    offset=int(
                        record.get("native_data_offset", 0)
                        if native_q4
                        else record.get("data_offset", 0)
                    ),
                )
                scales = np.memmap(
                    self._path(record, "scale_file"),
                    mode="r",
                    dtype=np.float32,
                    shape=(rows, blocks),
                    offset=int(record.get("scale_offset", 0)),
                )
            else:
                packed, scales = resident.data, resident.scales

            if native_q4:
                # PyTorch's AVX-512 weight-only kernel consumes its own packed
                # INT4 layout directly. Row chunks must begin/end on its 32-row
                # output tile boundary.
                chunk_rows = max(32, (chunk_rows // 32) * 32)
                ranges = [
                    (start, min(rows, start + chunk_rows)) for start in range(0, rows, chunk_rows)
                ]

                def load_native_q4(start: int, end: int):
                    if resident is not None:
                        return packed[start:end], scales[start:end]
                    return (
                        np.array(packed[start:end], dtype=np.uint8, copy=True),
                        np.array(scales[start:end], dtype=np.float32, copy=True),
                    )

                padded_inputs = F.pad(flat, (0, padded_columns - columns))
                for start, end, (packed_np, scale_np) in self._pipeline(ranges, load_native_q4):
                    if resident is None:
                        self.bytes_read += packed_np.nbytes + scale_np.nbytes
                    weight_packed = torch.from_numpy(packed_np)
                    weight_scales = torch.from_numpy(scale_np)
                    transposed_scales = weight_scales.t().contiguous()
                    scales_and_zeros = torch.stack(
                        (transposed_scales, torch.zeros_like(transposed_scales)),
                        dim=2,
                    ).contiguous()
                    output[:, start:end] = torch._weight_int4pack_mm_for_cpu(
                        padded_inputs,
                        weight_packed,
                        block_size,
                        scales_and_zeros,
                    )
                del packed, scales
                if bias_name is not None:
                    output += self.vector(bias_name)
                return output.reshape(*original_shape, rows)

            def load_q4(start: int, end: int):
                if resident is not None:
                    return packed[start:end], scales[start:end]
                packed_np = np.array(packed[start:end], dtype=np.uint8, copy=True)
                return (
                    packed_np,
                    np.array(scales[start:end], dtype=np.float32, copy=True),
                )

            padded_inputs = F.pad(flat, (0, padded_columns - columns))
            blocked_inputs = padded_inputs.reshape(flat.shape[0], blocks, block_size)
            input_q = torch.empty_like(blocked_inputs, dtype=torch.int8)
            input_scales = torch.empty((flat.shape[0], blocks), dtype=torch.float32)
            for block in range(blocks):
                input_q[:, block, :], input_scales[:, block] = self._quantize_activations(
                    blocked_inputs[:, block, :]
                )
            for start, end, (packed_np, scale_np) in self._pipeline(ranges, load_q4):
                if resident is None:
                    self.bytes_read += packed_np.nbytes + scale_np.nbytes
                unpacked_np = self._unpack_q4(packed_np, padded_columns)
                weight_q = torch.from_numpy(unpacked_np).reshape(end - start, blocks, block_size)
                weight_scales = torch.from_numpy(scale_np)
                if self.matmul_mode == "int8":
                    chunk_output = torch.zeros((flat.shape[0], end - start), dtype=torch.float32)
                    for block in range(blocks):
                        accumulator = torch._int_mm(
                            input_q[:, block, :].contiguous(),
                            weight_q[:, block, :].t().contiguous(),
                        )
                        chunk_output += (
                            accumulator.float()
                            * input_scales[:, block, None]
                            * weight_scales[None, :, block]
                        )
                    output[:, start:end] = chunk_output
                else:
                    expanded_scales = weight_scales.repeat_interleave(block_size, dim=1)
                    weights = weight_q.reshape(end - start, padded_columns).float()
                    weights *= expanded_scales
                    output[:, start:end] = torch.mm(padded_inputs, weights.t())
            del packed, scales
        else:
            raise StreamingModelError(f"Unsupported matrix storage {storage!r} for {name}")
        if bias_name is not None:
            output += self.vector(bias_name)
        return output.reshape(*original_shape, rows)

    def linear_pair(
        self,
        inputs: torch.Tensor,
        first_name: str,
        second_name: str,
        first_resident: ResidentMatrix | None = None,
        second_resident: ResidentMatrix | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse equal-shaped Q4 gate/up projections, otherwise use linear()."""
        first_record = self.record(first_name)
        second_record = self.record(second_name)
        rows, columns = map(int, first_record["shape"])
        flat = inputs.reshape(-1, inputs.shape[-1]).float().contiguous()
        base_compatible = (
            first_record["storage"] == "q4_block"
            and second_record["storage"] == "q4_block"
            and second_record["shape"] == first_record["shape"]
            and int(first_record["padded_columns"]) == columns
            and int(second_record["padded_columns"]) == columns
            and int(second_record["block_size"]) == int(first_record["block_size"])
        )
        native_compatible = (
            base_compatible
            and self.native_avx512_q4_available
            and _NATIVE_Q4_GEMM_PAIR is not None
            and int(first_record["block_size"]) == 128
        )
        numba_compatible = base_compatible and self.numba_q4_available and flat.shape[0] == 1
        if not native_compatible and not numba_compatible:
            return (
                self.linear(inputs, first_name, resident=first_resident),
                self.linear(inputs, second_name, resident=second_resident),
            )
        if inputs.shape[-1] != columns:
            raise StreamingModelError(
                f"{first_name} expects {columns} input features, received {inputs.shape[-1]}"
            )

        block_size = int(first_record["block_size"])
        padded_columns = int(first_record["padded_columns"])
        packed_columns = padded_columns // 2
        blocks = padded_columns // block_size

        def arrays(
            record: dict[str, Any], resident: ResidentMatrix | None
        ) -> tuple[np.ndarray, np.ndarray]:
            if resident is not None and not resident.native_q4:
                return resident.data, resident.scales
            packed = np.memmap(
                self._path(record),
                mode="r",
                dtype=np.uint8,
                shape=(rows, packed_columns),
                offset=int(record.get("data_offset", 0)),
            )
            scales = np.memmap(
                self._path(record, "scale_file"),
                mode="r",
                dtype=np.float32,
                shape=(rows, blocks),
                offset=int(record.get("scale_offset", 0)),
            )
            self.bytes_read += packed.nbytes + scales.nbytes
            return packed, scales

        first_packed, first_scales = arrays(first_record, first_resident)
        second_packed, second_scales = arrays(second_record, second_resident)
        if native_compatible:
            first = np.empty((flat.shape[0], rows), dtype=np.float32)
            second = np.empty((flat.shape[0], rows), dtype=np.float32)
            status = _NATIVE_Q4_GEMM_PAIR(
                flat.numpy().ctypes.data,
                flat.shape[0],
                first_packed.ctypes.data,
                first_scales.ctypes.data,
                second_packed.ctypes.data,
                second_scales.ctypes.data,
                rows,
                columns,
                block_size,
                first.ctypes.data,
                second.ctypes.data,
                self.q4_threads,
            )
            if status != 0:
                raise StreamingModelError(
                    f"Native AVX-512 paired Q4 kernel failed with status {status}"
                )
        else:
            first, second = _numba_q4_gemv_aligned_pair(
                flat[0].numpy(),
                first_packed,
                first_scales,
                second_packed,
                second_scales,
                block_size,
            )
        output_shape = (*inputs.shape[:-1], rows)
        return (
            torch.from_numpy(first).reshape(output_shape),
            torch.from_numpy(second).reshape(output_shape),
        )

    def embedding(self, token_ids: list[int]) -> torch.Tensor:
        name = "model.embed_tokens.weight"
        record = self.record(name)
        rows, columns = map(int, record["shape"])
        if any(token_id < 0 or token_id >= rows for token_id in token_ids):
            raise StreamingModelError("Input contains a token outside the embedding vocabulary")
        if record["storage"] == "q8_row":
            q_values = np.memmap(self._path(record), mode="r", dtype=np.int8, shape=(rows, columns))
            scales = np.memmap(
                self._path(record, "scale_file"), mode="r", dtype=np.float32, shape=(rows,)
            )
            q_rows_np = np.array(q_values[token_ids], dtype=np.int8, copy=True)
            scale_rows_np = np.array(scales[token_ids], dtype=np.float32, copy=True)
            self.bytes_read += q_rows_np.nbytes + scale_rows_np.nbytes
            result = torch.from_numpy(q_rows_np).float() * torch.from_numpy(scale_rows_np)[:, None]
            del q_values, scales
            return result
        if record["storage"] == "q4_block":
            block_size = int(record["block_size"])
            padded_columns = int(record["padded_columns"])
            blocks = padded_columns // block_size
            packed_columns = padded_columns // 2
            packed = np.memmap(
                self._path(record), mode="r", dtype=np.uint8, shape=(rows, packed_columns)
            )
            scales = np.memmap(
                self._path(record, "scale_file"), mode="r", dtype=np.float32, shape=(rows, blocks)
            )
            packed_np = np.array(packed[token_ids], dtype=np.uint8, copy=True)
            scale_np = np.array(scales[token_ids], dtype=np.float32, copy=True)
            self.bytes_read += packed_np.nbytes + scale_np.nbytes
            unpacked = torch.from_numpy(self._unpack_q4(packed_np, padded_columns)).float()
            unpacked = unpacked.reshape(len(token_ids), blocks, block_size)
            unpacked *= torch.from_numpy(scale_np)[:, :, None]
            del packed, scales
            return unpacked.reshape(len(token_ids), padded_columns)[:, :columns]
        raise StreamingModelError(f"Unsupported embedding storage {record['storage']!r}")

    def project_vocabulary(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project with a tied embedding or a separate language-model head."""
        name = (
            "model.embed_tokens.weight"
            if self.manifest["config"].get("tie_word_embeddings", False)
            else "lm_head.weight"
        )
        # Older Qwen2 manifests omitted lm_head because their weights are tied.
        if name not in self.tensors:
            name = "model.embed_tokens.weight"
        return self.linear(hidden, name)

    def close(self) -> None:
        self.expert_cache.clear()
        self._shared_matrices.clear()
        self._shared_matrix_bytes = 0
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._prediction_executor.shutdown(wait=True, cancel_futures=True)
        with self._file_handles_lock:
            for handle in self._file_handles:
                try:
                    handle.close()
                except OSError:
                    pass
            self._file_handles.clear()


class DiskKVCache:
    """A float16, SSD-backed cache; only one layer is copied into RAM at once."""

    def __init__(
        self,
        path: str | Path,
        layers: int,
        kv_heads: int,
        max_context: int,
        head_dim: int,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.layers = layers
        self.kv_heads = kv_heads
        self.max_context = max_context
        self.head_dim = head_dim
        self.length = 0
        self._shape = (layers, 2, kv_heads, max_context, head_dim)
        self._values = np.memmap(self.path, mode="w+", dtype=np.float16, shape=self._shape)

    @property
    def allocated_bytes(self) -> int:
        return int(np.prod(self._shape)) * np.dtype(np.float16).itemsize

    def write(self, layer: int, start: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        # keys/values: [kv_heads, sequence, head_dim]
        end = start + keys.shape[1]
        if end > self.max_context:
            raise StreamingModelError(
                f"Context length {end} exceeds configured maximum {self.max_context}"
            )
        self._values[layer, 0, :, start:end, :] = (
            keys.detach().cpu().numpy().astype(np.float16, copy=False)
        )
        self._values[layer, 1, :, start:end, :] = (
            values.detach().cpu().numpy().astype(np.float16, copy=False)
        )

    def read(self, layer: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        keys = np.array(self._values[layer, 0, :, :length, :], dtype=np.float32, copy=True)
        values = np.array(self._values[layer, 1, :, :length, :], dtype=np.float32, copy=True)
        return torch.from_numpy(keys), torch.from_numpy(values)

    def close(self) -> None:
        if getattr(self, "_values", None) is not None:
            self._values.flush()
            mmap = getattr(self._values, "_mmap", None)
            if mmap is not None:
                mmap.close()
            self._values = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class StreamedQwenModel:
    """Manual Qwen2/Qwen3-MoE decoder with bounded weight residency."""

    def __init__(self, store: WeightStore):
        self.store = store
        self.config = store.manifest["config"]
        self.architecture = store.manifest.get("architecture")
        if self.architecture not in {"qwen2", "qwen3_moe"}:
            raise StreamingModelError("Only qwen2 and qwen3_moe architectures are supported")
        self.hidden_size = int(self.config["hidden_size"])
        self.intermediate_size = int(self.config["intermediate_size"])
        self.layers = int(self.config["num_hidden_layers"])
        self.heads = int(self.config["num_attention_heads"])
        self.kv_heads = int(self.config["num_key_value_heads"])
        self.head_dim = int(self.config.get("head_dim", self.hidden_size // self.heads))
        self.query_size = self.heads * self.head_dim
        self.kv_groups = self.heads // self.kv_heads
        self.rms_eps = float(self.config["rms_norm_eps"])
        self.rope_theta = float(self.config.get("rope_theta", 10000.0))
        self.max_positions = int(self.config["max_position_embeddings"])
        if self.heads % self.kv_heads:
            raise StreamingModelError("Attention head count must be divisible by KV head count")
        if self.architecture == "qwen3_moe":
            self.num_experts = int(self.config["num_experts"])
            self.experts_per_token = int(self.config["num_experts_per_tok"])
            self.normalize_topk = bool(self.config.get("norm_topk_prob", True))
            self._last_routes: dict[int, tuple[int, ...]] = {}
            self._route_history: dict[int, deque[tuple[int, ...]]] = defaultdict(
                lambda: deque(maxlen=3)
            )
            self._route_stability: dict[int, dict[str, int]] = {
                depth: {
                    "observations": 0,
                    "candidates": 0,
                    "correct": 0,
                    "limited_candidates": 0,
                    "limited_correct": 0,
                    "targets": 0,
                }
                for depth in (1, 2, 3)
            }
            self._prefill_layout: dict[str, Any] = {}
            self.reset_prefill_layout_diagnostics()
            self.profile_prefill_layout = False
            self.coalesce_prefill = False
            self.prefill_coalescing_gap = 0
            self.predict_experts = False

    def reset_prefill_layout_diagnostics(self) -> None:
        if self.architecture != "qwen3_moe":
            return
        self._prefill_layout = {
            "layers": 0,
            "selected_experts": 0,
            "exact_bytes": 0,
            "full_span_bytes": 0,
            "policies": {str(gap): {"reads": 0, "bytes": 0} for gap in (0, 1, 2, 4, 8)},
        }

    def prefill_layout_stats(self) -> dict[str, Any]:
        if self.architecture != "qwen3_moe":
            return {}
        exact = int(self._prefill_layout["exact_bytes"])
        selected = int(self._prefill_layout["selected_experts"])
        result = dict(self._prefill_layout)
        result["full_span_amplification"] = int(result["full_span_bytes"]) / exact if exact else 0.0
        result["policies"] = {
            gap: {
                **values,
                "read_fraction": values["reads"] / selected if selected else 0.0,
                "amplification": values["bytes"] / exact if exact else 0.0,
            }
            for gap, values in self._prefill_layout["policies"].items()
        }
        return result

    def _record_prefill_layout(self, layer: int, experts: list[int]) -> None:
        """Measure coalescing cost for exact uncached experts without reading."""
        intervals: list[tuple[int, int, int]] = []
        common_path: Path | None = None
        for expert in sorted(set(map(int, experts))):
            interval = self.store.expert_cache.entry_interval(layer, expert)
            if interval is None:
                return
            path, start, end = interval
            if common_path is None:
                common_path = path
            elif path != common_path:
                return
            intervals.append((expert, start, end))
        if not intervals:
            return

        exact = sum(end - start for _, start, end in intervals)
        self._prefill_layout["layers"] += 1
        self._prefill_layout["selected_experts"] += len(intervals)
        self._prefill_layout["exact_bytes"] += exact
        self._prefill_layout["full_span_bytes"] += intervals[-1][2] - intervals[0][1]
        for gap_limit in (0, 1, 2, 4, 8):
            reads = 1
            read_bytes = 0
            group_expert, group_start, group_end = intervals[0]
            for expert, start, end in intervals[1:]:
                if expert - group_expert - 1 <= gap_limit:
                    group_end = end
                else:
                    read_bytes += group_end - group_start
                    reads += 1
                    group_start, group_end = start, end
                group_expert = expert
            read_bytes += group_end - group_start
            policy = self._prefill_layout["policies"][str(gap_limit)]
            policy["reads"] += reads
            policy["bytes"] += read_bytes

    def reset_route_diagnostics(self, *, clear_history: bool = False) -> None:
        if self.architecture != "qwen3_moe":
            return
        for values in self._route_stability.values():
            for key in values:
                values[key] = 0
        if clear_history:
            self._route_history.clear()

    def route_stability_stats(self) -> dict[str, dict[str, float | int]]:
        if self.architecture != "qwen3_moe":
            return {}
        result: dict[str, dict[str, float | int]] = {}
        for depth, values in self._route_stability.items():
            candidates = int(values["candidates"])
            correct = int(values["correct"])
            targets = int(values["targets"])
            observations = int(values["observations"])
            result[str(depth)] = {
                **values,
                "precision": correct / candidates if candidates else 0.0,
                "top2_precision": (
                    int(values["limited_correct"]) / int(values["limited_candidates"])
                    if values["limited_candidates"]
                    else 0.0
                ),
                "recall": correct / targets if targets else 0.0,
                "candidates_per_layer": (candidates / observations if observations else 0.0),
            }
        return result

    def _record_route_stability(self, layer: int, experts: tuple[int, ...]) -> None:
        """Measure whether intersections of recent routes predict this route."""
        history = self._route_history[layer]
        current = set(experts)
        for depth in (1, 2, 3):
            if len(history) < depth:
                continue
            candidate = set(history[-1])
            for offset in range(2, depth + 1):
                candidate.intersection_update(history[-offset])
            ranked = [expert for expert in history[-1] if expert in candidate]
            limited = set(ranked[:2])
            values = self._route_stability[depth]
            values["observations"] += 1
            values["candidates"] += len(candidate)
            values["correct"] += len(candidate & current)
            values["limited_candidates"] += len(limited)
            values["limited_correct"] += len(limited & current)
            values["targets"] += len(current)
        history.append(experts)

    def _stable_predictions(self, layer: int, *, depth: int = 2, limit: int = 2) -> tuple[int, ...]:
        """Return recent high-ranked routes stable across ``depth`` tokens."""
        history = self._route_history.get(int(layer))
        if history is None or len(history) < depth:
            return ()
        stable = set(history[-1])
        for offset in range(2, depth + 1):
            stable.intersection_update(history[-offset])
        return tuple(expert for expert in history[-1] if expert in stable)[:limit]

    def create_cache(self, path: str | Path, max_context: int) -> DiskKVCache:
        return DiskKVCache(path, self.layers, self.kv_heads, max_context, self.head_dim)

    def _rms_norm(self, hidden: torch.Tensor, weight_name: str) -> torch.Tensor:
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden.float() * torch.rsqrt(variance + self.rms_eps)
        return normalized * self.store.vector(weight_name)

    @staticmethod
    def _rotate_half(values: torch.Tensor) -> torch.Tensor:
        first, second = values.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def _apply_rope(
        self, query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        frequencies = torch.outer(positions.float(), inv_freq)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cosine = embedding.cos().unsqueeze(0)
        sine = embedding.sin().unsqueeze(0)
        return (
            query * cosine + self._rotate_half(query) * sine,
            key * cosine + self._rotate_half(key) * sine,
        )

    def _attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        start_position: int,
    ) -> torch.Tensor:
        keys = keys.repeat_interleave(self.kv_groups, dim=0)
        values = values.repeat_interleave(self.kv_groups, dim=0)
        scores = torch.matmul(query, keys.transpose(1, 2)) / math.sqrt(self.head_dim)
        query_positions = torch.arange(
            start_position, start_position + query.shape[1], dtype=torch.long
        )
        key_positions = torch.arange(keys.shape[1], dtype=torch.long)
        causal = key_positions[None, :] > query_positions[:, None]
        scores.masked_fill_(causal.unsqueeze(0), -torch.inf)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        attended = torch.matmul(probabilities, values)
        return attended.transpose(0, 1).contiguous().reshape(query.shape[1], self.query_size)

    def _moe(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        """Route tokens and evaluate only selected experts for one sparse layer."""
        prefix = f"model.layers.{layer}.mlp"
        router_logits = self.store.linear(hidden, f"{prefix}.gate.weight")
        probabilities = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected = torch.topk(probabilities, self.experts_per_token, dim=-1)
        if self.normalize_topk:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        result = torch.zeros_like(hidden, dtype=torch.float32)
        expert_counts = {
            int(expert): int((selected == expert).sum())
            for expert in torch.unique(selected).tolist()
        }
        # During batched prefill, admit the experts serving the most prompt
        # tokens first. This fills each fixed layer quota with useful routes
        # while every selected matrix is already being read for computation.
        unique_experts = sorted(expert_counts, key=lambda expert: (-expert_counts[expert], expert))
        if hidden.shape[0] > 1 and self.profile_prefill_layout:
            uncached = self.store.expert_cache.uncached_experts(layer, unique_experts)
            self._record_prefill_layout(layer, uncached)
        if hidden.shape[0] == 1:
            # torch.topk is weight-ranked; preserve that order so a limited
            # stability predictor chooses the strongest recurring routes.
            routes = tuple(int(expert) for expert in selected[0].tolist())
            self._record_route_stability(layer, routes)
            self.store.expert_cache.cancel_layer_predictions_except(layer, unique_experts)
            self._last_routes[layer] = routes

        # Exact routing is available before expert computation begins for both
        # decode and layer-major batched prefill. Keep a bounded window of
        # guaranteed-use reads ahead of computation. This lets batched prefill
        # benefit from --io-workers without allowing a layer containing many
        # prompt routes to materialize every expert outside the managed cache.
        batched_prefill = hidden.shape[0] > 1
        coalesced_prefill = batched_prefill and self.coalesce_prefill
        prefetch_window = (
            min(32, max(8, self.store.io_workers * 4))
            if coalesced_prefill
            else max(2, self.store.io_workers * 2)
        )
        if coalesced_prefill:
            self.store.expert_cache.prefetch_coalesced(
                layer,
                unique_experts[:prefetch_window],
                gap_experts=self.prefill_coalescing_gap,
            )
        else:
            self.store.expert_cache.prefetch(
                layer, unique_experts[:prefetch_window], predicted=False
            )
        if hidden.shape[0] == 1 and self.predict_experts and layer + 1 < self.layers:
            # Predict the next layer while this layer's MoE compute is still
            # ahead of us, rather than waiting for next-layer attention. Only
            # the two strongest routes stable across two prior tokens qualify.
            predicted = self._stable_predictions(layer + 1, depth=2, limit=2)
            if predicted:
                self.store.expert_cache.prefetch(layer + 1, predicted, predicted=True)

        # Group tokens by expert. One cache lookup and one set of matrix calls
        # serves every token routed to that expert in this batch.
        for expert_index, expert in enumerate(unique_experts):
            token_indices, slots = torch.where(selected == expert)
            current = hidden[token_indices]
            resident = self.store.expert_cache.get(layer, expert, frequency=expert_counts[expert])
            if coalesced_prefill:
                chunk_offset = expert_index % prefetch_window
                if chunk_offset == prefetch_window // 2 - 1:
                    next_start = (expert_index // prefetch_window + 1) * prefetch_window
                    self.store.expert_cache.prefetch_coalesced(
                        layer,
                        unique_experts[next_start : next_start + prefetch_window],
                        gap_experts=self.prefill_coalescing_gap,
                    )
            else:
                next_index = expert_index + prefetch_window
                if next_index < len(unique_experts):
                    self.store.expert_cache.prefetch(
                        layer, [unique_experts[next_index]], predicted=False
                    )

            def matrix(
                projection: str,
                resident_expert: dict[str, ResidentMatrix] | None = resident,
            ) -> ResidentMatrix | None:
                return None if resident_expert is None else resident_expert[projection]

            expert_prefix = f"{prefix}.experts.{expert}"
            gate, up = self.store.linear_pair(
                current,
                f"{expert_prefix}.gate_proj.weight",
                f"{expert_prefix}.up_proj.weight",
                first_resident=matrix("gate_proj"),
                second_resident=matrix("up_proj"),
            )
            projected = self.store.linear(
                F.silu(gate) * up,
                f"{expert_prefix}.down_proj.weight",
                resident=matrix("down_proj"),
            )
            weighted = projected * routing_weights[token_indices, slots, None]
            result.index_add_(0, token_indices, weighted.float())
        return result.to(hidden.dtype)

    def forward(self, token_ids: list[int], cache: DiskKVCache) -> torch.Tensor:
        if not token_ids:
            raise StreamingModelError("Cannot evaluate an empty token sequence")
        start = cache.length
        end = start + len(token_ids)
        if end > min(cache.max_context, self.max_positions):
            raise StreamingModelError(f"Context length {end} exceeds the model/runtime limit")
        hidden = self.store.embedding(token_ids)
        positions = torch.arange(start, end, dtype=torch.long)

        with torch.inference_mode():
            for layer in range(self.layers):
                prefix = f"model.layers.{layer}"
                if (
                    self.architecture == "qwen3_moe"
                    and len(token_ids) == 1
                    and self.predict_experts
                    and layer == 0
                ):
                    # Layer zero has no preceding MoE computation to cover its
                    # prediction, so launch its stable routes before attention.
                    predicted = self._stable_predictions(0, depth=2, limit=2)
                    if predicted:
                        self.store.expert_cache.prefetch(0, predicted, predicted=True)
                residual = hidden
                normalized = self._rms_norm(hidden, f"{prefix}.input_layernorm.weight")
                q_bias = f"{prefix}.self_attn.q_proj.bias"
                k_bias = f"{prefix}.self_attn.k_proj.bias"
                v_bias = f"{prefix}.self_attn.v_proj.bias"
                query = self.store.linear(
                    normalized,
                    f"{prefix}.self_attn.q_proj.weight",
                    q_bias if q_bias in self.store.tensors else None,
                )
                key = self.store.linear(
                    normalized,
                    f"{prefix}.self_attn.k_proj.weight",
                    k_bias if k_bias in self.store.tensors else None,
                )
                value = self.store.linear(
                    normalized,
                    f"{prefix}.self_attn.v_proj.weight",
                    v_bias if v_bias in self.store.tensors else None,
                )
                query = query.reshape(len(token_ids), self.heads, self.head_dim).transpose(0, 1)
                key = key.reshape(len(token_ids), self.kv_heads, self.head_dim).transpose(0, 1)
                value = value.reshape(len(token_ids), self.kv_heads, self.head_dim).transpose(0, 1)
                if self.architecture == "qwen3_moe":
                    query = self._rms_norm(query, f"{prefix}.self_attn.q_norm.weight")
                    key = self._rms_norm(key, f"{prefix}.self_attn.k_norm.weight")
                query, key = self._apply_rope(query, key, positions)
                cache.write(layer, start, key, value)
                all_keys, all_values = cache.read(layer, end)
                attended = self._attention(query, all_keys, all_values, start)
                hidden = residual + self.store.linear(attended, f"{prefix}.self_attn.o_proj.weight")

                residual = hidden
                normalized = self._rms_norm(hidden, f"{prefix}.post_attention_layernorm.weight")
                if self.architecture == "qwen3_moe":
                    hidden = residual + self._moe(normalized, layer)
                else:
                    gate = self.store.linear(normalized, f"{prefix}.mlp.gate_proj.weight")
                    up = self.store.linear(normalized, f"{prefix}.mlp.up_proj.weight")
                    hidden = residual + self.store.linear(
                        F.silu(gate) * up, f"{prefix}.mlp.down_proj.weight"
                    )
                    del gate, up
                del residual, normalized, query, key, value, all_keys, all_values, attended

            hidden = self._rms_norm(hidden, "model.norm.weight")
            logits = self.store.project_vocabulary(hidden[-1:])[0]
        cache.length = end
        return logits


class StreamingBackend:
    """TokenQL backend API implemented by the manual streamed executor."""

    backend_name = "streamed-q8"

    def __init__(
        self,
        model_dir: str | Path,
        weight_buffer_mb: int = 64,
        ram_budget_mb: int | None = None,
        max_context: int = 4096,
        kv_cache_dir: str | Path | None = None,
        matmul: str = "auto",
        prefetch: bool = True,
        thinking: bool = False,
        expert_prediction: bool = False,
        prefill_layout_profile: bool = False,
        prefill_coalescing: bool = False,
        prefill_coalescing_gap: int = 0,
        io_workers: int = 1,
    ):
        self.store = WeightStore(
            model_dir,
            weight_buffer_mb,
            matmul,
            prefetch,
            ram_budget_mb=ram_budget_mb,
            io_workers=io_workers,
        )
        self.runtime = StreamedQwenModel(self.store)
        if self.runtime.architecture == "qwen3_moe":
            self.runtime.predict_experts = bool(expert_prediction)
            self.runtime.profile_prefill_layout = bool(prefill_layout_profile)
            self.runtime.coalesce_prefill = bool(prefill_coalescing)
            self.runtime.prefill_coalescing_gap = max(0, int(prefill_coalescing_gap))
        quantization = str(self.store.manifest.get("quantization", "streamed"))
        bits = "q4" if quantization.startswith("q4") else "q8"
        native_suffix = ""
        if bits == "q4" and self.store.native_avx512_q4_available:
            native_suffix = "-avx512"
        elif bits == "q4" and self.store.numba_q4_available:
            native_suffix = "-gemv"
        elif bits == "q4" and self.store.native_q4_available and "native_q4" in self.store.manifest:
            native_suffix = "-native"
        self.backend_name = (
            f"paged-moe-{bits}{native_suffix}"
            if self.runtime.architecture == "qwen3_moe"
            else f"streamed-{bits}{native_suffix}"
        )
        self.model_name = self.store.manifest.get("source_model", str(model_dir))
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.store.model_dir), local_files_only=True
        )
        self.thinking_enabled = bool(thinking)
        self.max_context = min(int(max_context), self.runtime.max_positions)
        if self.max_context < 1:
            raise StreamingModelError("max_context must be positive")
        self._caches: list[DiskKVCache] = []
        self._owns_cache_dir = kv_cache_dir is None
        if kv_cache_dir is None:
            self.kv_cache_dir = Path(tempfile.mkdtemp(prefix="tokenql-kv-"))
        else:
            self.kv_cache_dir = Path(kv_cache_dir).resolve()
            self.kv_cache_dir.mkdir(parents=True, exist_ok=True)
        self._closed = False
        atexit.register(self.close)

    def set_thinking(self, enabled: bool) -> None:
        self.thinking_enabled = bool(enabled)

    def _format_messages(self, messages: list[dict[str, str]]) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=self.thinking_enabled,
        )
        if ids.ndim == 2:
            ids = ids[0]
        return ids.tolist()

    def _format_prompt(self, prompt: str) -> list[int]:
        return self._format_messages(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )

    def _new_cache(self, session_id: str) -> DiskKVCache:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)[:50] or "session"
        path = self.kv_cache_dir / f"{safe_name}-{uuid.uuid4().hex}.kv.f16"
        cache = self.runtime.create_cache(path, self.max_context)
        self._caches.append(cache)
        return cache

    def _discard_cache(self, cache: DiskKVCache) -> None:
        cache.close()
        try:
            cache.path.unlink(missing_ok=True)
        except OSError:
            pass
        if cache in self._caches:
            self._caches.remove(cache)

    def release(self, session: Any) -> None:
        if session.cache is not None and isinstance(session.cache, DiskKVCache):
            self._discard_cache(session.cache)
        session.cache = None
        session.initialized = False

    def close(self) -> None:
        if self._closed:
            return
        for cache in self._caches:
            cache.close()
        self._caches.clear()
        self.store.close()
        if self._owns_cache_dir:
            shutil.rmtree(self.kv_cache_dir, ignore_errors=True)
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def initialize(self, session: Any, rebuild: bool = False) -> None:
        if session.initialized and not rebuild:
            return
        if session.cache is not None and isinstance(session.cache, DiskKVCache):
            self._discard_cache(session.cache)
        session.prompt_ids = (
            self._format_messages(session.messages)
            if session.messages is not None
            else self._format_prompt(session.prompt)
        )
        all_ids = session.prompt_ids + session.generated_ids
        session.cache = self._new_cache(session.session_id)
        session.decode_seconds = 0.0
        session.decode_steps = 0
        if len(all_ids) > 1 and self.runtime.architecture == "qwen3_moe":
            self.runtime._last_routes.clear()
            self.runtime.reset_route_diagnostics(clear_history=True)
            self.runtime.reset_prefill_layout_diagnostics()
        started = time.perf_counter()
        session.next_logits = self.runtime.forward(all_ids, session.cache)
        session.prefill_seconds = time.perf_counter() - started
        session.prefill_tokens = len(all_ids)
        cache_stats = self.store.expert_cache.stats()
        session.cache_hits_after_prefill = int(cache_stats["hits"])
        session.cache_misses_after_prefill = int(cache_stats["misses"])
        session.prefetch_wait_seconds_after_prefill = float(
            cache_stats["prefetch"].get("wait_seconds", 0.0)
        )
        session.initialized = True

    def prepare_continuation(self, session: Any, messages: list[dict[str, str]]) -> bool:
        """Append a chat turn while reusing the unchanged KV-cache prefix."""
        if not session.initialized or not isinstance(session.cache, DiskKVCache):
            session.messages = messages
            return False

        previous_messages = session.messages or []
        previous_reply = self.text(session)
        is_append_only = (
            len(messages) == len(previous_messages) + 2
            and messages[:-2] == previous_messages
            and messages[-2].get("role") == "assistant"
            and messages[-2].get("content") == previous_reply
            and messages[-1].get("role") == "user"
        )
        previous_ids = session.prompt_ids + session.generated_ids
        reused = int(session.cache.length)
        can_reuse = is_append_only and reused > 0 and reused <= len(previous_ids)
        if not can_reuse:
            self.release(session)
            session.messages = messages
            session.prompt_ids = []
            session.generated_ids = []
            session.finished = False
            session.next_logits = None
            return False

        # Qwen's thinking-disabled generation prompt includes an empty
        # <think></think> prefix, but apply_chat_template omits it when the
        # same answer later appears as a historical assistant message. Keep
        # the exact evaluated prefix and derive only the syntax after an empty
        # assistant terminator, avoiding that asymmetric re-render.
        probe = self._format_messages(
            [
                {"role": "assistant", "content": ""},
                {"role": "user", "content": messages[-1]["content"]},
            ]
        )
        eos_positions = [
            index for index, token_id in enumerate(probe) if token_id in self.eos_ids()
        ]
        if not eos_positions:
            self.release(session)
            session.messages = messages
            session.generated_ids = []
            return False
        turn_suffix = probe[eos_positions[0] + 1 :]
        next_prompt_ids = previous_ids + turn_suffix
        suffix = previous_ids[reused:] + turn_suffix
        session.messages = messages
        session.prompt = messages[-1]["content"] if messages else ""
        session.prompt_ids = next_prompt_ids
        session.generated_ids = []
        session.finished = False
        session.next_logits = None
        session.prefill_seconds = 0.0
        session.prefill_tokens = len(suffix)
        session.decode_seconds = 0.0
        session.decode_steps = 0

        if len(suffix) > 1 and self.runtime.architecture == "qwen3_moe":
            # Batched prompt routing is not a predecessor for single-token
            # route stability or previous-token prediction.
            self.runtime._last_routes.clear()
            self.runtime.reset_route_diagnostics(clear_history=True)
            self.runtime.reset_prefill_layout_diagnostics()

        started = time.perf_counter()
        session.next_logits = self.runtime.forward(suffix, session.cache)
        session.prefill_seconds = time.perf_counter() - started
        cache_stats = self.store.expert_cache.stats()
        session.cache_hits_after_prefill = int(cache_stats["hits"])
        session.cache_misses_after_prefill = int(cache_stats["misses"])
        session.prefetch_wait_seconds_after_prefill = float(
            cache_stats["prefetch"].get("wait_seconds", 0.0)
        )
        session.initialized = True
        return True

    def eos_ids(self) -> set[int]:
        value = self.runtime.config.get("eos_token_id", self.tokenizer.eos_token_id)
        return set(value if isinstance(value, list) else [value])

    def distribution(
        self,
        session: Any,
        *,
        strategy: str = "sample",
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        self.initialize(session)
        if strategy not in {"sample", "greedy"}:
            raise StreamingModelError("strategy must be 'sample' or 'greedy'")
        if temperature <= 0:
            raise StreamingModelError("temperature must be greater than zero")
        if not 0 < top_p <= 1:
            raise StreamingModelError("top_p must be in the interval (0, 1]")
        if top_k < 0:
            raise StreamingModelError("top_k cannot be negative")
        logits = session.next_logits.float().clone()
        if strategy == "sample":
            logits /= temperature
            if top_k:
                keep = min(int(top_k), logits.numel())
                cutoff = torch.topk(logits, keep).values[-1]
                logits[logits < cutoff] = -torch.inf
            if top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cumulative > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                logits[sorted_indices[remove]] = -torch.inf
        return torch.softmax(logits, dim=-1)

    def candidates(self, session: Any, count: int, **options: Any) -> list[dict[str, Any]]:
        probabilities = self.distribution(session, **options)
        values, indices = torch.topk(probabilities, min(int(count), probabilities.numel()))
        return [
            self.describe_token(int(index), float(value))
            for value, index in zip(values, indices, strict=False)
        ]

    def choose(self, session: Any, *, seed: int | None = None, **options: Any) -> dict[str, Any]:
        strategy = str(options.get("strategy", "sample"))
        probabilities = self.distribution(session, **options)
        if strategy == "greedy":
            token_id = int(torch.argmax(probabilities))
        else:
            generator = torch.Generator().manual_seed(seed) if seed is not None else None
            token_id = int(torch.multinomial(probabilities, 1, generator=generator))
        return self.describe_token(token_id, float(probabilities[token_id]))

    def describe_token(self, token_id: int, probability: float | None = None) -> dict[str, Any]:
        vocabulary = int(self.runtime.config["vocab_size"])
        if not 0 <= token_id < vocabulary:
            raise StreamingModelError(f"Token id {token_id} is outside the vocabulary")
        result: dict[str, Any] = {
            "token_id": token_id,
            "token": self.tokenizer.decode([token_id]),
            "token_piece": self.tokenizer.convert_ids_to_tokens(token_id),
        }
        if probability is not None:
            result["probability"] = probability
        return result

    def commit(self, session: Any, token_id: int, *, advance: bool = True) -> dict[str, Any]:
        self.initialize(session)
        if session.finished:
            raise StreamingModelError("Session is finished; rewind it before appending more tokens")
        self.describe_token(token_id)
        finished = token_id in self.eos_ids()
        if advance and not finished:
            started = time.perf_counter()
            session.next_logits = self.runtime.forward([token_id], session.cache)
            session.decode_seconds += time.perf_counter() - started
            session.decode_steps += 1
        session.generated_ids.append(token_id)
        session.finished = finished
        result = self.describe_token(token_id)
        result.update(
            {
                "position": len(session.generated_ids) - 1,
                "committed": True,
                "finish_reason": "stop" if session.finished else None,
            }
        )
        return result

    def text(self, session: Any) -> str:
        return self.tokenizer.decode(session.generated_ids, skip_special_tokens=True)

    def stats(self) -> dict[str, Any]:
        result = {
            "backend": self.backend_name,
            "architecture": self.runtime.architecture,
            "weight_bytes_on_disk": int(self.store.manifest["weight_bytes"]),
            "logical_weight_bytes_read": int(self.store.bytes_read),
            "weight_buffer_bytes": int(self.store.buffer_bytes),
            "managed_ram_budget_bytes": int(self.store.ram_budget_bytes),
            "resident_vector_bytes": int(self.store._vector_bytes),
            "resident_shared_matrix_bytes": int(self.store._shared_matrix_bytes),
            "resident_shared_matrices": len(self.store._shared_matrices),
            "quantization": self.store.manifest.get("quantization"),
            "matmul": self.store.matmul_mode,
            "q4_kernel": (
                "avx512-vbmi-q4xf32"
                if self.store.native_avx512_q4_available
                else "numba-packed-gemv-gemm/native-int4pack"
                if self.store.numba_q4_available
                and self.store.native_q4_available
                and any("native_data_file" in record for record in self.store.tensors.values())
                else (
                    "numba-packed-gemv-gemm"
                    if self.store.numba_q4_available
                    else "native-int4pack"
                    if self.store.native_q4_available
                    and any("native_data_file" in record for record in self.store.tensors.values())
                    else "portable"
                )
            ),
            "async_prefetch": self.store.prefetch,
            "io_workers": self.store.io_workers,
            "expert_prediction": bool(getattr(self.runtime, "predict_experts", False)),
            "prefill_coalescing": bool(getattr(self.runtime, "coalesce_prefill", False)),
            "prefill_coalescing_gap": int(getattr(self.runtime, "prefill_coalescing_gap", 0)),
            "route_stability": self.runtime.route_stability_stats(),
            "prefill_layout": self.runtime.prefill_layout_stats(),
            "kv_cache_directory": str(self.kv_cache_dir),
            "max_context": self.max_context,
        }
        if self.runtime.architecture == "qwen3_moe":
            result["expert_cache"] = self.store.expert_cache.stats()
            result["moe"] = self.store.manifest.get("moe", {})
        return result
