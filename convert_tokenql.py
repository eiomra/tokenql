"""Convert dense Qwen2/Qwen2.5 or sparse Qwen3-MoE weights to TokenQL.

The converter never materializes the whole checkpoint. Two-dimensional weight
matrices are read in row chunks and quantized to Q8 or block-Q4. Small vectors
remain FP32. Qwen3 experts are coalesced into offset-addressable per-layer packs.
The result executes without constructing a Transformers model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* or chardet.*doesn't match a supported version!",
)

from huggingface_hub import snapshot_download
from safetensors import safe_open

FORMAT_NAME = "tokenql-qwen-stream"
FORMAT_VERSION = 1
TOKENIZER_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)


class ConversionError(RuntimeError):
    pass


def _install_windows_hf_link_fallback() -> None:
    """Make Hugging Face cache finalization work without symlink privilege.

    Windows normally requires Developer Mode or an elevated process to create
    symbolic links.  The Hub cache can still be represented safely with a hard
    link because its blobs and snapshots live on the same volume.  Copying is
    the final fallback for unusual cache layouts spanning volumes.
    """
    if os.name != "nt":
        return
    try:
        import huggingface_hub.file_download as hub_download
    except ImportError:
        return
    original = hub_download._create_symlink
    if getattr(original, "_tokenql_windows_fallback", False):
        return

    def create_link(src: str, dst: str, new_blob: bool = False) -> None:
        try:
            original(src, dst, new_blob)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) != 1314:
                raise
        destination = Path(dst)
        source = Path(src)
        if not source.is_absolute():
            source = (destination.parent / source).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    create_link._tokenql_windows_fallback = True  # type: ignore[attr-defined]
    hub_download._create_symlink = create_link


def resolve_source(source: str, offline: bool) -> Path:
    candidate = Path(source).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    _install_windows_hf_link_fallback()
    try:
        return Path(snapshot_download(repo_id=source, local_files_only=offline)).resolve()
    except Exception as exc:
        mode = "cached" if offline else "downloadable"
        raise ConversionError(f"Could not resolve {mode} model {source!r}: {exc}") from exc


def tensor_files(source: Path) -> dict[str, Path]:
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return {name: source / filename for name, filename in index["weight_map"].items()}

    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise ConversionError(f"No safetensors checkpoint found under {source}")
    result: dict[str, Path] = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                if name in result:
                    raise ConversionError(f"Tensor {name!r} occurs in more than one shard")
                result[name] = file
    return result


def expected_qwen_tensors(config: dict[str, Any]) -> set[str]:
    expected = {"model.embed_tokens.weight", "model.norm.weight"}
    for layer in range(int(config["num_hidden_layers"])):
        prefix = f"model.layers.{layer}"
        expected.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.v_proj.bias",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            }
        )
    if not config.get("tie_word_embeddings", False):
        expected.add("lm_head.weight")
    return expected


def expected_qwen3_moe_tensors(config: dict[str, Any]) -> set[str]:
    """Return the independently addressable tensors used by Qwen3-MoE."""
    expected = {"model.embed_tokens.weight", "model.norm.weight"}
    experts = int(config["num_experts"])
    for layer in range(int(config["num_hidden_layers"])):
        prefix = f"model.layers.{layer}"
        expected.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
                f"{prefix}.mlp.gate.weight",
            }
        )
        for expert in range(experts):
            expert_prefix = f"{prefix}.mlp.experts.{expert}"
            expected.update(
                {
                    f"{expert_prefix}.gate_proj.weight",
                    f"{expert_prefix}.up_proj.weight",
                    f"{expert_prefix}.down_proj.weight",
                }
            )
    if not config.get("tie_word_embeddings", False):
        expected.add("lm_head.weight")
    return expected


def relative_tensor_path(name: str, suffix: str) -> Path:
    return Path("weights", *name.split(".")).with_suffix(suffix)


def chunk_ranges(rows: int, columns: int, chunk_mb: int) -> Iterator[tuple[int, int]]:
    # During conversion each element can briefly occupy BF16/FP16 source,
    # FP32 working, and INT8 output storage. Seven bytes also accounts for a
    # possible BF16/FP16 source slice.
    # approximation for selecting a bounded row count.
    budget = max(1, chunk_mb) * 1024 * 1024
    rows_per_chunk = max(1, budget // max(1, columns * 7))
    for start in range(0, rows, rows_per_chunk):
        yield start, min(rows, start + rows_per_chunk)


def convert_matrix(
    tensor_slice: Any,
    shape: list[int],
    output: Path,
    name: str,
    chunk_mb: int,
) -> dict[str, Any]:
    rows, columns = shape
    q_path = output / relative_tensor_path(name, ".q8")
    scale_path = output / relative_tensor_path(name, ".scale.f32")
    q_path.parent.mkdir(parents=True, exist_ok=True)
    scale_path.parent.mkdir(parents=True, exist_ok=True)

    with q_path.open("wb") as q_file, scale_path.open("wb") as scale_file:
        for start, end in chunk_ranges(rows, columns, chunk_mb):
            values = tensor_slice[start:end].to(dtype=torch.float32)
            scales = values.abs().amax(dim=1) / 127.0
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)
            quantized = torch.round(values / scales[:, None]).clamp_(-127, 127).to(torch.int8)
            q_file.write(quantized.contiguous().numpy().tobytes())
            scale_file.write(scales.contiguous().numpy().tobytes())
            del values, scales, quantized

    return {
        "shape": shape,
        "storage": "q8_row",
        "data_file": q_path.relative_to(output).as_posix(),
        "scale_file": scale_path.relative_to(output).as_posix(),
        "bytes": q_path.stat().st_size + scale_path.stat().st_size,
    }


def convert_matrix_q4(
    tensor_slice: Any,
    shape: list[int],
    output: Path,
    name: str,
    chunk_mb: int,
    block_size: int,
) -> dict[str, Any]:
    rows, columns = shape
    blocks = (columns + block_size - 1) // block_size
    padded_columns = blocks * block_size
    data_path = output / relative_tensor_path(name, ".q4")
    scale_path = output / relative_tensor_path(name, ".scale.f32")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    scale_path.parent.mkdir(parents=True, exist_ok=True)

    with data_path.open("wb") as data_file, scale_path.open("wb") as scale_file:
        for start, end in chunk_ranges(rows, columns, chunk_mb):
            values = tensor_slice[start:end].to(dtype=torch.float32)
            if padded_columns != columns:
                values = F.pad(values, (0, padded_columns - columns))
            blocked = values.reshape(end - start, blocks, block_size)
            scales = blocked.abs().amax(dim=2) / 7.0
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)
            quantized = torch.round(blocked / scales[:, :, None]).clamp_(-7, 7).to(torch.int8)
            quantized = quantized.reshape(end - start, padded_columns)
            unsigned = quantized.to(torch.int16).bitwise_and(0x0F).to(torch.uint8)
            packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
            data_file.write(packed.contiguous().numpy().tobytes())
            scale_file.write(scales.contiguous().numpy().tobytes())
            del values, blocked, scales, quantized, unsigned, packed

    return {
        "shape": shape,
        "storage": "q4_block",
        "block_size": block_size,
        "padded_columns": padded_columns,
        "data_file": data_path.relative_to(output).as_posix(),
        "scale_file": scale_path.relative_to(output).as_posix(),
        "bytes": data_path.stat().st_size + scale_path.stat().st_size,
    }


def convert_vector(tensor_slice: Any, shape: list[int], output: Path, name: str) -> dict[str, Any]:
    path = output / relative_tensor_path(name, ".f32")
    path.parent.mkdir(parents=True, exist_ok=True)
    values = tensor_slice[:].to(dtype=torch.float32).contiguous()
    with path.open("wb") as file:
        file.write(values.numpy().tobytes())
    del values
    return {
        "shape": shape,
        "storage": "f32",
        "data_file": path.relative_to(output).as_posix(),
        "bytes": path.stat().st_size,
    }


def existing_tensor_record(
    output: Path,
    name: str,
    shape: list[int],
    quantization: str,
    block_size: int,
) -> dict[str, Any] | None:
    """Recover metadata for a complete tensor from an interrupted conversion."""
    if len(shape) == 1:
        path = output / relative_tensor_path(name, ".f32")
        expected_bytes = int(shape[0]) * 4
        if path.is_file() and path.stat().st_size == expected_bytes:
            return {
                "shape": shape,
                "storage": "f32",
                "data_file": path.relative_to(output).as_posix(),
                "bytes": expected_bytes,
            }
        return None
    if len(shape) != 2:
        return None

    rows, columns = map(int, shape)
    if quantization == "q8":
        data_path = output / relative_tensor_path(name, ".q8")
        scale_path = output / relative_tensor_path(name, ".scale.f32")
        data_bytes = rows * columns
        scale_bytes = rows * 4
        if (
            data_path.is_file()
            and scale_path.is_file()
            and data_path.stat().st_size == data_bytes
            and scale_path.stat().st_size == scale_bytes
        ):
            return {
                "shape": shape,
                "storage": "q8_row",
                "data_file": data_path.relative_to(output).as_posix(),
                "scale_file": scale_path.relative_to(output).as_posix(),
                "bytes": data_bytes + scale_bytes,
            }
        return None

    blocks = (columns + block_size - 1) // block_size
    padded_columns = blocks * block_size
    data_path = output / relative_tensor_path(name, ".q4")
    scale_path = output / relative_tensor_path(name, ".scale.f32")
    data_bytes = rows * (padded_columns // 2)
    scale_bytes = rows * blocks * 4
    if (
        data_path.is_file()
        and scale_path.is_file()
        and data_path.stat().st_size == data_bytes
        and scale_path.stat().st_size == scale_bytes
    ):
        return {
            "shape": shape,
            "storage": "q4_block",
            "block_size": block_size,
            "padded_columns": padded_columns,
            "data_file": data_path.relative_to(output).as_posix(),
            "scale_file": scale_path.relative_to(output).as_posix(),
            "bytes": data_bytes + scale_bytes,
        }
    return None


def pack_moe_experts(
    records: dict[str, dict[str, Any]], output: Path, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Coalesce thousands of expert fragments into one addressable pack per layer."""
    packs: list[dict[str, Any]] = []
    projections = ("gate_proj", "up_proj", "down_proj")
    for layer in range(int(config["num_hidden_layers"])):
        relative_pack = Path("weights", "experts", f"layer-{layer:03d}.experts.pack")
        pack_path = output / relative_pack
        pack_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = pack_path.with_suffix(pack_path.suffix + ".tmp")
        source_files: set[Path] = set()
        expert_offsets: list[int] = []
        with temporary.open("wb") as packed:
            for expert in range(int(config["num_experts"])):
                expert_offsets.append(packed.tell())
                for projection in projections:
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                    record = records[name]
                    for file_key, offset_key in (
                        ("data_file", "data_offset"),
                        ("scale_file", "scale_offset"),
                    ):
                        source = output / record[file_key]
                        source_files.add(source)
                        record[file_key] = relative_pack.as_posix()
                        record[offset_key] = packed.tell()
                        with source.open("rb") as source_handle:
                            shutil.copyfileobj(source_handle, packed, length=1024 * 1024)
        os.replace(temporary, pack_path)
        for source in source_files:
            source.unlink(missing_ok=True)
        shutil.rmtree(
            output / "weights" / "model" / "layers" / str(layer) / "mlp" / "experts",
            ignore_errors=True,
        )
        packs.append(
            {
                "layer": layer,
                "file": relative_pack.as_posix(),
                "bytes": pack_path.stat().st_size,
                "expert_offsets": expert_offsets,
            }
        )
    return packs


def convert(
    source_name: str,
    output: Path,
    offline: bool,
    chunk_mb: int,
    overwrite: bool,
    quantization: str = "q8",
    block_size: int = 128,
) -> dict[str, Any]:
    source = resolve_source(source_name, offline)
    config_path = source / "config.json"
    if not config_path.exists():
        raise ConversionError("Source model has no config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    if model_type not in {"qwen2", "qwen3_moe"}:
        raise ConversionError(
            f"Version 1 supports Qwen2/Qwen2.5 and Qwen3-MoE; found model_type={model_type!r}"
        )

    locations = tensor_files(source)
    expected = (
        expected_qwen3_moe_tensors(config)
        if model_type == "qwen3_moe"
        else expected_qwen_tensors(config)
    )
    missing = sorted(expected - locations.keys())
    if missing:
        raise ConversionError(f"Checkpoint is missing required tensors: {', '.join(missing[:5])}")

    resuming = False
    if output.exists() and any(output.iterdir()):
        manifest = output / "tokenql_manifest.json"
        if overwrite:
            if not manifest.exists():
                raise ConversionError(
                    "Refusing to overwrite a directory that is not a TokenQL model"
                )
            shutil.rmtree(output / "weights", ignore_errors=True)
        elif manifest.exists():
            raise ConversionError(f"Output directory {output} is complete; use --overwrite")
        else:
            partial_config = output / "config.json"
            if not partial_config.exists() or not (output / "weights").is_dir():
                raise ConversionError(
                    f"Output directory {output} is not a recognizable partial conversion"
                )
            try:
                previous_config = json.loads(partial_config.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ConversionError("Partial conversion has an invalid config.json") from exc
            if previous_config != config:
                raise ConversionError(
                    "Partial conversion belongs to a different model configuration"
                )
            resuming = True
    output.mkdir(parents=True, exist_ok=True)

    for filename in TOKENIZER_FILES:
        source_file = source / filename
        if source_file.exists():
            shutil.copy2(source_file, output / filename)

    started = time.perf_counter()
    records: dict[str, Any] = {}
    reused = 0
    ordered_names = sorted(expected, key=_tensor_order)
    current_shard_path: Path | None = None
    current_shard_context = None
    current_shard = None
    tensor_slice = None
    try:
        for number, name in enumerate(ordered_names, start=1):
            shard_path = locations[name]
            if shard_path != current_shard_path:
                # Checkpoint shards can be several GiB. Retaining every visited
                # mapping eventually crashes torch_cpu.dll on Windows even though
                # tensor conversion itself is streamed.
                tensor_slice = None
                current_shard = None
                if current_shard_context is not None:
                    current_shard_context.__exit__(None, None, None)
                current_shard_context = safe_open(shard_path, framework="pt", device="cpu")
                current_shard = current_shard_context.__enter__()
                current_shard_path = shard_path

            tensor_slice = current_shard.get_slice(name)
            shape = list(tensor_slice.get_shape())
            existing = (
                existing_tensor_record(output, name, shape, quantization, block_size)
                if resuming
                else None
            )
            if existing is not None:
                records[name] = existing
                reused += 1
                if reused == 1 or number % 250 == 0:
                    print(
                        f"[{number:05d}/{len(ordered_names):05d}] resume: {reused} tensors reused",
                        flush=True,
                    )
                continue
            print(f"[{number:03d}/{len(ordered_names):03d}] {name} {tuple(shape)}", flush=True)
            if len(shape) == 2 and quantization == "q4":
                record = convert_matrix_q4(tensor_slice, shape, output, name, chunk_mb, block_size)
            elif len(shape) == 2:
                record = convert_matrix(tensor_slice, shape, output, name, chunk_mb)
            elif len(shape) == 1:
                record = convert_vector(tensor_slice, shape, output, name)
            else:
                raise ConversionError(f"Unsupported rank {len(shape)} for tensor {name}")
            records[name] = record
    finally:
        tensor_slice = None
        current_shard = None
        if current_shard_context is not None:
            current_shard_context.__exit__(None, None, None)

    expert_packs: list[dict[str, Any]] = []
    if model_type == "qwen3_moe":
        expert_packs = pack_moe_experts(records, output, config)

    expert_pattern = re.compile(r"\.mlp\.experts\.\d+\.")
    expert_weight_bytes = sum(
        record["bytes"] for name, record in records.items() if expert_pattern.search(name)
    )
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "source_model": source_name,
        "source_path": str(source),
        "architecture": model_type,
        "quantization": (
            f"q4_block_symmetric_{block_size}" if quantization == "q4" else "q8_row_symmetric"
        ),
        "config": config,
        "tensors": records,
        "weight_bytes": sum(record["bytes"] for record in records.values()),
        "shared_weight_bytes": sum(record["bytes"] for record in records.values())
        - expert_weight_bytes,
        "expert_weight_bytes": expert_weight_bytes,
        "conversion_seconds": time.perf_counter() - started,
        "resumed_tensors": reused,
    }
    if model_type == "qwen3_moe":
        manifest["moe"] = {
            "num_experts": int(config["num_experts"]),
            "experts_per_token": int(config["num_experts_per_tok"]),
            "expert_projections": ["gate_proj", "up_proj", "down_proj"],
            "packs": expert_packs,
        }
    temporary_manifest = output / "tokenql_manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, output / "tokenql_manifest.json")
    return manifest


def _tensor_order(name: str) -> tuple[int, int, str]:
    if name == "model.embed_tokens.weight":
        return (-2, 0, name)
    if name.startswith("model.layers."):
        layer = int(name.split(".")[2])
        expert = re.search(r"\.experts\.(\d+)\.", name)
        # Shared tensors first, then experts in numeric order for sequential
        # writes that preserve layer/expert locality on disk.
        return (layer, 1 + int(expert.group(1)) if expert else 0, name)
    return (10**9, 0, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Qwen2/Qwen2.5/Qwen3-MoE safetensors to TokenQL"
    )
    parser.add_argument("--source", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-mb", type=int, default=64, help="Maximum conversion working chunk")
    parser.add_argument("--quantization", choices=["q8", "q4"], default="q8")
    parser.add_argument("--block-size", type=int, default=128, help="Q4 scaling block size")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.block_size < 16 or args.block_size % 2:
        parser.error("--block-size must be an even integer of at least 16")
    try:
        manifest = convert(
            args.source,
            args.output.resolve(),
            args.offline,
            args.chunk_mb,
            args.overwrite,
            args.quantization,
            args.block_size,
        )
    except (ConversionError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    size_mb = manifest["weight_bytes"] / (1024 * 1024)
    print(
        f"Converted {manifest['source_model']} to {args.output.resolve()} "
        f"({size_mb:.1f} MiB, {manifest['conversion_seconds']:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
