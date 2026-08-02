"""Prepack TokenQL Q4 matrices for PyTorch's native AVX CPU kernel.

The portable TokenQL Q4 layout remains untouched. This command creates
resumable sidecar files and atomically adds their locations to the manifest, so
models remain usable on PyTorch builds that do not expose the native kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


class OptimizationError(RuntimeError):
    pass


def _model_path(model_dir: Path, relative: str) -> Path:
    model_dir = model_dir.resolve()
    path = (model_dir / relative).resolve()
    try:
        path.relative_to(model_dir)
    except ValueError as exc:
        raise OptimizationError(f"Weight path escapes model directory: {relative}") from exc
    return path


def _native_relative_path(data_file: str) -> Path:
    source = Path(data_file)
    try:
        under_weights = source.relative_to("weights")
    except ValueError:
        under_weights = Path(*source.parts)
    return Path("weights", "native", under_weights.as_posix() + ".int4pack")


def _unpack_signed_q4(packed: np.ndarray, padded_columns: int) -> np.ndarray:
    unpacked = np.empty((packed.shape[0], padded_columns), dtype=np.int8)
    unpacked[:, 0::2] = (packed.astype(np.int8) << 4) >> 4
    unpacked[:, 1::2] = packed.view(np.int8) >> 4
    return unpacked


def _prepack_tensor_range(
    source: Path,
    destination_handle: Any,
    record: dict[str, Any],
    chunk_mb: int,
) -> None:
    rows, _ = map(int, record["shape"])
    padded_columns = int(record["padded_columns"])
    packed_columns = padded_columns // 2
    source_offset = int(record.get("data_offset", 0))
    if rows % 32:
        raise OptimizationError(f"Native INT4 output rows must be divisible by 32; received {rows}")
    target_bytes = max(1, int(chunk_mb)) * 1024 * 1024
    rows_per_chunk = max(32, (target_bytes // packed_columns // 32) * 32)

    for start in range(0, rows, rows_per_chunk):
        end = min(rows, start + rows_per_chunk)
        packed = np.fromfile(
            source,
            dtype=np.uint8,
            count=(end - start) * packed_columns,
            offset=source_offset + start * packed_columns,
        ).reshape(end - start, packed_columns)
        if packed.size != (end - start) * packed_columns:
            raise OptimizationError(f"Short read while prepacking {source}")
        signed = _unpack_signed_q4(packed, padded_columns)
        unsigned = (torch.from_numpy(signed).to(torch.int32) + 8).contiguous()
        native = torch._convert_weight_to_int4pack_for_cpu(unsigned, 1)
        if native.nbytes != packed.nbytes:
            raise OptimizationError("Native INT4 packing unexpectedly changed tensor byte size")
        destination_handle.seek(source_offset + start * packed_columns)
        destination_handle.write(native.numpy().tobytes())


def optimize(model_dir: Path, chunk_mb: int) -> dict[str, Any]:
    if not all(
        hasattr(torch, name)
        for name in (
            "_convert_weight_to_int4pack_for_cpu",
            "_weight_int4pack_mm_for_cpu",
        )
    ):
        raise OptimizationError("This PyTorch build has no native CPU INT4 kernel")

    model_dir = model_dir.resolve()
    manifest_path = model_dir / "tokenql_manifest.json"
    if not manifest_path.is_file():
        raise OptimizationError(f"No tokenql_manifest.json under {model_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensors: dict[str, dict[str, Any]] = manifest.get("tensors", {})
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for name, record in tensors.items():
        if record.get("storage") != "q4_block":
            continue
        # Embeddings require random row lookup. Keep their portable row-major
        # layout; a separate LM head is prepacked normally.
        if name == "model.embed_tokens.weight":
            continue
        grouped[str(record["data_file"])].append((name, record))
    if not grouped:
        raise OptimizationError("The model has no eligible Q4 matrices")

    started = time.perf_counter()
    completed = 0
    reused = 0
    for index, (data_file, members) in enumerate(sorted(grouped.items()), start=1):
        source = _model_path(model_dir, data_file)
        native_relative = _native_relative_path(data_file)
        destination = _model_path(model_dir, native_relative.as_posix())
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            reused += 1
            print(
                f"[{index:04d}/{len(grouped):04d}] resume: {native_relative}",
                flush=True,
            )
        else:
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.unlink(missing_ok=True)
            print(
                f"[{index:04d}/{len(grouped):04d}] prepack {len(members)} matrices -> "
                f"{native_relative}",
                flush=True,
            )
            with temporary.open("w+b") as output:
                output.truncate(source.stat().st_size)
                for _, record in sorted(
                    members, key=lambda item: int(item[1].get("data_offset", 0))
                ):
                    _prepack_tensor_range(source, output, record, chunk_mb)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        for _, record in members:
            record["native_data_file"] = native_relative.as_posix()
            record["native_data_offset"] = int(record.get("data_offset", 0))
        completed += len(members)

    manifest["native_q4"] = {
        "format": "pytorch-int4pack-cpu-v1",
        "torch_version": torch.__version__,
        "files": len(grouped),
        "tensors": completed,
        "reused_files": reused,
        "optimization_seconds": time.perf_counter() - started,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return manifest["native_q4"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepack TokenQL Q4 weights for native CPU inference"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.chunk_mb < 1:
        parser.error("--chunk-mb must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        result = optimize(args.model_dir, args.chunk_mb)
    except (OptimizationError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
