# SPDX-License-Identifier: MIT

"""Disk-first TokenQL Q8 to block-Q4 converter using bounded NumPy buffers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


class RequantizeError(RuntimeError):
    pass


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RequantizeError(f"Tensor path escapes model directory: {relative}") from exc
    return path


def _q4_paths(record: dict[str, Any]) -> tuple[str, str]:
    source = Path(record["data_file"])
    data = source.with_suffix(".q4").as_posix()
    scale = source.with_suffix(".scale.f32").as_posix()
    return data, scale


def requantize_matrix(
    source_root: Path,
    output_root: Path,
    record: dict[str, Any],
    block_size: int | None,
    chunk_mb: int,
) -> dict[str, Any]:
    rows, columns = map(int, record["shape"])
    if block_size is None:
        padded_columns = columns + (columns % 2)
        effective_block_size = padded_columns
        blocks = 1
    else:
        effective_block_size = block_size
        blocks = (columns + effective_block_size - 1) // effective_block_size
        padded_columns = blocks * effective_block_size
    data_relative, scale_relative = _q4_paths(record)
    data_path = output_root / data_relative
    scale_path = output_root / scale_relative
    data_path.parent.mkdir(parents=True, exist_ok=True)
    scale_path.parent.mkdir(parents=True, exist_ok=True)

    source_q = np.memmap(
        _safe_path(source_root, record["data_file"]),
        mode="r",
        dtype=np.int8,
        shape=(rows, columns),
    )
    source_scales = np.memmap(
        _safe_path(source_root, record["scale_file"]),
        mode="r",
        dtype=np.float32,
        shape=(rows,),
    )
    budget = max(1, chunk_mb) * 1024 * 1024
    rows_per_chunk = max(1, budget // max(1, padded_columns * 8))

    with data_path.open("wb") as data_file, scale_path.open("wb") as scale_file:
        for start in range(0, rows, rows_per_chunk):
            end = min(rows, start + rows_per_chunk)
            q8 = np.array(source_q[start:end], dtype=np.int8, copy=True)
            row_scales = np.array(source_scales[start:end], dtype=np.float32, copy=True)
            values = q8.astype(np.float32)
            values *= row_scales[:, None]
            if padded_columns != columns:
                values = np.pad(values, ((0, 0), (0, padded_columns - columns)))
            blocked = values.reshape(end - start, blocks, effective_block_size)
            scales = np.max(np.abs(blocked), axis=2) / np.float32(7.0)
            scales[scales == 0] = 1.0
            quantized = np.rint(blocked / scales[:, :, None])
            np.clip(quantized, -7, 7, out=quantized)
            quantized = quantized.astype(np.int8).reshape(end - start, padded_columns)
            unsigned = np.bitwise_and(quantized.astype(np.int16), 0x0F).astype(np.uint8)
            packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
            data_file.write(packed.tobytes(order="C"))
            scale_file.write(scales.astype(np.float32, copy=False).tobytes(order="C"))

    del source_q, source_scales
    return {
        "shape": [rows, columns],
        "storage": "q4_block",
        "block_size": effective_block_size,
        "padded_columns": padded_columns,
        "data_file": data_relative,
        "scale_file": scale_relative,
        "bytes": data_path.stat().st_size + scale_path.stat().st_size,
    }


def requantize(
    source: Path,
    output: Path,
    block_size: int | None,
    chunk_mb: int,
    overwrite: bool,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    manifest_path = source / "tokenql_manifest.json"
    if not manifest_path.exists():
        raise RequantizeError(f"No TokenQL manifest under {source}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("quantization") != "q8_row_symmetric":
        raise RequantizeError("Source must be a TokenQL row-Q8 model")
    if manifest.get("architecture") == "qwen3_moe":
        raise RequantizeError(
            "Packed MoE models must be converted directly from safetensors with "
            "convert_tokenql.py --quantization q4"
        )
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise RequantizeError(f"Output directory {output} is not empty; use --overwrite")
        if not (output / "tokenql_manifest.json").exists():
            raise RequantizeError("Refusing to overwrite a directory that is not a TokenQL model")
        shutil.rmtree(output / "weights", ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)

    for source_file in source.iterdir():
        if source_file.is_file() and source_file.name != "tokenql_manifest.json":
            shutil.copy2(source_file, output / source_file.name)

    started = time.perf_counter()
    converted: dict[str, Any] = {}
    items = list(manifest["tensors"].items())
    for index, (name, record) in enumerate(items, start=1):
        print(f"[{index:03d}/{len(items):03d}] {name}", flush=True)
        if record["storage"] == "q8_row":
            converted[name] = requantize_matrix(source, output, record, block_size, chunk_mb)
        elif record["storage"] == "f32":
            source_path = _safe_path(source, record["data_file"])
            output_path = output / record["data_file"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, output_path)
            converted[name] = dict(record)
        else:
            raise RequantizeError(f"Unsupported source storage {record['storage']!r}")

    result = dict(manifest)
    result.update(
        {
            "source_path": str(source),
            "quantization": (
                "q4_row_symmetric" if block_size is None else f"q4_block_symmetric_{block_size}"
            ),
            "tensors": converted,
            "weight_bytes": sum(record["bytes"] for record in converted.values()),
            "conversion_seconds": time.perf_counter() - started,
        }
    )
    temporary = output / "tokenql_manifest.json.tmp"
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temporary, output / "tokenql_manifest.json")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Requantize TokenQL Q8 weights to disk-first Q4")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--row-wise",
        action="store_true",
        help="Use one Q4 scale per row; faster integer matmul but less accurate",
    )
    parser.add_argument("--chunk-mb", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.row_wise and (args.block_size < 16 or args.block_size % 2):
        parser.error("--block-size must be an even integer of at least 16")
    try:
        manifest = requantize(
            args.source,
            args.output,
            None if args.row_wise else args.block_size,
            args.chunk_mb,
            args.overwrite,
        )
    except (RequantizeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Requantized to {args.output.resolve()} "
        f"({manifest['weight_bytes'] / 1048576:.1f} MiB, "
        f"{manifest['conversion_seconds']:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
