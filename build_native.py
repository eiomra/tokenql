"""Build TokenQL's optional AVX-512 VNNI Q4 kernel on Windows."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def find_vcvars() -> Path:
    roots = [
        Path(r"C:\Program Files\Microsoft Visual Studio"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
    ]
    matches: list[Path] = []
    for root in roots:
        if root.exists():
            matches.extend(root.glob(r"*\*\VC\Auxiliary\Build\vcvars64.bat"))
    if not matches:
        raise RuntimeError("Visual Studio C++ Build Tools with vcvars64.bat were not found")
    return sorted(matches)[-1]


def build(output: Path) -> Path:
    root = Path(__file__).resolve().parent
    source = root / "native_q4.cpp"
    if not source.exists():
        raise RuntimeError(f"Missing native kernel source: {source}")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    vcvars = find_vcvars()
    with tempfile.TemporaryDirectory(prefix="tokenql-native-") as temporary:
        build_dir = Path(temporary)
        temporary_dll = build_dir / output.name
        object_file = build_dir / "native_q4.obj"
        batch_file = build_dir / "build.bat"
        batch_file.write_text(
            "@echo off\n"
            f'call "{vcvars}" >nul\n'
            "if errorlevel 1 exit /b %errorlevel%\n"
            f"cl /nologo /O2 /W4 /EHsc /LD /openmp /arch:AVX512 "
            f'/Fo"{object_file}" /Fe:"{temporary_dll}" "{source}"\n'
            "exit /b %errorlevel%\n",
            encoding="utf-8",
        )
        subprocess.run(["cmd.exe", "/d", "/c", str(batch_file)], check=True)
        shutil.copy2(temporary_dll, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "_tokenql_q4_avx512.dll",
    )
    args = parser.parse_args()
    result = build(args.output)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
