#!/usr/bin/env python3
"""One-command Colab launcher for the bounded frontier metadata experiment."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parent
SENTINEL_NAME = ".devorar-frontier-plan-v1"
SENTINEL_CONTENT = "DragonBRX/Devorar managed frontier plan v1\n"


def _default_output() -> Path:
    content = Path("/content")
    return content / "devorar-frontier-plan" if content.is_dir() else REPOSITORY_ROOT / "devorar-frontier-plan"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the DeepSeek-V4-Flash remote range scanner and create a ZIP report."
    )
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, default=_default_output())
    return parser


def _resolve(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = REPOSITORY_ROOT / expanded
    return expanded.resolve()


def _verify_owned_directory(output_dir: Path) -> None:
    try:
        value = (output_dir / SENTINEL_NAME).read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"Refusing to archive unmanaged output: {output_dir}") from error
    if value != SENTINEL_CONTENT:
        raise RuntimeError(f"Invalid frontier output sentinel: {output_dir / SENTINEL_NAME}")


def _archive(output_dir: Path) -> Path:
    _verify_owned_directory(output_dir)
    archive = Path(f"{output_dir}.zip")
    if archive.exists():
        if not archive.is_file() or archive.is_symlink():
            raise RuntimeError(f"Refusing non-file archive target: {archive}")
        try:
            with zipfile.ZipFile(archive) as previous:
                value = previous.read(SENTINEL_NAME).decode("utf-8").replace("\r\n", "\n")
            if value != SENTINEL_CONTENT:
                raise RuntimeError("archive has an invalid ownership marker")
        except (KeyError, OSError, UnicodeDecodeError, zipfile.BadZipFile, RuntimeError) as error:
            raise RuntimeError(f"Refusing to replace an unmanaged archive: {archive}") from error
    temporary_base = output_dir.parent / f".{output_dir.name}.archive-{uuid.uuid4().hex}"
    temporary_archive = Path(shutil.make_archive(str(temporary_base), "zip", root_dir=output_dir))
    try:
        os.replace(temporary_archive, archive)
    finally:
        if temporary_archive.exists():
            temporary_archive.unlink()
    return archive


def main(argv: Sequence[str] | None = None) -> int:
    args, scanner_args = build_parser().parse_known_args(argv)
    output_dir = _resolve(args.output_dir)
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "frontier_scan.py"),
        "--output-dir",
        str(output_dir),
        "--overwrite-output",
        *scanner_args,
    ]
    try:
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        archive = _archive(output_dir)
    except subprocess.CalledProcessError as error:
        print(f"Devorar Frontier failed (exit {error.returncode}).", file=sys.stderr)
        return error.returncode or 1
    except OSError as error:
        print(f"Devorar Frontier could not run or archive its output: {error}", file=sys.stderr)
        return 1
    except RuntimeError as error:
        print(f"Scan completed, but ZIP was not replaced: {error}", file=sys.stderr)
        return 1
    print(f"\nFrontier plan ready: {output_dir}")
    print(f"ZIP: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
