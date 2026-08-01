#!/usr/bin/env python3
"""One-file bootstrap for running Devorar after a Git clone in Colab."""

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


def _colab_or_local_path(name: str) -> Path:
    content = Path("/content")
    return content / name if content.is_dir() else REPOSITORY_ROOT / name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install dependencies and run the complete Devorar experiment automatically."
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        default=_colab_or_local_path("devorar-output"),
    )
    parser.add_argument("--cache-dir", type=Path, default=_colab_or_local_path("huggingface-cache"))
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--no-install", action="store_true", help="Skip dependency installation")
    return parser


def _resolve_runtime_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = REPOSITORY_ROOT / expanded
    return expanded.resolve()


def _create_archive(output_dir: Path) -> Path:
    archive_path = Path(f"{output_dir}.zip")
    sentinel_name = ".devorar-output-v1"
    sentinel_content = "DragonBRX/Devorar managed output v1\n"
    try:
        current_sentinel = (output_dir / sentinel_name).read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"Refusing to archive unmanaged output: {output_dir}") from error
    if current_sentinel != sentinel_content:
        raise RuntimeError(f"Refusing invalid output sentinel: {output_dir / sentinel_name}")
    if archive_path.exists():
        if not archive_path.is_file() or archive_path.is_symlink():
            raise RuntimeError(f"Refusing non-file archive target: {archive_path}")
        try:
            with zipfile.ZipFile(archive_path) as previous:
                info = previous.getinfo(sentinel_name)
                if info.file_size > 128:
                    raise RuntimeError("archive sentinel is unexpectedly large")
                archived_sentinel = previous.read(info).decode("utf-8").replace("\r\n", "\n")
                if archived_sentinel != sentinel_content:
                    raise RuntimeError("archive sentinel content differs")
        except (KeyError, OSError, UnicodeDecodeError, zipfile.BadZipFile, RuntimeError) as error:
            raise RuntimeError(
                f"Refusing to replace an archive not owned by Devorar: {archive_path}"
            ) from error
    temporary_base = output_dir.parent / f".{output_dir.name}.archive-{uuid.uuid4().hex}"
    temporary_archive = Path(
        shutil.make_archive(str(temporary_base), "zip", root_dir=output_dir)
    )
    try:
        os.replace(temporary_archive, archive_path)
    finally:
        if temporary_archive.exists():
            temporary_archive.unlink()
    return archive_path


def main(argv: Sequence[str] | None = None) -> int:
    args, runner_args = build_parser().parse_known_args(argv)
    output_dir = _resolve_runtime_path(args.output_dir)
    cache_dir = _resolve_runtime_path(args.cache_dir)
    requirements = REPOSITORY_ROOT / "requirements-colab.txt"
    runner = REPOSITORY_ROOT / "run_colab.py"
    try:
        if not args.no_install:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "--disable-pip-version-check",
                    "-r",
                    str(requirements),
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
            )

        command = [
            sys.executable,
            str(runner),
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(cache_dir),
            "--overwrite-output",
        ]
        if args.cpu:
            command.append("--cpu")
        command.extend(runner_args)
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    except subprocess.CalledProcessError as error:
        print(f"Devorar falhou na etapa externa (código {error.returncode}).", file=sys.stderr)
        return error.returncode or 1

    try:
        archive = _create_archive(output_dir)
    except RuntimeError as error:
        print(f"Resultado concluído, mas o ZIP não foi substituído: {error}", file=sys.stderr)
        return 1
    print(f"\nDevorar concluído. Resultado: {output_dir}")
    print(f"Arquivo ZIP: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
