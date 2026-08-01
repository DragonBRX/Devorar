#!/usr/bin/env python3
"""Create a bounded Lira feasibility map for a frontier Hugging Face model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.frontier_stream import (
    FrontierScanError,
    HardwareBudget,
    HuggingFaceRangeTransport,
    build_frontier_plan,
    inspect_repository,
    token_from_environment,
)


SOURCE_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SOURCE_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
SOURCE_LICENSE = "MIT"
MANIFEST_NAME = "frontier-scan.lira.json"
SUMMARY_NAME = "run-summary.json"
OUTPUT_SENTINEL_NAME = ".devorar-frontier-plan-v1"
OUTPUT_SENTINEL_CONTENT = "DragonBRX/Devorar managed frontier plan v1\n"
REPOSITORY_ROOT = Path(__file__).resolve().parent


def _default_output() -> Path:
    content = Path("/content")
    return content / "devorar-frontier-plan" if content.is_dir() else REPOSITORY_ROOT / "devorar-frontier-plan"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_output_target(target: Path) -> None:
    resolved = target.resolve()
    protected = {Path.cwd().resolve(), Path.home().resolve(), REPOSITORY_ROOT.resolve()}
    contains_protected = any(item == resolved or item.is_relative_to(resolved) for item in protected)
    if resolved == Path(resolved.anchor) or contains_protected:
        raise RuntimeError(f"Refusing broad output directory: {resolved}")
    if target.exists() and (not target.is_dir() or target.is_symlink()):
        raise RuntimeError(f"Output target must be a real directory: {target}")


def _require_managed_output(target: Path) -> None:
    sentinel = target / OUTPUT_SENTINEL_NAME
    try:
        value = sentinel.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"Refusing to overwrite an unmanaged frontier plan: {target}") from error
    if value != OUTPUT_SENTINEL_CONTENT:
        raise RuntimeError(f"Invalid frontier plan sentinel: {sentinel}")


def _commit(staging: Path, target: Path, *, overwrite: bool) -> None:
    backup: Path | None = None
    if target.exists():
        if not overwrite:
            raise RuntimeError(f"Output already exists: {target}; pass --overwrite-output")
        _require_managed_output(target)
        backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
        os.replace(target, backup)
    try:
        os.replace(staging, target)
    except BaseException:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    if backup is not None:
        if backup.parent.resolve() != target.parent.resolve() or not backup.name.startswith(
            f".{target.name}.backup-"
        ):
            raise RuntimeError(f"Refusing to remove unexpected backup path: {backup}")
        try:
            shutil.rmtree(backup)
        except OSError as error:
            # Publication already succeeded.  A stale, managed backup is safer than
            # reporting that the new plan failed after it became canonical.
            warnings.warn(f"published plan, but could not remove managed backup {backup}: {error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the immutable DeepSeek-V4-Flash safetensors map with HTTP ranges. "
            "This command creates a feasibility manifest, not a language-model checkpoint."
        )
    )
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, default=_default_output())
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--ram-gib", type=float, default=12.7)
    parser.add_argument("--vram-gib", type=float, default=14.5)
    parser.add_argument("--disk-gib", type=float, default=63.0)
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable containing an optional Hub token; its value is never recorded",
    )
    return parser


def _run_staged(args: argparse.Namespace) -> dict[str, Any]:
    token = token_from_environment(args.hf_token_env)
    transport = HuggingFaceRangeTransport(
        SOURCE_MODEL,
        SOURCE_REVISION,
        token=token,
        timeout_seconds=args.timeout_seconds,
    )
    inspection = inspect_repository(
        repo_id=SOURCE_MODEL,
        revision=SOURCE_REVISION,
        license_name=SOURCE_LICENSE,
        transport=transport,
        workers=args.workers,
    )
    hardware = HardwareBudget(
        ram_gib=args.ram_gib,
        vram_gib=args.vram_gib,
        disk_gib=args.disk_gib,
    )
    plan = build_frontier_plan(inspection, hardware=hardware)
    plan["build_id"] = str(uuid.uuid4())
    plan["created_at"] = _utc_now()
    plan["artifact_status"] = "experimental_non_canonical_inspection_only"

    args.output_dir.mkdir(parents=False, exist_ok=False)
    (args.output_dir / OUTPUT_SENTINEL_NAME).write_text(
        OUTPUT_SENTINEL_CONTENT,
        encoding="utf-8",
    )
    manifest_payload = _json_bytes(plan)
    _write_bytes_atomic(args.output_dir / MANIFEST_NAME, manifest_payload)
    summary = {
        "schema": "dragonbrx.devorar.frontier-scan-summary",
        "schema_version": 1,
        "status": plan["status"],
        "build_id": plan["build_id"],
        "manifest": MANIFEST_NAME,
        "manifest_sha256": _sha256_bytes(manifest_payload),
        "source": f"{SOURCE_MODEL}@{SOURCE_REVISION}",
        "source_payload_bytes": inspection.source_payload_bytes,
        "metadata_bytes_received": inspection.transfer_stats.bytes_received,
        "full_shards_downloaded": 0,
        "standalone_candidate_created": False,
    }
    _write_bytes_atomic(args.output_dir / SUMMARY_NAME, _json_bytes(summary))
    return plan


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_target = args.output_dir.expanduser().absolute()
    if raw_target.is_symlink():
        raise RuntimeError(f"Refusing symlink output directory: {raw_target}")
    target = raw_target.resolve()
    _validate_output_target(target)
    if target.exists():
        if not args.overwrite_output:
            raise RuntimeError(f"Output already exists: {target}; pass --overwrite-output")
        _require_managed_output(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    # _run_staged owns creation of its final leaf, so use a child of the temp root.
    staged_output = staging / "result"
    staged_args = argparse.Namespace(**vars(args))
    staged_args.output_dir = staged_output
    try:
        plan = _run_staged(staged_args)
        os.replace(staged_output, staging.with_name(staging.name + "-ready"))
        ready = staging.with_name(staging.name + "-ready")
        staging.rmdir()
        _commit(ready, target, overwrite=args.overwrite_output)
        return plan
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        ready = staging.with_name(staging.name + "-ready")
        if ready.exists() and ready.parent.resolve() == target.parent.resolve():
            shutil.rmtree(ready, ignore_errors=True)
        raise


def _category_map(plan: Mapping[str, Any]) -> dict[str, int]:
    rows = plan["inspection"]["inventory"]["categories"]
    return {str(row["category"]): int(row["stored_bytes"]) for row in rows}


def print_report(plan: Mapping[str, Any], output_dir: Path) -> None:
    inspection = plan["inspection"]
    inventory = inspection["inventory"]
    scan = inspection["scan"]
    categories = _category_map(plan)
    print("\nDragonBRX / Devorar Frontier — mapa remoto concluído")
    print(f"Fonte: {SOURCE_MODEL}@{SOURCE_REVISION}")
    print(f"Inventário: {inventory['tensor_count']:,} tensores / {inventory['shard_count']} shards")
    print(f"Pesos declarados: {inventory['source_payload_bytes'] / (2**30):.3f} GiB")
    print(f"Metadados transferidos: {scan['bytes_received'] / (2**20):.3f} MiB")
    print(f"Shards completos baixados: {scan['full_shards_downloaded']}")
    print(f"Experts roteados: {categories.get('routed_experts', 0) / (2**30):.3f} GiB")
    print(
        "Tronco + shared experts: "
        f"{(inventory['source_payload_bytes'] - categories.get('routed_experts', 0)) / (2**30):.3f} GiB"
    )
    print("Checkpoint criado: não — gate científico/runtime bloqueou materialização")
    print(f"Manifesto: {output_dir / MANIFEST_NAME}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = run(args)
        print_report(plan, args.output_dir.expanduser().absolute().resolve())
        return 0
    except (FrontierScanError, RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
