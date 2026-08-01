#!/usr/bin/env python3
"""Fetch bounded micro-samples from selected DeepSeek-V4-Flash tensors."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.frontier_stream import (
    FrontierScanError,
    HuggingFaceRangeTransport,
    fingerprint_tensors,
    locate_tensors,
    token_from_environment,
    transport_receipts,
)


SOURCE_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SOURCE_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
SOURCE_LICENSE = "MIT"
PROBE_FORMAT = "lira.experimental.frontier-tensor-microsample"
PROBE_VERSION = 1
DEFAULT_TENSORS = (
    "embed.weight",
    "layers.0.attn.wq_a.weight",
    "layers.0.ffn.gate.weight",
    "layers.0.ffn.shared_experts.w1.weight",
    "layers.0.ffn.experts.0.w1.weight",
    "head.weight",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_output() -> Path:
    content = Path("/content")
    return (
        content / "dragonbrx-frontier-tensor-probe.lira.json"
        if content.is_dir()
        else Path.cwd() / "dragonbrx-frontier-tensor-probe.lira.json"
    )


def bounded_int(minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            result = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be an integer") from error
        if not minimum <= result <= maximum:
            raise argparse.ArgumentTypeError(f"must be within [{minimum}, {maximum}]")
        return result

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a few real parameter bytes from an immutable frontier checkpoint without "
            "downloading a shard. The output is structural/statistical, not a model response."
        )
    )
    parser.add_argument(
        "--tensor",
        action="append",
        help="Exact tensor name; repeat up to 16 times (defaults cover six structural roles)",
    )
    parser.add_argument("--sample-bytes", type=bounded_int(256, 65_536), default=4_096)
    parser.add_argument("--json-output", type=Path, default=_default_output())
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser


def _write_json_atomic(path: Path, report: Mapping[str, Any]) -> None:
    destination = path.expanduser().absolute()
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise FrontierScanError(f"probe output must be a regular file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def run(args: argparse.Namespace) -> dict[str, Any]:
    tensor_names = tuple(args.tensor or DEFAULT_TENSORS)
    if not 1 <= len(tensor_names) <= 16:
        raise FrontierScanError("select between 1 and 16 tensors")
    transport = HuggingFaceRangeTransport(
        SOURCE_MODEL,
        SOURCE_REVISION,
        token=token_from_environment(args.hf_token_env),
        timeout_seconds=args.timeout_seconds,
    )
    locations, index_sha256 = locate_tensors(transport, tensor_names)
    samples = fingerprint_tensors(
        transport,
        locations,
        sample_bytes_per_tensor=args.sample_bytes,
    )
    transfer = transport.stats()
    receipts = transport_receipts(transport)
    index_raw_hashes = [
        item.get("sha256")
        for item in receipts
        if item.get("kind") == "json"
        and item.get("path") == "model.safetensors.index.json"
    ]
    index_raw_sha256 = index_raw_hashes[0] if len(index_raw_hashes) == 1 else None
    sampled_payload = sum(int(item["sampled_bytes"]) for item in samples)
    return {
        "format": PROBE_FORMAT,
        "format_version": PROBE_VERSION,
        "canonical_lira": False,
        "status": "completed_bounded_tensor_microsample",
        "probe_id": str(uuid.uuid4()),
        "created_at": _utc_now(),
        "source": {
            "repo_id": SOURCE_MODEL,
            "revision": SOURCE_REVISION,
            "license": SOURCE_LICENSE,
            "index_sha256": index_sha256,
            "index_sha256_scope": "canonical_parsed_json",
            "index_raw_sha256": index_raw_sha256,
        },
        "transfer": {
            "http_requests": transfer.requests,
            "total_bytes_received": transfer.bytes_received,
            "tensor_payload_bytes_sampled": sampled_payload,
            "full_shards_downloaded": 0,
            "donor_model_instantiated": False,
            "donor_forward_calls": 0,
            "accepted_transfer_receipts": list(receipts),
        },
        "samples": samples,
        "scientific_limits": {
            "standalone_parameter_can_generate_text": False,
            "sample_classifies_structure_and_storage_only": True,
            "sample_decodes_semantic_knowledge": False,
            "sample_decodes_hidden_chain_of_thought": False,
            "whole_tensor_integrity_proven": False,
            "note": (
                "A micro-sample can fingerprint storage and confirm structural role. Functional "
                "meaning requires the surrounding network, prompt-conditioned activations, and "
                "causal interventions."
            ),
        },
    }


def print_report(report: Mapping[str, Any], destination: Path) -> None:
    transfer = report["transfer"]
    print("\nDragonBRX / Devorar Frontier — sonda de micropedaços")
    print(f"Tensores amostrados: {len(report['samples'])}")
    print(f"Payload real de pesos lido: {transfer['tensor_payload_bytes_sampled']} bytes")
    print(f"Total de rede (índice + headers + amostras): {transfer['total_bytes_received']} bytes")
    print("Shards completos: 0; inferência do doador: 0")
    for item in report["samples"]:
        print(
            f"  {item['tensor']}: {item['dtype']} {item['shape']} — "
            f"{item['sampled_bytes']} bytes / sha256={item['sample_fingerprint_sha256'][:12]}…"
        )
    print(f"Relatório: {destination}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
        _write_json_atomic(args.json_output, report)
        print_report(report, args.json_output.expanduser().absolute())
        return 0
    except (FrontierScanError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
