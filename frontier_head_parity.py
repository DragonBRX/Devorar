#!/usr/bin/env python3
"""Execute real DeepSeek-V4-Flash output-head rows and demand numeric parity."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.frontier_compute import (
    compare_logits,
    deterministic_float32_hidden,
    float32_sha256,
    load_complete_bf16_rows,
    reference_linear_logits,
    select_evenly_spaced_rows,
)
from src.frontier_stream import (
    FrontierScanError,
    HuggingFaceRangeTransport,
    locate_tensors,
    token_from_environment,
    transport_receipts,
)


SOURCE_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SOURCE_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
SOURCE_LICENSE = "MIT"
SOURCE_TENSOR = "head.weight"
REPORT_FORMAT = "lira.experimental.frontier-head-slice-parity"
REPORT_VERSION = 1
DEFAULT_ROW_COUNT = 16
DEFAULT_ATOL = 2e-4
DEFAULT_RTOL = 2e-4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_output() -> Path:
    content = Path("/content")
    return (
        content / "dragonbrx-frontier-head-parity.lira.json"
        if content.is_dir()
        else Path.cwd() / "dragonbrx-frontier-head-parity.lira.json"
    )


def _bounded_int(minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            result = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be an integer") from error
        if not minimum <= result <= maximum:
            raise argparse.ArgumentTypeError(f"must be within [{minimum}, {maximum}]")
        return result

    return parse


def _bounded_float(minimum: float, maximum: float):
    def parse(value: str) -> float:
        try:
            result = float(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be a number") from error
        if not math.isfinite(result) or not minimum <= result <= maximum:
            raise argparse.ArgumentTypeError(
                f"must be finite and within [{minimum}, {maximum}]"
            )
        return result

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream complete rows from the pinned DeepSeek-V4-Flash output head, execute a "
            "bounded linear projection, and compare device logits with an independent CPU "
            "float64 reference. This is not full-model logit parity."
        )
    )
    parser.add_argument("--row-count", type=_bounded_int(1, 64), default=DEFAULT_ROW_COUNT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless the candidate operator actually runs on a CUDA device",
    )
    parser.add_argument("--atol", type=_bounded_float(0.0, 0.1), default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=_bounded_float(0.0, 0.1), default=DEFAULT_RTOL)
    parser.add_argument("--json-output", type=Path, default=_default_output())
    parser.add_argument("--timeout-seconds", type=_bounded_float(1.0, 600.0), default=60.0)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    return parser


def _candidate_logits(
    weights: Sequence[Sequence[float]],
    hidden: Sequence[float],
    *,
    requested_device: str,
    require_cuda: bool,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    try:
        import torch
    except ImportError as error:
        raise FrontierScanError("PyTorch is required for the candidate linear operator") from error

    cuda_available = bool(torch.cuda.is_available())
    if requested_device == "cuda" and not cuda_available:
        raise FrontierScanError("CUDA was requested but PyTorch cannot access a CUDA device")
    use_cuda = requested_device == "cuda" or (
        requested_device == "auto" and cuda_available
    )
    selected = "cuda" if use_cuda else "cpu"
    if require_cuda and selected != "cuda":
        raise FrontierScanError("--require-cuda was set, but no CUDA execution device is available")

    if selected == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    with torch.inference_mode():
        matrix = torch.tensor(weights, dtype=torch.float32, device=selected)
        vector = torch.tensor(hidden, dtype=torch.float32, device=selected)
        output = torch.nn.functional.linear(vector, matrix)
        if selected == "cuda":
            torch.cuda.synchronize()
        values = tuple(float(value) for value in output.detach().cpu().tolist())

    metadata: dict[str, Any] = {
        "framework": "pytorch",
        "torch_version": str(torch.__version__),
        "requested_device": requested_device,
        "selected_device": selected,
        "cuda_available": cuda_available,
        "dtype": "float32",
        "tf32_allowed": False if selected == "cuda" else None,
    }
    if selected == "cuda":
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        metadata.update(
            {
                "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                "compute_capability": [int(properties.major), int(properties.minor)],
                "total_device_memory_bytes": int(properties.total_memory),
            }
        )
    else:
        metadata["device_name"] = platform.processor() or platform.machine() or "CPU"
    return values, metadata


def _write_json_atomic(path: Path, report: Mapping[str, Any]) -> None:
    destination = path.expanduser().absolute()
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise FrontierScanError(f"parity output must be a regular file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def run(args: argparse.Namespace) -> dict[str, Any]:
    transport = HuggingFaceRangeTransport(
        SOURCE_MODEL,
        SOURCE_REVISION,
        token=token_from_environment(args.hf_token_env),
        timeout_seconds=args.timeout_seconds,
    )
    locations, index_sha256 = locate_tensors(transport, (SOURCE_TENSOR,))
    location = locations[SOURCE_TENSOR]
    descriptor = location.descriptor
    if len(descriptor.shape) != 2:
        raise FrontierScanError("pinned output head is not a matrix")
    row_ids = select_evenly_spaced_rows(descriptor.shape[0], args.row_count)
    loaded = load_complete_bf16_rows(transport, location, row_ids)
    hidden = deterministic_float32_hidden(loaded.source_shape[1])
    reference = reference_linear_logits(loaded.values, hidden)
    candidate, device = _candidate_logits(
        loaded.values,
        hidden,
        requested_device=args.device,
        require_cuda=args.require_cuda,
    )
    comparison = compare_logits(reference, candidate, atol=args.atol, rtol=args.rtol)
    for row_id, result in zip(loaded.row_ids, comparison["rows"]):
        result["source_row_id"] = row_id

    transfer = transport.stats()
    report = {
        "format": REPORT_FORMAT,
        "format_version": REPORT_VERSION,
        "canonical_lira": False,
        "status": (
            "passed_bounded_head_slice_parity"
            if comparison["passed"]
            else "failed_bounded_head_slice_parity"
        ),
        "test_id": str(uuid.uuid4()),
        "created_at": _utc_now(),
        "source": {
            "repo_id": SOURCE_MODEL,
            "revision": SOURCE_REVISION,
            "license": SOURCE_LICENSE,
            "index_sha256": index_sha256,
            "tensor": loaded.tensor,
            "shard": loaded.shard,
            "dtype": loaded.dtype,
            "shape": list(loaded.source_shape),
        },
        "executed_slice": {
            "operation": (
                "partial_output_head_logits = "
                "selected_head_rows_fp32 @ synthetic_hidden_fp32"
            ),
            "complete_source_rows": True,
            "selected_row_ids": list(loaded.row_ids),
            "selected_row_count": len(loaded.row_ids),
            "hidden_width": len(hidden),
            "weight_payload_bytes": loaded.payload_bytes,
            "weight_payload_sha256": loaded.payload_sha256,
            "hidden_input_kind": "deterministic_synthetic_non_model_activation",
            "hidden_input_float32_sha256": float32_sha256(hidden),
            "row_range_receipts": list(loaded.range_receipts),
        },
        "reference": {
            "implementation": "python_math_fsum_cpu",
            "accumulation": "binary64_correctly_rounded_fsum",
            "weight_decode": "exact_bfloat16_to_float32_values",
            "input_storage": "float32",
        },
        "candidate": device,
        "parity": comparison,
        "transfer": {
            "http_requests": transfer.requests,
            "total_bytes_received": transfer.bytes_received,
            "tensor_payload_bytes_executed": loaded.payload_bytes,
            "full_shards_downloaded": 0,
            "donor_model_instantiated": False,
            "full_donor_forward_calls": 0,
            "accepted_transfer_receipts": list(transport_receipts(transport)),
        },
        "evidence": {
            "remote_checkpoint_bytes_became_numeric_weights": True,
            "real_source_tensor_rows_executed": True,
            "bounded_linear_operator_parity_passed": bool(comparison["passed"]),
            "cuda_operator_executed": device["selected_device"] == "cuda",
        },
        "scientific_limits": {
            "synthetic_hidden_is_a_real_deepseek_activation": False,
            "partial_slice_logits_are_full_model_logits": False,
            "official_deepseek_runtime_logits_compared": False,
            "complete_transformer_graph_executed": False,
            "next_token_or_phrase_attributable_to_deepseek": False,
            "hidden_chain_of_thought_accessed": False,
            "conclusion_allowed": (
                "The pinned checkpoint's complete selected BF16 output-head rows were streamed, "
                "decoded, and executed correctly for this declared linear operation."
            ),
            "next_required_gate": (
                "Execute the complete token path and compare full-vocabulary next-token logits "
                "against an authoritative runtime on identical tokens, revision, dtype, "
                "and cache state."
            ),
        },
    }
    return report


def print_report(report: Mapping[str, Any], destination: Path) -> None:
    parity = report["parity"]
    source = report["source"]
    executed = report["executed_slice"]
    candidate = report["candidate"]
    print("\nDragonBRX / Devorar Frontier — execução real de uma fatia")
    print(f"Fonte fixada: {source['repo_id']}@{source['revision']}")
    print(
        f"Tensor: {source['tensor']} {source['dtype']} {source['shape']} — "
        f"{executed['selected_row_count']} linhas completas"
    )
    print(f"Pesos executados por Range: {executed['weight_payload_bytes']} bytes")
    print(f"Dispositivo candidato: {candidate['selected_device']} / {candidate['device_name']}")
    print(
        f"Paridade da fatia: {'PASSOU' if parity['passed'] else 'FALHOU'} — "
        f"erro absoluto máximo={parity['max_absolute_error']:.9g}"
    )
    print(
        "Limite: estes são logits parciais para uma ativação sintética, "
        "não logits do modelo completo."
    )
    print(f"Relatório: {destination}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
        _write_json_atomic(args.json_output, report)
        print_report(report, args.json_output.expanduser().absolute())
        return 0 if report["parity"]["passed"] else 3
    except (FrontierScanError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
