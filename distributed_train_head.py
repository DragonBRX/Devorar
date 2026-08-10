#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.distributed_cluster import ClusterError, ClusterState
from src.frontier_compute import deterministic_float32_hidden_seeded


def _bounded_int(minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be an integer") from error
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"must be within [{minimum}, {maximum}]")
        return parsed

    return parse


def _bounded_float(minimum: float, maximum: float):
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be a number") from error
        if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"must be finite and within [{minimum}, {maximum}]")
        return parsed

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a compact low-rank surrogate of the sampled DeepSeek output-head rows collected by the distributed cluster."
    )
    parser.add_argument("--state-dir", type=Path, default=Path("cluster-state"))
    parser.add_argument("--output", type=Path, default=Path("cluster-state/student-head.safetensors"))
    parser.add_argument("--manifest", type=Path, default=Path("cluster-state/student-head-manifest.json"))
    parser.add_argument("--rank", type=_bounded_int(1, 256), default=8)
    parser.add_argument("--epochs", type=_bounded_int(1, 10000), default=1000)
    parser.add_argument("--learning-rate", type=_bounded_float(1e-6, 1.0), default=0.01)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def collect_training_matrix(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ClusterError("no completed distributed results are available")
    source_identity: tuple[str, str, str, int] | None = None
    values: dict[int, dict[int, float]] = {}
    all_rows: set[int] = set()
    for record in records:
        result = record.get("result")
        if not isinstance(result, dict) or result.get("operation") != "remote_bf16_head_teacher":
            continue
        source = result.get("source")
        shape = result.get("shape")
        if not isinstance(source, dict) or not isinstance(shape, list) or len(shape) != 2:
            raise ClusterError("distributed result has invalid source metadata")
        identity = (
            str(source.get("repo_id", "")),
            str(source.get("revision", "")),
            str(result.get("tensor", "")),
            int(shape[1]),
        )
        if source_identity is None:
            source_identity = identity
        elif source_identity != identity:
            raise ClusterError("results from different teacher tensors cannot be mixed")
        samples = result.get("samples")
        if not isinstance(samples, list):
            raise ClusterError("distributed result is missing samples")
        for sample in samples:
            if not isinstance(sample, dict):
                raise ClusterError("invalid sample record")
            seed = int(sample.get("seed"))
            row_map = values.setdefault(seed, {})
            row_logits = sample.get("row_logits")
            if not isinstance(row_logits, list):
                raise ClusterError("sample is missing row_logits")
            for item in row_logits:
                row_id = int(item["row_id"])
                value = float(item["value"])
                if not math.isfinite(value):
                    raise ClusterError("teacher target contains a non-finite value")
                previous = row_map.get(row_id)
                if previous is not None and previous != value:
                    raise ClusterError("conflicting teacher target for the same seed/row")
                row_map[row_id] = value
                all_rows.add(row_id)
    if source_identity is None or not values or not all_rows:
        raise ClusterError("no compatible teacher-head samples were found")
    row_ids = sorted(all_rows)
    seeds = sorted(values)
    for seed in seeds:
        if set(values[seed]) != set(row_ids):
            missing = len(set(row_ids) - set(values[seed]))
            raise ClusterError(f"training matrix is incomplete for seed {seed}: {missing} rows missing")
    width = source_identity[3]
    inputs = [deterministic_float32_hidden_seeded(width, seed) for seed in seeds]
    targets = [[values[seed][row_id] for row_id in row_ids] for seed in seeds]
    return {
        "source": {
            "repo_id": source_identity[0],
            "revision": source_identity[1],
            "tensor": source_identity[2],
        },
        "hidden_width": width,
        "seeds": seeds,
        "row_ids": row_ids,
        "inputs": inputs,
        "targets": targets,
    }


def train(matrix: Mapping[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import torch
    except ImportError as error:
        raise ClusterError("PyTorch is required on the PC for distributed_train_head.py") from error
    cuda = bool(torch.cuda.is_available())
    if args.device == "cuda" and not cuda:
        raise ClusterError("CUDA was requested but is not available")
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and cuda) else "cpu"
    x = torch.tensor(matrix["inputs"], dtype=torch.float32, device=device)
    y = torch.tensor(matrix["targets"], dtype=torch.float32, device=device)
    rank = min(args.rank, x.shape[0], x.shape[1], y.shape[1])
    if rank < 1:
        raise ClusterError("training matrix is too small for the requested low-rank student")
    down = torch.nn.Linear(x.shape[1], rank, bias=False, device=device, dtype=torch.float32)
    up = torch.nn.Linear(rank, y.shape[1], bias=False, device=device, dtype=torch.float32)
    torch.manual_seed(123456)
    torch.nn.init.normal_(down.weight, mean=0.0, std=0.01)
    torch.nn.init.zeros_(up.weight)
    optimizer = torch.optim.AdamW(list(down.parameters()) + list(up.parameters()), lr=args.learning_rate, weight_decay=0.0)
    initial_loss = None
    final_loss = None
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = up(down(x))
        loss = torch.nn.functional.mse_loss(prediction, y)
        if not torch.isfinite(loss):
            raise ClusterError("student training produced a non-finite loss")
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    with torch.inference_mode():
        prediction = up(down(x))
        mae = float(torch.mean(torch.abs(prediction - y)).detach().cpu())
        rmse = float(torch.sqrt(torch.mean((prediction - y) ** 2)).detach().cpu())
    artifact = {
        "down.weight": down.weight.detach().cpu().contiguous(),
        "up.weight": up.weight.detach().cpu().contiguous(),
        "row_ids": torch.tensor(matrix["row_ids"], dtype=torch.int64),
    }
    metrics = {
        "device": device,
        "torch_version": str(torch.__version__),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "sample_count": len(matrix["seeds"]),
        "selected_output_rows": len(matrix["row_ids"]),
        "hidden_width": int(matrix["hidden_width"]),
        "rank": int(rank),
        "initial_mse": initial_loss,
        "final_mse": final_loss,
        "final_mae": mae,
        "final_rmse": rmse,
    }
    return artifact, metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        state = ClusterState(args.state_dir.expanduser().absolute() / "cluster.sqlite3")
        matrix = collect_training_matrix(state.results())
        artifact, metrics = train(matrix, args)
        try:
            from safetensors.torch import save_file
        except ImportError as error:
            raise ClusterError("safetensors is required on the PC to save the student head") from error

        output = args.output.expanduser().absolute()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
        save_file(artifact, str(temporary), metadata={"format": "devorar.distributed.student-head", "format_version": "1"})
        os.replace(temporary, output)
        manifest = {
            "format": "devorar.distributed.student-head-manifest",
            "format_version": 1,
            "artifact": str(output),
            "source": matrix["source"],
            "training": metrics,
            "scientific_scope": {
                "actual_training_performed": True,
                "teacher_signal_from_remote_deepseek_weights": True,
                "teacher_signal_scope": "selected BF16 output-head rows on synthetic hidden vectors",
                "full_deepseek_forward_used": False,
                "language_model_created": False,
                "tokenizer_or_generation_supported": False,
                "claim_allowed": "compact low-rank surrogate of the sampled remote output-head operation",
            },
        }
        manifest_path = args.manifest.expanduser().absolute()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Student head: {output}")
        print(f"Manifesto: {manifest_path}")
        print(f"MSE: {metrics['initial_mse']:.6g} -> {metrics['final_mse']:.6g}")
        print("Isto treina um surrogate numérico da fatia coletada; ainda não cria um LLM capaz de conversar.")
        return 0
    except (ClusterError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
