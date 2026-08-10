#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.distributed_cluster import ClusterError, ClusterState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export distributed Devorar teacher-slice results from the PC coordinator database.")
    parser.add_argument("--state-dir", type=Path, default=Path("cluster-state"))
    parser.add_argument("--output", type=Path, default=Path("cluster-state/teacher-head-samples.jsonl"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        state = ClusterState(args.state_dir.expanduser().absolute() / "cluster.sqlite3")
        records = state.results()
        destination = args.output.expanduser().absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for record in records:
                result = record["result"]
                for sample in result.get("samples", []):
                    handle.write(
                        json.dumps(
                            {
                                "job_id": record["job_id"],
                                "worker_id": record["worker_id"],
                                "source": result.get("source"),
                                "tensor": result.get("tensor"),
                                "shape": result.get("shape"),
                                "seed": sample.get("seed"),
                                "hidden_float32_sha256": sample.get("hidden_float32_sha256"),
                                "row_logits": sample.get("row_logits"),
                                "scientific_scope": result.get("scientific_scope"),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n"
                    )
        print(f"Resultados: {len(records)} jobs")
        print(f"Dataset numérico: {destination}")
        print("Limite: isto é supervisão de fatias da cabeça de saída para ativações sintéticas, não respostas completas do DeepSeek.")
        return 0
    except (ClusterError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
