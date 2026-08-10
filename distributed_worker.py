#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import platform
import secrets
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.distributed_cluster import ClusterError, canonical_json_bytes, sign_request
from src.frontier_compute import (
    deterministic_float32_hidden_seeded,
    float32_sha256,
    load_complete_bf16_rows,
    reference_linear_logits,
)
from src.frontier_stream import (
    FrontierScanError,
    HuggingFaceRangeTransport,
    TensorDescriptor,
    TensorLocation,
    token_from_environment,
    transport_receipts,
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Termux/PC worker for the Devorar distributed coordinator.")
    parser.add_argument("--server", required=True, help="Coordinator URL, for example http://192.168.1.10:8765")
    parser.add_argument("--name", default=socket.gethostname())
    parser.add_argument("--processes", type=_bounded_int(1, 16), default=1)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--token-env", default="DEVORAR_CLUSTER_TOKEN")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--once", action="store_true")
    return parser


class CoordinatorClient:
    def __init__(self, base_url: str, token: str, *, timeout: float = 60.0) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ClusterError("--server must be an http:// or https:// URL")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        body = b"" if payload is None else canonical_json_bytes(payload)
        timestamp = f"{time.time():.6f}"
        nonce = secrets.token_hex(16)
        signature = sign_request(self.token, method, path, body, timestamp, nonce)
        headers = {
            "User-Agent": "DragonBRX-Devorar-Worker/1",
            "X-Devorar-Timestamp": timestamp,
            "X-Devorar-Nonce": nonce,
            "X-Devorar-Signature": signature,
        }
        if body:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base_url + path, data=body if method != "GET" else None, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = response.read(8 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as error:
            detail = error.read(4096).decode("utf-8", errors="replace")
            raise ClusterError(f"coordinator HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise ClusterError(f"coordinator unavailable: {error.reason}") from error
        if len(data) > 8 * 1024 * 1024:
            raise ClusterError("coordinator response is too large")
        value = json.loads(data.decode("utf-8"))
        if not isinstance(value, dict) or not value.get("ok"):
            raise ClusterError(str(value.get("error", "invalid coordinator response")) if isinstance(value, dict) else "invalid coordinator response")
        return value


def _worker_meta(slot: int) -> dict[str, Any]:
    return {
        "slot": slot,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "pid": os.getpid(),
    }


def _build_location(spec: Mapping[str, Any]) -> TensorLocation:
    tensor = spec.get("tensor")
    if not isinstance(tensor, dict):
        raise ClusterError("job tensor metadata is missing")
    shape_raw = tensor.get("shape")
    if not isinstance(shape_raw, list) or not all(isinstance(value, int) and value > 0 for value in shape_raw):
        raise ClusterError("job tensor shape is invalid")
    descriptor = TensorDescriptor(
        name=str(tensor.get("name", "")),
        shard=str(tensor.get("shard", "")),
        dtype=str(tensor.get("dtype", "")),
        shape=tuple(shape_raw),
        data_start=int(tensor.get("data_start")),
        data_end=int(tensor.get("data_end")),
    )
    return TensorLocation(
        descriptor=descriptor,
        data_origin=int(tensor.get("data_origin")),
        shard_file_bytes=int(tensor.get("shard_file_bytes")),
    )


def execute_remote_head_job(spec: Mapping[str, Any], hf_token_env: str) -> dict[str, Any]:
    if spec.get("operation") != "remote_bf16_head_teacher":
        raise ClusterError("unsupported job operation")
    source = spec.get("source")
    if not isinstance(source, dict):
        raise ClusterError("job source metadata is missing")
    location = _build_location(spec)
    if location.descriptor.dtype != "BF16" or len(location.descriptor.shape) != 2:
        raise ClusterError("worker supports only 2-D BF16 teacher tensors")
    row_ids_raw = spec.get("row_ids")
    seeds_raw = spec.get("sample_seeds")
    if not isinstance(row_ids_raw, list) or not row_ids_raw:
        raise ClusterError("job row_ids are missing")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ClusterError("job sample_seeds are missing")
    row_ids = tuple(int(value) for value in row_ids_raw)
    seeds = tuple(int(value) for value in seeds_raw)
    transport = HuggingFaceRangeTransport(
        str(source.get("repo_id", "")),
        str(source.get("revision", "")),
        token=token_from_environment(hf_token_env),
    )
    started = time.perf_counter()
    loaded = load_complete_bf16_rows(transport, location, row_ids)
    fetch_done = time.perf_counter()
    samples = []
    for seed in seeds:
        hidden = deterministic_float32_hidden_seeded(loaded.source_shape[1], seed)
        logits = reference_linear_logits(loaded.values, hidden)
        samples.append(
            {
                "seed": seed,
                "hidden_float32_sha256": float32_sha256(hidden),
                "row_logits": [
                    {"row_id": row_id, "value": value}
                    for row_id, value in zip(loaded.row_ids, logits)
                ],
            }
        )
    finished = time.perf_counter()
    stats = transport.stats()
    return {
        "operation": "remote_bf16_head_teacher",
        "source": {
            "repo_id": source.get("repo_id"),
            "revision": source.get("revision"),
            "index_sha256": source.get("index_sha256"),
        },
        "tensor": loaded.tensor,
        "dtype": loaded.dtype,
        "shape": list(loaded.source_shape),
        "row_ids": list(loaded.row_ids),
        "weight_payload_bytes": loaded.payload_bytes,
        "weight_payload_sha256": loaded.payload_sha256,
        "samples": samples,
        "timing_seconds": {
            "fetch": fetch_done - started,
            "compute": finished - fetch_done,
            "total": finished - started,
        },
        "transfer": {
            "http_requests": stats.requests,
            "bytes_received": stats.bytes_received,
            "accepted_transfer_receipts": list(transport_receipts(transport)),
        },
        "scientific_scope": "bounded_output_head_slice_only_not_full_deepseek_teacher",
    }


def worker_loop(args: argparse.Namespace, slot: int) -> int:
    token = os.getenv(args.token_env, "").strip()
    if len(token) < 24:
        raise ClusterError(f"set {args.token_env} to the token created by the PC coordinator")
    client = CoordinatorClient(args.server, token)
    registration = client.request(
        "POST",
        "/v1/register",
        {"name": f"{args.name}-{slot}", "meta": _worker_meta(slot)},
    )
    worker_id = str(registration["worker_id"])
    print(f"[{args.name}-{slot}] conectado como {worker_id}", flush=True)
    completed = 0
    while True:
        claim = client.request("POST", "/v1/claim", {"worker_id": worker_id})
        job = claim.get("job")
        if job is None:
            if args.once:
                print(f"[{args.name}-{slot}] sem jobs pendentes", flush=True)
                return 0
            time.sleep(max(0.2, float(args.poll_seconds)))
            continue
        job_id = str(job["job_id"])
        spec_hash = str(job["spec_sha256"])
        try:
            result = execute_remote_head_job(job["spec"], args.hf_token_env)
            client.request(
                "POST",
                "/v1/result",
                {
                    "worker_id": worker_id,
                    "job_id": job_id,
                    "spec_sha256": spec_hash,
                    "ok": True,
                    "result": result,
                },
            )
            completed += 1
            timing = result["timing_seconds"]
            print(
                f"[{args.name}-{slot}] {job_id} concluído: rows={len(result['row_ids'])} samples={len(result['samples'])} total={timing['total']:.2f}s",
                flush=True,
            )
            if args.once:
                return 0
        except Exception as error:
            try:
                client.request(
                    "POST",
                    "/v1/result",
                    {
                        "worker_id": worker_id,
                        "job_id": job_id,
                        "spec_sha256": spec_hash,
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
            except Exception:
                pass
            print(f"[{args.name}-{slot}] erro em {job_id}: {error}", file=sys.stderr, flush=True)
            if args.once:
                return 2


def _process_entry(args: argparse.Namespace, slot: int) -> None:
    try:
        raise SystemExit(worker_loop(args, slot))
    except (ClusterError, FrontierScanError, OSError, ValueError) as error:
        print(f"ERROR worker {slot}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.poll_seconds) or not 0.2 <= args.poll_seconds <= 300:
        print("ERROR: --poll-seconds must be within [0.2, 300]", file=sys.stderr)
        return 2
    if args.processes == 1:
        try:
            return worker_loop(args, 0)
        except (ClusterError, FrontierScanError, OSError, ValueError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
    processes = [
        multiprocessing.Process(target=_process_entry, args=(args, slot), name=f"devorar-worker-{slot}")
        for slot in range(args.processes)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()
        return 130
    return max((process.exitcode or 0) for process in processes)


if __name__ == "__main__":
    raise SystemExit(main())
