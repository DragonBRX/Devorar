#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.distributed_cluster import (
    ClusterError,
    ClusterState,
    MAX_BODY_BYTES,
    chunked,
    evenly_spaced_indices,
    load_or_create_token,
    verify_request_signature,
)
from src.frontier_stream import (
    FrontierScanError,
    HuggingFaceRangeTransport,
    locate_tensors,
    token_from_environment,
)


DEFAULT_SOURCE_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_SOURCE_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
DEFAULT_TENSOR = "head.weight"


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
    parser = argparse.ArgumentParser(
        description="Run the Devorar PC coordinator and distribute bounded remote DeepSeek work to Termux workers."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=_bounded_int(1, 65535), default=8765)
    parser.add_argument("--state-dir", type=Path, default=Path("cluster-state"))
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--source-revision", default=DEFAULT_SOURCE_REVISION)
    parser.add_argument("--tensor", default=DEFAULT_TENSOR)
    parser.add_argument("--job-count", type=_bounded_int(1, 32768), default=256)
    parser.add_argument("--rows-per-job", type=_bounded_int(1, 64), default=4)
    parser.add_argument("--samples-per-job", type=_bounded_int(1, 64), default=8)
    parser.add_argument("--lease-seconds", type=_bounded_int(30, 86400), default=900)
    parser.add_argument("--max-attempts", type=_bounded_int(1, 100), default=5)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--token-env", default="DEVORAR_CLUSTER_TOKEN")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--no-create-jobs", action="store_true")
    parser.add_argument("--advertise-host", help="LAN IP/hostname printed in the automatic Termux install block")
    parser.add_argument("--termux-processes", type=_bounded_int(1, 16), default=1)
    parser.add_argument("--no-termux-block", action="store_true")
    return parser


def build_jobs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transport = HuggingFaceRangeTransport(
        args.source_model,
        args.source_revision,
        token=token_from_environment(args.hf_token_env),
    )
    locations, index_sha256 = locate_tensors(transport, (args.tensor,))
    location = locations[args.tensor]
    descriptor = location.descriptor
    if descriptor.dtype != "BF16" or len(descriptor.shape) != 2:
        raise FrontierScanError("distributed head teacher jobs currently require a 2-D BF16 tensor")
    total_rows, width = descriptor.shape
    requested_rows = min(total_rows, args.job_count * args.rows_per_job)
    selected = evenly_spaced_indices(total_rows, requested_rows)
    groups = list(chunked(selected, args.rows_per_job))
    specs: list[dict[str, Any]] = []
    for index, row_ids in enumerate(groups):
        specs.append(
            {
                "protocol_version": 1,
                "operation": "remote_bf16_head_teacher",
                "source": {
                    "repo_id": args.source_model,
                    "revision": args.source_revision,
                    "index_sha256": index_sha256,
                },
                "tensor": {
                    "name": descriptor.name,
                    "shard": descriptor.shard,
                    "dtype": descriptor.dtype,
                    "shape": list(descriptor.shape),
                    "data_start": descriptor.data_start,
                    "data_end": descriptor.data_end,
                    "data_origin": location.data_origin,
                    "shard_file_bytes": location.shard_file_bytes,
                },
                "row_ids": list(row_ids),
                "sample_seeds": [offset + 1 for offset in range(args.samples_per_job)],
                "scientific_scope": "bounded_output_head_slice_only_not_full_deepseek_teacher",
            }
        )
    metadata = {
        "source_model": args.source_model,
        "source_revision": args.source_revision,
        "tensor": args.tensor,
        "tensor_shape": list(descriptor.shape),
        "selected_rows": requested_rows,
        "jobs": len(specs),
        "samples_per_job": args.samples_per_job,
        "hidden_width": width,
        "index_sha256": index_sha256,
        "remote_scan_bytes_received": transport.stats().bytes_received,
    }
    return specs, metadata


class CoordinatorHandler(BaseHTTPRequestHandler):
    server_version = "DevorarCluster/1"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))
        sys.stdout.flush()

    @property
    def app(self) -> "CoordinatorHTTPServer":
        return self.server  # type: ignore[return-value]

    def _read_body(self) -> bytes:
        raw = self.headers.get("Content-Length")
        if raw is None:
            return b""
        try:
            length = int(raw)
        except ValueError as error:
            raise ClusterError("invalid Content-Length") from error
        if not 0 <= length <= MAX_BODY_BYTES:
            raise ClusterError("request body is too large")
        return self.rfile.read(length)

    def _authenticate(self, body: bytes) -> None:
        timestamp = self.headers.get("X-Devorar-Timestamp", "")
        nonce = self.headers.get("X-Devorar-Nonce", "")
        signature = self.headers.get("X-Devorar-Signature", "")
        verify_request_signature(self.app.token, self.command, self.path, body, timestamp, nonce, signature)
        if not self.app.accept_nonce(nonce):
            raise ClusterError("replayed request nonce")

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _parse_json(self, body: bytes) -> Mapping[str, Any]:
        try:
            value = json.loads(body.decode("utf-8")) if body else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ClusterError("invalid JSON request") from error
        if not isinstance(value, dict):
            raise ClusterError("JSON request must be an object")
        return value

    def do_GET(self) -> None:
        try:
            if self.path == "/health":
                self._json(200, {"ok": True, "protocol_version": 1})
                return
            body = b""
            self._authenticate(body)
            if self.path == "/v1/status":
                self._json(200, self.app.state.status())
                return
            self._json(404, {"ok": False, "error": "not found"})
        except ClusterError as error:
            self._json(401, {"ok": False, "error": str(error)})

    def do_POST(self) -> None:
        try:
            body = self._read_body()
            self._authenticate(body)
            request = self._parse_json(body)
            if self.path == "/v1/register":
                worker_id = self.app.state.register_worker(str(request.get("name", "")), request.get("meta", {}))
                self._json(200, {"ok": True, "worker_id": worker_id, "protocol_version": 1})
                return
            if self.path == "/v1/heartbeat":
                self.app.state.heartbeat(str(request.get("worker_id", "")))
                self._json(200, {"ok": True})
                return
            if self.path == "/v1/claim":
                job = self.app.state.claim_job(str(request.get("worker_id", "")))
                self._json(200, {"ok": True, "job": None if job is None else job.to_dict()})
                return
            if self.path == "/v1/result":
                self.app.state.submit_result(
                    str(request.get("worker_id", "")),
                    str(request.get("job_id", "")),
                    str(request.get("spec_sha256", "")),
                    ok=bool(request.get("ok")),
                    result=request.get("result") if isinstance(request.get("result"), dict) else None,
                    error=str(request.get("error", "")) if request.get("error") is not None else None,
                )
                self._json(200, {"ok": True})
                return
            self._json(404, {"ok": False, "error": "not found"})
        except ClusterError as error:
            self._json(400, {"ok": False, "error": str(error)})
        except Exception as error:
            self._json(500, {"ok": False, "error": f"internal error: {type(error).__name__}"})


class CoordinatorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: ClusterState, token: str):
        super().__init__(address, CoordinatorHandler)
        self.state = state
        self.token = token
        self._nonces: dict[str, float] = {}

    def accept_nonce(self, nonce: str) -> bool:
        import time

        now = time.time()
        self._nonces = {key: created for key, created in self._nonces.items() if now - created < 600}
        if nonce in self._nonces:
            return False
        self._nonces[nonce] = now
        return True


def _resolve_token(args: argparse.Namespace, state_dir: Path) -> tuple[str, Path | None, bool]:
    environment = os.getenv(args.token_env, "").strip()
    if environment:
        if len(environment) < 24:
            raise ClusterError(f"{args.token_env} is too short")
        return environment, None, False
    token_file = args.token_file or (state_dir / "cluster-token.txt")
    token, created = load_or_create_token(token_file)
    return token, token_file.expanduser().absolute(), created


def _detect_lan_host() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        candidate = str(sock.getsockname()[0])
        if candidate and not candidate.startswith("127."):
            return candidate
    except OSError:
        pass
    finally:
        sock.close()
    try:
        candidate = socket.gethostbyname(socket.gethostname())
        if candidate and not candidate.startswith("127."):
            return candidate
    except OSError:
        pass
    return "IP_DO_PC"


def build_termux_install_block(server_url: str, token: str, processes: int) -> str:
    if not server_url.startswith(("http://", "https://")):
        raise ClusterError("Termux server URL must be http:// or https://")
    if len(token) < 24:
        raise ClusterError("cluster token is too short for Termux install block")
    if not 1 <= processes <= 16:
        raise ClusterError("Termux process count must be within [1, 16]")
    env_server = shlex.quote(server_url)
    env_token = shlex.quote(token)
    env_processes = shlex.quote(str(processes))
    continuation = " " + chr(92)
    return "\n".join(
        (
            "pkg update -y && pkg install -y python git tmux &&" + continuation,
            'if [ -d "$HOME/Devorar/.git" ]; then git -C "$HOME/Devorar" pull --ff-only; else git clone --depth 1 https://github.com/DragonBRX/Devorar.git "$HOME/Devorar"; fi &&' + continuation,
            'cd "$HOME/Devorar" && chmod +x termux_install.sh &&' + continuation,
            f"DEVORAR_SERVER={env_server} DEVORAR_CLUSTER_TOKEN={env_token} DEVORAR_PROCESSES={env_processes} ./termux_install.sh",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state_dir = args.state_dir.expanduser().absolute()
    state_dir.mkdir(parents=True, exist_ok=True)
    try:
        token, token_file, token_created = _resolve_token(args, state_dir)
        state = ClusterState(
            state_dir / "cluster.sqlite3",
            lease_seconds=args.lease_seconds,
            max_attempts=args.max_attempts,
        )
        plan_meta: dict[str, Any] | None = None
        if not args.no_create_jobs:
            jobs, plan_meta = build_jobs(args)
            added = state.add_jobs(jobs)
            (state_dir / "plan.json").write_text(
                json.dumps(plan_meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"Plano remoto: {plan_meta['jobs']} jobs / {plan_meta['selected_rows']} linhas selecionadas; novos jobs={added}")
        server = CoordinatorHTTPServer((args.host, args.port), state, token)
        print(f"Coordenador Devorar: http://{args.host}:{args.port}")
        print(f"Estado: {state_dir / 'cluster.sqlite3'}")
        if token_file is not None:
            print(f"Token do cluster: {token_file}")
            if token_created:
                print("Copie o conteúdo desse arquivo para DEVORAR_CLUSTER_TOKEN em cada worker Termux.")
        else:
            print(f"Token do cluster: variável {args.token_env}")
        print("Workers acessam pesos por HTTP Range no Hugging Face e devolvem apenas resultados ao PC.")
        print("Este estágio executa fatias BF16 de head.weight; ainda não é o forward completo do DeepSeek.")
        if not args.no_termux_block:
            advertised_host = args.advertise_host or _detect_lan_host()
            server_url = f"http://{advertised_host}:{args.port}"
            print("\n=== TERMUX: COLE ESTE BLOCO INTEIRO EM CADA CELULAR ===")
            print(build_termux_install_block(server_url, token, args.termux_processes))
            print("=== FIM DO BLOCO TERMUX ===\n")

        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    except (ClusterError, FrontierScanError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
