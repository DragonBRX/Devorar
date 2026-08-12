#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import sys
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

import distributed_server as base
from src.distributed_cluster import ClusterError, ClusterState, canonical_json_bytes, utc_timestamp
from src.frontier_stream import FrontierScanError
from src.lan_discovery import (
    DISCOVERY_PORT,
    MAX_PACKET_BYTES,
    DiscoveryError,
    build_response,
    is_local_address,
    parse_request,
)
from src.system_info import collect_system_info, describe_system, format_bytes


class HardwareCoordinatorHandler(base.CoordinatorHandler):
    def do_GET(self) -> None:
        try:
            if self.path == "/health":
                self._json(200, {"ok": True, "protocol_version": 1})
                return
            body = b""
            self._authenticate(body)
            if self.path == "/v1/status":
                payload = self.app.state.status()
                payload["ok"] = True
                payload["coordinator"] = collect_system_info(device_id="pc-coordinator")
                self._json(200, payload)
                return
            self._json(404, {"ok": False, "error": "not found"})
        except ClusterError as error:
            self._json(401, {"ok": False, "error": str(error)})

    def do_POST(self) -> None:
        if self.path != "/v1/heartbeat":
            return super().do_POST()
        try:
            body = self._read_body()
            self._authenticate(body)
            request = self._parse_json(body)
            worker_id = str(request.get("worker_id", ""))
            meta = request.get("meta")
            self.app.state.heartbeat(worker_id)
            if isinstance(meta, Mapping):
                payload = canonical_json_bytes(meta).decode("utf-8")
                with self.app.state._connection() as connection:
                    cursor = connection.execute(
                        "UPDATE workers SET last_seen=?, meta_json=? WHERE worker_id=?",
                        (utc_timestamp(), payload, worker_id),
                    )
                    if cursor.rowcount != 1:
                        raise ClusterError("unknown worker")
            self._json(200, {"ok": True})
        except ClusterError as error:
            self._json(400, {"ok": False, "error": str(error)})
        except Exception as error:
            self._json(500, {"ok": False, "error": f"internal error: {type(error).__name__}"})


def _termux_install_block() -> str:
    continuation = " " + chr(92)
    return "\n".join(
        (
            "pkg update -y && pkg install -y python git tmux &&" + continuation,
            'if [ -d "$HOME/Devorar/.git" ]; then git -C "$HOME/Devorar" pull --ff-only; else git clone --depth 1 https://github.com/DragonBRX/Devorar.git "$HOME/Devorar"; fi &&' + continuation,
            'cd "$HOME/Devorar" && chmod +x termux_auto_install.sh && ./termux_auto_install.sh',
        )
    )


def _termux_reinstall_block() -> str:
    continuation = " " + chr(92)
    return "\n".join(
        (
            "pkg update -y && pkg install -y python git tmux &&" + continuation,
            '(tmux kill-session -t devorar-worker 2>/dev/null || true) &&' + continuation,
            'cd "$HOME" && rm -rf "$HOME/Devorar" && git clone --depth 1 https://github.com/DragonBRX/Devorar.git "$HOME/Devorar" &&' + continuation,
            'cd "$HOME/Devorar" && chmod +x termux_auto_install.sh && ./termux_auto_install.sh',
        )
    )


def _physical_devices(status: Mapping[str, Any]) -> list[dict[str, Any]]:
    devices: dict[str, dict[str, Any]] = {}
    for worker in status.get("workers", []):
        if not isinstance(worker, Mapping):
            continue
        meta = worker.get("meta") if isinstance(worker.get("meta"), Mapping) else {}
        key = str(meta.get("device_id") or worker.get("worker_id") or worker.get("name"))
        age = float(worker.get("seconds_since_seen", 1e9))
        previous = devices.get(key)
        if previous is None or age < previous["age"]:
            devices[key] = {"age": age, "worker": worker, "meta": dict(meta), "slots": 1}
        else:
            previous["slots"] += 1
    return sorted(devices.values(), key=lambda item: item["age"])


def print_dashboard(state: ClusterState, reason: str = "atualização") -> None:
    status = state.status()
    pc = collect_system_info(device_id="pc-coordinator")
    jobs = status["jobs"]
    devices = _physical_devices(status)
    print(f"\n=== DEVORAR HARDWARE / {reason} ===", flush=True)
    print("PC:", describe_system(pc), flush=True)
    print(
        f"Jobs: fila={jobs['queued']} ativos={jobs['leased']} concluídos={jobs['done']} falhos={jobs['failed']} | dispositivos={len(devices)}",
        flush=True,
    )
    if not devices:
        print("- nenhum celular conectado ainda", flush=True)
    for item in devices:
        worker = item["worker"]
        meta = item["meta"]
        online = item["age"] <= 45.0
        label = meta.get("device_label") or worker.get("name") or "device"
        cpu = meta.get("cpu_model") or meta.get("machine") or "?"
        cores = meta.get("cpu_logical_cores") or meta.get("cpu_count") or "?"
        total = format_bytes(meta.get("ram_total_bytes"))
        available = format_bytes(meta.get("ram_available_bytes"))
        state_text = "ONLINE" if online else f"OFFLINE {item['age']:.0f}s"
        print(
            f"- {label} [{state_text}] | CPU: {cpu} | núcleos: {cores} | RAM: {total} total / {available} disponível",
            flush=True,
        )
    print("=== FIM HARDWARE ===\n", flush=True)


def _dashboard_loop(state: ClusterState, seconds: int, stop: threading.Event) -> None:
    while not stop.wait(seconds):
        try:
            print_dashboard(state)
        except Exception as error:
            print(f"Aviso: painel de hardware falhou: {error}", file=sys.stderr, flush=True)


def _prepare_jobs_loop(
    args: argparse.Namespace,
    state: ClusterState,
    state_dir: Path,
    stop: threading.Event,
) -> None:
    retry_seconds = 5
    while not stop.is_set():
        try:
            print("Preparando plano remoto do DeepSeek em segundo plano...", flush=True)
            jobs, plan_meta = base.build_jobs(args)
            added = state.add_jobs(jobs)
            (state_dir / "plan.json").write_text(
                json.dumps(plan_meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"Plano remoto pronto: {plan_meta['jobs']} jobs / {plan_meta['selected_rows']} linhas; novos jobs={added}",
                flush=True,
            )
            return
        except Exception as error:
            print(
                f"Aviso: não foi possível preparar os jobs agora ({type(error).__name__}: {error}). "
                f"O servidor continua ativo; nova tentativa em {retry_seconds}s.",
                file=sys.stderr,
                flush=True,
            )
            if stop.wait(retry_seconds):
                return
            retry_seconds = min(retry_seconds * 2, 60)


def _create_discovery_socket(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        sock.settimeout(0.5)
        return sock
    except BaseException:
        sock.close()
        raise


def _lan_discovery_loop(
    sock: socket.socket,
    stop: threading.Event,
    *,
    coordinator_port: int,
    token: str,
    worker_processes: int,
) -> None:
    pc_name = socket.gethostname() or "PC-Devorar"
    try:
        while not stop.is_set():
            try:
                payload, address = sock.recvfrom(MAX_PACKET_BYTES)
            except socket.timeout:
                continue
            except OSError:
                if stop.is_set():
                    return
                raise
            source_ip = str(address[0])
            if not is_local_address(source_ip):
                continue
            try:
                request = parse_request(payload)
                response = build_response(
                    request["nonce"],
                    coordinator_port=coordinator_port,
                    token=token,
                    worker_processes=worker_processes,
                    pc_name=pc_name,
                )
                sock.sendto(response, address)
                print(f"Descoberta LAN: celular encontrou este PC ({source_ip}).", flush=True)
            except DiscoveryError:
                continue
    finally:
        sock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = "Devorar Windows/PC coordinator with zero-config LAN discovery and Termux hardware telemetry."
    parser.add_argument("--status-seconds", type=int, default=15)
    parser.add_argument("--discovery-port", type=int, default=DISCOVERY_PORT)
    parser.add_argument("--no-lan-discovery", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.status_seconds <= 300:
        print("ERROR: --status-seconds must be within [0, 300]", file=sys.stderr)
        return 2
    if not 1 <= args.discovery_port <= 65535:
        print("ERROR: --discovery-port must be within [1, 65535]", file=sys.stderr)
        return 2
    state_dir = args.state_dir.expanduser().absolute()
    state_dir.mkdir(parents=True, exist_ok=True)
    discovery_socket: socket.socket | None = None
    try:
        token, token_file, token_created = base._resolve_token(args, state_dir)
        state = ClusterState(
            state_dir / "cluster.sqlite3",
            lease_seconds=args.lease_seconds,
            max_attempts=args.max_attempts,
        )

        base.CoordinatorHandler = HardwareCoordinatorHandler
        server = base.CoordinatorHTTPServer((args.host, args.port), state, token)
        if not args.no_lan_discovery:
            discovery_socket = _create_discovery_socket(args.discovery_port)

        print(f"Coordenador Devorar ativo na rede local (TCP {args.port}).", flush=True)
        if discovery_socket is not None:
            print(
                f"Descoberta automática ativa (UDP {args.discovery_port}). No Termux não é necessário digitar IP, porta ou token.",
                flush=True,
            )
        if token_file is not None:
            print(f"Token privado do cluster: {token_file}", flush=True)
            if token_created:
                print("Novo token criado para este cluster.", flush=True)

        print_dashboard(state, "inicialização")
        if not args.no_termux_block:
            print("=== TERMUX: INSTALAÇÃO AUTOMÁTICA NA MESMA WI-FI ===", flush=True)
            print(_termux_install_block(), flush=True)
            print("=== FIM INSTALAÇÃO TERMUX ===\n", flush=True)
            print("=== TERMUX: REINSTALAÇÃO AUTOMÁTICA ===", flush=True)
            print(_termux_reinstall_block(), flush=True)
            print("=== FIM REINSTALAÇÃO TERMUX ===\n", flush=True)
            print(
                "Os blocos não contêm IP, porta nem token. O celular encontra este PC automaticamente pela rede local.",
                flush=True,
            )

        stop = threading.Event()
        discovery_thread = None
        if discovery_socket is not None:
            discovery_thread = threading.Thread(
                target=_lan_discovery_loop,
                args=(discovery_socket, stop),
                kwargs={
                    "coordinator_port": args.port,
                    "token": token,
                    "worker_processes": args.termux_processes,
                },
                daemon=True,
                name="devorar-lan-discovery",
            )
            discovery_thread.start()
            discovery_socket = None

        dashboard_thread = None
        if args.status_seconds:
            dashboard_thread = threading.Thread(
                target=_dashboard_loop,
                args=(state, args.status_seconds, stop),
                daemon=True,
                name="devorar-dashboard",
            )
            dashboard_thread.start()

        jobs_thread = None
        if not args.no_create_jobs:
            jobs_thread = threading.Thread(
                target=_prepare_jobs_loop,
                args=(args, state, state_dir, stop),
                daemon=True,
                name="devorar-job-preparation",
            )
            jobs_thread.start()

        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            server.server_close()
            if discovery_thread is not None:
                discovery_thread.join(timeout=1.0)
            if dashboard_thread is not None:
                dashboard_thread.join(timeout=1.0)
            if jobs_thread is not None:
                jobs_thread.join(timeout=1.0)
        return 0
    except (ClusterError, FrontierScanError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2
    finally:
        if discovery_socket is not None:
            discovery_socket.close()


if __name__ == "__main__":
    raise SystemExit(main())
