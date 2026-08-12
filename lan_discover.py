#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from src.lan_discovery import (
    DISCOVERY_PORT,
    MAX_PACKET_BYTES,
    DiscoveryError,
    build_request,
    is_local_address,
    parse_response,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find a Devorar coordinator automatically on the current local network.")
    parser.add_argument("--discovery-port", type=int, default=DISCOVERY_PORT)
    parser.add_argument("--scan-seconds", type=float, default=2.0)
    parser.add_argument("--retry-seconds", type=float, default=2.0)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{secrets.token_hex(8)}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def discover(port: int, scan_seconds: float) -> dict[str, Any] | None:
    nonce = secrets.token_hex(16)
    request = build_request(nonce)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", 0))
        sock.settimeout(0.25)
        sock.sendto(request, ("255.255.255.255", port))
        deadline = time.monotonic() + scan_seconds
        while time.monotonic() < deadline:
            try:
                payload, address = sock.recvfrom(MAX_PACKET_BYTES)
            except socket.timeout:
                continue
            source_ip = str(address[0])
            if not is_local_address(source_ip):
                continue
            try:
                response = parse_response(payload, expected_nonce=nonce)
            except DiscoveryError:
                continue
            response["source_ip"] = source_ip
            response["server_url"] = f"http://{source_ip}:{response['coordinator_port']}"
            return response
        return None
    finally:
        sock.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.discovery_port <= 65535:
        print("ERROR: discovery port must be within [1, 65535]", file=sys.stderr)
        return 2
    if not 0.2 <= args.scan_seconds <= 30 or not 0.2 <= args.retry_seconds <= 60:
        print("ERROR: invalid discovery timing", file=sys.stderr)
        return 2

    print("Procurando um PC Devorar na mesma rede Wi-Fi...", flush=True)
    attempt = 0
    while True:
        attempt += 1
        try:
            result = discover(args.discovery_port, args.scan_seconds)
        except OSError as error:
            print(f"Rede indisponível ({error}); nova tentativa em {args.retry_seconds:.0f}s.", flush=True)
            time.sleep(args.retry_seconds)
            continue
        if result is not None:
            _write_json_atomic(args.json_output, result)
            print(
                f"PC encontrado: {result['pc_name']} em {result['source_ip']}. Conectando automaticamente...",
                flush=True,
            )
            return 0
        if attempt == 1:
            print("Nenhum PC Devorar respondeu ainda. O celular continuará procurando automaticamente.", flush=True)
        time.sleep(args.retry_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
