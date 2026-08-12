from __future__ import annotations

import ipaddress
import json
from typing import Any, Mapping

DISCOVERY_PORT = 8764
PROTOCOL_VERSION = 1
REQUEST_MAGIC = "DEVORAR_DISCOVER"
RESPONSE_MAGIC = "DEVORAR_HERE"
MAX_PACKET_BYTES = 4096


class DiscoveryError(RuntimeError):
    pass


def _loads(payload: bytes) -> Mapping[str, Any]:
    if not isinstance(payload, (bytes, bytearray)) or not payload or len(payload) > MAX_PACKET_BYTES:
        raise DiscoveryError("invalid discovery packet size")
    try:
        value = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DiscoveryError("invalid discovery JSON") from error
    if not isinstance(value, dict):
        raise DiscoveryError("discovery packet must be a JSON object")
    return value


def _dumps(value: Mapping[str, Any]) -> bytes:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(payload) > MAX_PACKET_BYTES:
        raise DiscoveryError("discovery packet is too large")
    return payload


def _valid_nonce(value: Any) -> str:
    nonce = str(value or "")
    if not 16 <= len(nonce) <= 128 or any(character not in "0123456789abcdefABCDEF" for character in nonce):
        raise DiscoveryError("invalid discovery nonce")
    return nonce


def is_local_address(address: str) -> bool:
    try:
        value = ipaddress.ip_address(address)
    except ValueError:
        return False
    return bool(value.is_private or value.is_link_local or value.is_loopback)


def build_request(nonce: str) -> bytes:
    return _dumps({"magic": REQUEST_MAGIC, "protocol_version": PROTOCOL_VERSION, "nonce": _valid_nonce(nonce)})


def parse_request(payload: bytes) -> dict[str, Any]:
    value = _loads(payload)
    if value.get("magic") != REQUEST_MAGIC or value.get("protocol_version") != PROTOCOL_VERSION:
        raise DiscoveryError("unsupported discovery request")
    return {"nonce": _valid_nonce(value.get("nonce"))}


def build_response(
    nonce: str,
    *,
    coordinator_port: int,
    token: str,
    worker_processes: int,
    pc_name: str,
) -> bytes:
    if not isinstance(coordinator_port, int) or isinstance(coordinator_port, bool) or not 1 <= coordinator_port <= 65535:
        raise DiscoveryError("invalid coordinator port")
    if not isinstance(worker_processes, int) or isinstance(worker_processes, bool) or not 1 <= worker_processes <= 16:
        raise DiscoveryError("invalid worker process count")
    token = str(token or "")
    if len(token) < 24:
        raise DiscoveryError("invalid cluster token")
    pc_name = str(pc_name or "PC-Devorar").strip()[:200] or "PC-Devorar"
    return _dumps(
        {
            "magic": RESPONSE_MAGIC,
            "protocol_version": PROTOCOL_VERSION,
            "nonce": _valid_nonce(nonce),
            "coordinator_port": coordinator_port,
            "cluster_token": token,
            "worker_processes": worker_processes,
            "pc_name": pc_name,
        }
    )


def parse_response(payload: bytes, *, expected_nonce: str) -> dict[str, Any]:
    value = _loads(payload)
    if value.get("magic") != RESPONSE_MAGIC or value.get("protocol_version") != PROTOCOL_VERSION:
        raise DiscoveryError("unsupported discovery response")
    nonce = _valid_nonce(value.get("nonce"))
    if nonce != _valid_nonce(expected_nonce):
        raise DiscoveryError("discovery nonce mismatch")
    port = value.get("coordinator_port")
    processes = value.get("worker_processes")
    token = str(value.get("cluster_token") or "")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise DiscoveryError("invalid coordinator port")
    if not isinstance(processes, int) or isinstance(processes, bool) or not 1 <= processes <= 16:
        raise DiscoveryError("invalid worker process count")
    if len(token) < 24:
        raise DiscoveryError("invalid cluster token")
    return {
        "coordinator_port": port,
        "cluster_token": token,
        "worker_processes": processes,
        "pc_name": str(value.get("pc_name") or "PC-Devorar")[:200],
    }
