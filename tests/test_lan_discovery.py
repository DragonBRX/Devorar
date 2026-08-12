from __future__ import annotations

import pytest

from src.lan_discovery import (
    DiscoveryError,
    build_request,
    build_response,
    is_local_address,
    parse_request,
    parse_response,
)


def test_discovery_request_round_trip() -> None:
    nonce = "0123456789abcdef0123456789abcdef"
    assert parse_request(build_request(nonce)) == {"nonce": nonce}


def test_discovery_response_round_trip() -> None:
    nonce = "abcdef0123456789abcdef0123456789"
    payload = build_response(
        nonce,
        coordinator_port=8765,
        token="a" * 32,
        worker_processes=2,
        pc_name="NOTEBOOK-DEVORAR",
    )
    result = parse_response(payload, expected_nonce=nonce)
    assert result["coordinator_port"] == 8765
    assert result["cluster_token"] == "a" * 32
    assert result["worker_processes"] == 2
    assert result["pc_name"] == "NOTEBOOK-DEVORAR"


def test_discovery_rejects_wrong_nonce() -> None:
    payload = build_response(
        "0" * 32,
        coordinator_port=8765,
        token="b" * 32,
        worker_processes=1,
        pc_name="PC",
    )
    with pytest.raises(DiscoveryError, match="nonce mismatch"):
        parse_response(payload, expected_nonce="1" * 32)


def test_local_address_filter() -> None:
    assert is_local_address("192.168.1.20")
    assert is_local_address("10.0.0.5")
    assert is_local_address("127.0.0.1")
    assert not is_local_address("8.8.8.8")
