#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from typing import Any, Mapping, Sequence

import distributed_worker as base
from src.system_info import collect_system_info, describe_system


class TelemetryCoordinatorClient(base.CoordinatorClient):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._worker_id: str | None = None
        self._slot = 0
        self._last_telemetry = 0.0

    def request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        if path == "/v1/claim" and self._worker_id and time.monotonic() - self._last_telemetry >= 15.0:
            meta = _worker_meta(self._slot)
            super().request(
                "POST",
                "/v1/heartbeat",
                {"worker_id": self._worker_id, "meta": meta},
            )
            self._last_telemetry = time.monotonic()
        response = super().request(method, path, payload)
        if path == "/v1/register":
            self._worker_id = str(response.get("worker_id", "")) or None
            if isinstance(payload, Mapping) and isinstance(payload.get("meta"), Mapping):
                try:
                    self._slot = int(payload["meta"].get("slot", 0))
                except (TypeError, ValueError):
                    self._slot = 0
            self._last_telemetry = 0.0
        return response


def _worker_meta(slot: int) -> dict[str, Any]:
    info = collect_system_info(device_id=os.getenv("DEVORAR_DEVICE_ID", "").strip() or None)
    info.update(
        {
            "slot": slot,
            "pid": os.getpid(),
            "worker_processes": int(os.getenv("DEVORAR_PROCESSES", "1") or "1"),
        }
    )
    return info


def main(argv: Sequence[str] | None = None) -> int:
    print("Hardware local:", describe_system(collect_system_info()), flush=True)
    base.CoordinatorClient = TelemetryCoordinatorClient
    base._worker_meta = _worker_meta
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
