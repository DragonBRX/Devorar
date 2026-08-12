from __future__ import annotations

from src.system_info import collect_system_info, format_bytes


def test_system_info_reports_cpu_and_memory_without_external_dependency() -> None:
    info = collect_system_info(device_id="test-device")
    assert info["device_id"] == "test-device"
    assert "cpu_logical_cores" in info
    assert "ram_total_bytes" in info
    total = info["ram_total_bytes"]
    if total is not None:
        assert total > 0
        assert format_bytes(total).endswith(("MiB", "GiB", "TiB", "KiB", "B"))
