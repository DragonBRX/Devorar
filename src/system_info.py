from __future__ import annotations

import ctypes
import json
import os
import platform
import socket
import subprocess
from typing import Any


def _read_linux_meminfo() -> tuple[int | None, int | None]:
    try:
        values: dict[str, int] = {}
        with open('/proc/meminfo', 'r', encoding='utf-8') as handle:
            for line in handle:
                if ':' not in line:
                    continue
                key, rest = line.split(':', 1)
                parts = rest.strip().split()
                if not parts:
                    continue
                try:
                    value = int(parts[0])
                except ValueError:
                    continue
                multiplier = 1024 if len(parts) > 1 and parts[1].lower() == 'kb' else 1
                values[key] = value * multiplier
        total = values.get('MemTotal')
        available = values.get('MemAvailable')
        if available is None:
            available = values.get('MemFree', 0) + values.get('Buffers', 0) + values.get('Cached', 0)
        return total, available
    except OSError:
        return None, None


def _read_windows_memory() -> tuple[int | None, int | None]:
    if os.name != 'nt':
        return None, None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ('dwLength', ctypes.c_ulong),
            ('dwMemoryLoad', ctypes.c_ulong),
            ('ullTotalPhys', ctypes.c_ulonglong),
            ('ullAvailPhys', ctypes.c_ulonglong),
            ('ullTotalPageFile', ctypes.c_ulonglong),
            ('ullAvailPageFile', ctypes.c_ulonglong),
            ('ullTotalVirtual', ctypes.c_ulonglong),
            ('ullAvailVirtual', ctypes.c_ulonglong),
            ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except (AttributeError, OSError):
        return None, None
    if not ok:
        return None, None
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _read_sysconf_memory() -> tuple[int | None, int | None]:
    try:
        page_size = int(os.sysconf('SC_PAGE_SIZE'))
        total_pages = int(os.sysconf('SC_PHYS_PAGES'))
        available_pages = int(os.sysconf('SC_AVPHYS_PAGES'))
        return page_size * total_pages, page_size * available_pages
    except (AttributeError, OSError, ValueError):
        return None, None


def memory_info() -> dict[str, int | float | None]:
    total: int | None = None
    available: int | None = None
    if os.name == 'nt':
        total, available = _read_windows_memory()
    elif os.path.exists('/proc/meminfo'):
        total, available = _read_linux_meminfo()
    if total is None or available is None:
        total, available = _read_sysconf_memory()
    if total is None or available is None or total <= 0:
        return {
            'ram_total_bytes': None,
            'ram_available_bytes': None,
            'ram_used_bytes': None,
            'ram_usage_percent': None,
        }
    available = max(0, min(int(available), int(total)))
    used = int(total) - available
    return {
        'ram_total_bytes': int(total),
        'ram_available_bytes': available,
        'ram_used_bytes': used,
        'ram_usage_percent': round((used / int(total)) * 100.0, 1),
    }


def _getprop(name: str) -> str:
    try:
        result = subprocess.run(
            ['getprop', name],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ''
    return result.stdout.strip()


def _cpu_model() -> str:
    value = platform.processor().strip()
    if value:
        return value
    env_value = os.getenv('PROCESSOR_IDENTIFIER', '').strip()
    if env_value:
        return env_value
    try:
        with open('/proc/cpuinfo', 'r', encoding='utf-8', errors='replace') as handle:
            fallback = ''
            for line in handle:
                if ':' not in line:
                    continue
                key, raw = line.split(':', 1)
                candidate = raw.strip()
                normalized = key.strip().lower()
                if normalized in {'model name', 'hardware'} and candidate:
                    return candidate
                if normalized == 'processor' and candidate and not candidate.isdigit() and not fallback:
                    fallback = candidate
            if fallback:
                return fallback
    except OSError:
        pass
    return platform.machine() or 'unknown'


def collect_system_info(*, device_id: str | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        'device_id': (device_id or os.getenv('DEVORAR_DEVICE_ID', '')).strip() or None,
        'hostname': socket.gethostname(),
        'system': platform.system(),
        'release': platform.release(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'python': platform.python_version(),
        'cpu_model': _cpu_model(),
        'cpu_logical_cores': os.cpu_count(),
    }
    info.update(memory_info())
    manufacturer = _getprop('ro.product.manufacturer')
    model = _getprop('ro.product.model')
    android = _getprop('ro.build.version.release')
    sdk = _getprop('ro.build.version.sdk')
    if manufacturer or model or android:
        info['android_manufacturer'] = manufacturer or None
        info['android_model'] = model or None
        info['android_version'] = android or None
        info['android_sdk'] = sdk or None
        info['device_label'] = ' '.join(part for part in (manufacturer, model) if part).strip() or info['hostname']
    else:
        info['device_label'] = info['hostname']
    return info


def format_bytes(value: int | float | None) -> str:
    if value is None:
        return '?'
    amount = float(value)
    units = ('B', 'KiB', 'MiB', 'GiB', 'TiB')
    unit = units[0]
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    if unit == 'B':
        return f'{amount:.0f} {unit}'
    return f'{amount:.2f} {unit}'


def describe_system(info: dict[str, Any] | None = None) -> str:
    data = collect_system_info() if info is None else info
    total = format_bytes(data.get('ram_total_bytes'))
    available = format_bytes(data.get('ram_available_bytes'))
    cores = data.get('cpu_logical_cores') or '?'
    cpu = data.get('cpu_model') or data.get('machine') or '?'
    label = data.get('device_label') or data.get('hostname') or 'device'
    return f'{label} | CPU: {cpu} | núcleos lógicos: {cores} | RAM: {total} total / {available} disponível'


def main() -> int:
    info = collect_system_info()
    print(describe_system(info))
    print(json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
