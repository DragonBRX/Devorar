"""Bounded execution helpers for real rows from a remote Safetensors tensor.

This module is deliberately smaller than a model runtime.  It can fetch complete
BF16 rows with HTTP Range, decode them, and execute one linear operator.  It does
not construct a DeepSeek model or turn a synthetic activation into model logits.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .frontier_stream import (
    FrontierScanError,
    RepositoryTransport,
    TensorLocation,
)


MAX_EXECUTED_ROWS = 64
MAX_ROW_WIDTH = 65_536
MAX_EXECUTED_WEIGHT_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class LoadedTensorRows:
    """Decoded rows plus receipts tying them to exact remote byte intervals."""

    tensor: str
    shard: str
    dtype: str
    source_shape: tuple[int, int]
    row_ids: tuple[int, ...]
    values: tuple[tuple[float, ...], ...]
    payload_bytes: int
    payload_sha256: str
    range_receipts: tuple[Mapping[str, Any], ...]


def _bounded_integer(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise FrontierScanError(f"{name} must be an integer within [{minimum}, {maximum}]")
    return value


def select_evenly_spaced_rows(total_rows: int, count: int) -> tuple[int, ...]:
    """Select deterministic rows spanning a matrix without consulting its values."""

    total_rows = _bounded_integer(
        total_rows,
        name="total_rows",
        minimum=1,
        maximum=10_000_000,
    )
    count = _bounded_integer(
        count,
        name="count",
        minimum=1,
        maximum=MAX_EXECUTED_ROWS,
    )
    if count > total_rows:
        raise FrontierScanError("row count exceeds the source tensor height")
    if count == 1:
        return (0,)
    result = tuple((index * (total_rows - 1)) // (count - 1) for index in range(count))
    if len(set(result)) != count:
        raise FrontierScanError("row selection produced duplicate source rows")
    return result


def decode_bfloat16_le(payload: bytes) -> tuple[float, ...]:
    """Decode little-endian BF16 storage into exactly representable float32 values."""

    if not isinstance(payload, bytes) or not payload or len(payload) % 2:
        raise FrontierScanError("BF16 payload must contain a positive even number of bytes")
    values = tuple(
        struct.unpack("<f", struct.pack("<I", bits << 16))[0]
        for (bits,) in struct.iter_unpack("<H", payload)
    )
    if not all(math.isfinite(value) for value in values):
        raise FrontierScanError("executed BF16 rows contain NaN or infinity")
    return values


def load_complete_bf16_rows(
    transport: RepositoryTransport,
    location: TensorLocation,
    row_ids: Sequence[int],
) -> LoadedTensorRows:
    """Fetch complete matrix rows while refusing an unbounded or quantized request."""

    if not isinstance(location, TensorLocation):
        raise FrontierScanError("location must be a TensorLocation")
    descriptor = location.descriptor
    if descriptor.dtype != "BF16":
        raise FrontierScanError("bounded execution currently accepts only BF16 tensors")
    if len(descriptor.shape) != 2:
        raise FrontierScanError("bounded execution requires a two-dimensional tensor")
    source_rows, width = descriptor.shape
    _bounded_integer(source_rows, name="tensor row count", minimum=1, maximum=10_000_000)
    _bounded_integer(width, name="tensor row width", minimum=1, maximum=MAX_ROW_WIDTH)
    expected_stored_bytes = source_rows * width * 2
    if descriptor.stored_bytes != expected_stored_bytes:
        raise FrontierScanError("BF16 tensor byte length does not match its declared shape")
    if not isinstance(row_ids, Sequence) or isinstance(row_ids, (str, bytes, bytearray)):
        raise FrontierScanError("row_ids must be a bounded sequence of integers")
    if not 1 <= len(row_ids) <= MAX_EXECUTED_ROWS:
        raise FrontierScanError(f"select between 1 and {MAX_EXECUTED_ROWS} rows")

    normalized: list[int] = []
    seen: set[int] = set()
    for value in row_ids:
        row_id = _bounded_integer(value, name="row ID", minimum=0, maximum=source_rows - 1)
        if row_id in seen:
            raise FrontierScanError(f"duplicate row ID: {row_id}")
        seen.add(row_id)
        normalized.append(row_id)

    row_bytes = width * 2
    requested_bytes = row_bytes * len(normalized)
    if requested_bytes > MAX_EXECUTED_WEIGHT_BYTES:
        raise FrontierScanError("requested execution slice exceeds the weight-byte safety limit")

    digest = hashlib.sha256()
    values: list[tuple[float, ...]] = []
    receipts: list[dict[str, Any]] = []
    for row_id in normalized:
        relative_start = row_id * row_bytes
        absolute_start = location.absolute_data_start + relative_start
        absolute_end = absolute_start + row_bytes - 1
        chunk = transport.get_range(descriptor.shard, absolute_start, absolute_end)
        if chunk.total_file_bytes != location.shard_file_bytes:
            raise FrontierScanError(f"remote shard size changed while reading row {row_id}")
        if len(chunk.data) != row_bytes:
            raise FrontierScanError(f"remote row {row_id} has the wrong byte length")
        raw_sha256 = hashlib.sha256(chunk.data).hexdigest()
        if chunk.sha256 is not None and chunk.sha256 != raw_sha256:
            raise FrontierScanError(f"range provenance hash mismatch for row {row_id}")
        digest.update(chunk.data)
        values.append(decode_bfloat16_le(chunk.data))
        receipts.append(
            {
                "row_id": row_id,
                "absolute_http_start": absolute_start,
                "absolute_http_end_inclusive": absolute_end,
                "bytes": row_bytes,
                "sha256": raw_sha256,
                "etag": chunk.etag,
                "final_url": chunk.final_url,
            }
        )

    return LoadedTensorRows(
        tensor=descriptor.name,
        shard=descriptor.shard,
        dtype=descriptor.dtype,
        source_shape=(source_rows, width),
        row_ids=tuple(normalized),
        values=tuple(values),
        payload_bytes=requested_bytes,
        payload_sha256=digest.hexdigest(),
        range_receipts=tuple(receipts),
    )


def deterministic_float32_hidden(width: int) -> tuple[float, ...]:
    """Create a reproducible, non-model activation and quantize it to float32."""

    width = _bounded_integer(width, name="hidden width", minimum=1, maximum=MAX_ROW_WIDTH)
    raw = [
        math.sin((index + 1) * 0.017) + 0.5 * math.cos((index + 1) * 0.031)
        for index in range(width)
    ]
    rms = math.sqrt(math.fsum(value * value for value in raw) / width)
    if not math.isfinite(rms) or rms == 0.0:
        raise FrontierScanError("deterministic hidden vector has an invalid RMS")
    payload = struct.pack(f"<{width}f", *(value / rms for value in raw))
    result = tuple(value[0] for value in struct.iter_unpack("<f", payload))
    if not all(math.isfinite(value) for value in result):
        raise FrontierScanError("deterministic hidden vector contains non-finite values")
    return result


def float32_sha256(values: Sequence[float]) -> str:
    if not values:
        raise FrontierScanError("cannot hash an empty float32 vector")
    try:
        payload = struct.pack(f"<{len(values)}f", *values)
    except (OverflowError, struct.error) as error:
        raise FrontierScanError("invalid float32 vector") from error
    return hashlib.sha256(payload).hexdigest()


def reference_linear_logits(
    weights: Sequence[Sequence[float]],
    hidden: Sequence[float],
) -> tuple[float, ...]:
    """Reference y = W x using Python products and correctly rounded ``fsum``."""

    if not hidden or not weights:
        raise FrontierScanError("linear reference requires non-empty weights and hidden input")
    width = len(hidden)
    if width > MAX_ROW_WIDTH:
        raise FrontierScanError("linear reference hidden width exceeds the safety limit")
    if len(weights) > MAX_EXECUTED_ROWS:
        raise FrontierScanError("linear reference row count exceeds the safety limit")
    if not all(len(row) == width for row in weights):
        raise FrontierScanError("linear reference received incoherent matrix dimensions")
    if not all(math.isfinite(float(value)) for value in hidden):
        raise FrontierScanError("linear reference hidden input contains non-finite values")

    result = tuple(
        math.fsum(float(weight) * float(activation) for weight, activation in zip(row, hidden))
        for row in weights
    )
    if not all(math.isfinite(value) for value in result):
        raise FrontierScanError("linear reference produced non-finite logits")
    return result


def compare_logits(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Apply the standard combined absolute/relative parity inequality."""

    if not reference or len(reference) != len(candidate):
        raise FrontierScanError("reference and candidate logits must have equal non-zero length")
    for name, value in (("atol", atol), ("rtol", rtol)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise FrontierScanError(f"{name} must be a finite non-negative number")
        if not math.isfinite(float(value)) or value < 0:
            raise FrontierScanError(f"{name} must be a finite non-negative number")

    rows: list[dict[str, Any]] = []
    for expected, observed in zip(reference, candidate):
        expected = float(expected)
        observed = float(observed)
        if not math.isfinite(expected) or not math.isfinite(observed):
            raise FrontierScanError("cannot compare non-finite logits")
        absolute_error = abs(observed - expected)
        allowed_error = float(atol) + float(rtol) * abs(expected)
        rows.append(
            {
                "reference_logit": expected,
                "candidate_logit": observed,
                "absolute_error": absolute_error,
                "relative_error": absolute_error / max(abs(expected), 1e-30),
                "allowed_error": allowed_error,
                "passed": absolute_error <= allowed_error,
            }
        )
    return {
        "passed": all(row["passed"] for row in rows),
        "atol": float(atol),
        "rtol": float(rtol),
        "max_absolute_error": max(row["absolute_error"] for row in rows),
        "max_relative_error": max(row["relative_error"] for row in rows),
        "rows": rows,
    }


__all__ = [
    "LoadedTensorRows",
    "MAX_EXECUTED_ROWS",
    "MAX_EXECUTED_WEIGHT_BYTES",
    "MAX_ROW_WIDTH",
    "compare_logits",
    "decode_bfloat16_le",
    "deterministic_float32_hidden",
    "float32_sha256",
    "load_complete_bf16_rows",
    "reference_linear_logits",
    "select_evenly_spaced_rows",
]
