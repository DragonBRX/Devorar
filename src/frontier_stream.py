"""Out-of-core metadata inspection for very large safetensors repositories.

This module is the safety/feasibility gate for the Frontier branch of Devorar.
It reads the Hugging Face index plus safetensors headers with HTTP byte ranges;
it never downloads a complete shard, instantiates a donor model, performs a
donor forward pass, or claims to recover hidden reasoning.

The scanner intentionally stops at a plan.  A tensor sketch or a weight-size
estimate is not a language model.  A later compression backend must provide a
real executable checkpoint and pass behavioural evaluation before promotion.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import re
import statistics
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Protocol, Sequence


FRONTIER_MANIFEST_FORMAT = "lira.experimental.frontier-weight-map"
FRONTIER_MANIFEST_VERSION = 1
USER_AGENT = "DragonBRX-Devorar-Frontier/0.4.0"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
MAX_SHARDS = 1_024
MAX_TENSORS = 1_000_000
MAX_DIMENSIONS = 16
MAX_RETRIES = 3

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_ROUTED_EXPERT = re.compile(r"(?:^|\.)ffn\.experts\.(\d+)\.")
_LAYER_ROUTED_EXPERT = re.compile(r"(?:^|\.)layers\.(\d+)\.ffn\.experts\.(\d+)\.")
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")

# Keep this table in lock-step with safetensors' public Dtype enum.  Byte-size
# checks use bits (rather than rounded bytes) because F4/F6 are packed formats.
_DTYPE_BITS = {
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "C64": 64,
    "F64": 64,
    "I64": 64,
    "U64": 64,
}


class FrontierScanError(RuntimeError):
    """Raised when a remote repository violates the bounded scan contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise FrontierScanError(f"duplicate JSON key rejected: {key!r}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> None:
    raise FrontierScanError(f"non-finite JSON number rejected: {value}")


def _strict_json_loads(payload: bytes, source: str) -> Any:
    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except FrontierScanError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FrontierScanError(f"invalid UTF-8 JSON in {source}: {error}") from error


@dataclass(frozen=True)
class TransferStats:
    requests: int
    bytes_received: int


@dataclass(frozen=True)
class RangeChunk:
    data: bytes
    total_file_bytes: int
    etag: str | None = None
    final_url: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class TransferReceipt:
    """Non-secret provenance for bytes accepted from a remote repository."""

    path: str
    kind: str
    start: int | None
    end_inclusive: int | None
    total_file_bytes: int | None
    bytes_received: int
    sha256: str
    etag: str | None
    final_url: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class _RangeBinding:
    etag: str
    total_file_bytes: int


class RepositoryTransport(Protocol):
    """Minimal transport contract, injectable for deterministic tests."""

    def get_json(self, path: str, *, max_bytes: int = MAX_JSON_BYTES) -> Any: ...

    def get_range(self, path: str, start: int, end: int) -> RangeChunk: ...

    def stats(self) -> TransferStats: ...


def _is_trusted_hf_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    host = hostname.rstrip(".").lower()
    return (
        host == "huggingface.co"
        or host.endswith(".huggingface.co")
        or host == "hf.co"
        or host.endswith(".hf.co")
    )


def _safe_remote_identity(url: str) -> str:
    """Validate an HF HTTPS endpoint and remove signed query credentials."""

    parsed = urllib.parse.urlsplit(url)
    try:
        port = parsed.port
    except ValueError as error:
        raise FrontierScanError("remote endpoint contains an invalid port") from error
    if (
        parsed.scheme.lower() != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or not _is_trusted_hf_host(parsed.hostname)
    ):
        raise FrontierScanError("remote endpoint left the trusted Hugging Face HTTPS boundary")
    host = parsed.hostname.rstrip(".").lower()  # type: ignore[union-attr]
    return urllib.parse.urlunsplit(("https", host, parsed.path, "", ""))


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urllib.parse.urlsplit(url)
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    return parsed.scheme.lower(), (parsed.hostname or "").rstrip(".").lower(), port


class _SafeHuggingFaceRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Allow only trusted HTTPS redirects and never forward bearer auth cross-origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        absolute = urllib.parse.urljoin(req.full_url, newurl)
        _safe_remote_identity(absolute)
        redirected = super().redirect_request(req, fp, code, msg, headers, absolute)
        if redirected is not None and _origin(req.full_url) != _origin(absolute):
            redirected.remove_header("Authorization")
            redirected.remove_header("Proxy-Authorization")
        return redirected


def validate_repo_id(repo_id: str) -> str:
    if not isinstance(repo_id, str) or not _REPO_ID.fullmatch(repo_id):
        raise FrontierScanError("repo_id must have the form trusted-owner/trusted-model")
    return repo_id


def validate_revision(revision: str) -> str:
    if not isinstance(revision, str) or not _SHA40.fullmatch(revision):
        raise FrontierScanError("revision must be an immutable 40-character lowercase SHA")
    return revision


def validate_repository_path(path: str, *, suffix: str | None = None) -> str:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise FrontierScanError(f"unsafe repository path: {path!r}")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise FrontierScanError(f"unsafe repository path: {path!r}")
    if suffix is not None and not path.endswith(suffix):
        raise FrontierScanError(f"repository path must end in {suffix!r}: {path!r}")
    return path


class HuggingFaceRangeTransport:
    """Strict bounded reader for immutable files on huggingface.co.

    Redirects to the Hub's content-addressed storage are handled by urllib.
    Range responses must remain HTTP 206; a server that ignores Range is
    rejected before its response body can be read in full.
    """

    def __init__(
        self,
        repo_id: str,
        revision: str,
        *,
        token: str | None = None,
        timeout_seconds: float = 60.0,
        retries: int = MAX_RETRIES,
    ) -> None:
        self.repo_id = validate_repo_id(repo_id)
        self.revision = validate_revision(revision)
        if not math.isfinite(timeout_seconds) or not 1.0 <= timeout_seconds <= 300.0:
            raise FrontierScanError("timeout_seconds must be within [1, 300]")
        if not isinstance(retries, int) or isinstance(retries, bool) or not 1 <= retries <= 5:
            raise FrontierScanError("retries must be an integer within [1, 5]")
        self.timeout_seconds = float(timeout_seconds)
        self.retries = retries
        self._token = token.strip() if isinstance(token, str) and token.strip() else None
        self._opener = urllib.request.build_opener(_SafeHuggingFaceRedirectHandler())
        self._requests = 0
        self._bytes = 0
        self._stats_lock = threading.Lock()
        self._bindings: dict[str, _RangeBinding] = {}
        self._bindings_lock = threading.Lock()
        self._receipts: list[TransferReceipt] = []

    def _url(self, path: str) -> str:
        safe_path = validate_repository_path(path)
        owner, model = self.repo_id.split("/", 1)
        encoded_path = "/".join(urllib.parse.quote(part, safe="") for part in safe_path.split("/"))
        return (
            f"https://huggingface.co/{urllib.parse.quote(owner, safe='')}/"
            f"{urllib.parse.quote(model, safe='')}/resolve/{self.revision}/{encoded_path}"
        )

    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"
        if extra:
            headers.update(extra)
        return headers

    def _open(self, request: urllib.request.Request):
        last_error: BaseException | None = None
        for attempt in range(self.retries):
            try:
                return self._opener.open(request, timeout=self.timeout_seconds)
            except urllib.error.HTTPError as error:
                last_error = error
                retryable = error.code in {408, 425, 429, 500, 502, 503, 504}
                error.close()
                if not retryable:
                    break
            except urllib.error.URLError as error:
                last_error = error
            if attempt + 1 < self.retries:
                time.sleep(0.25 * (2**attempt))
        if isinstance(last_error, urllib.error.HTTPError):
            detail = f"HTTP {last_error.code}"
        else:
            detail = type(last_error).__name__ if last_error is not None else "unknown error"
        raise FrontierScanError(f"bounded remote request failed: {detail}") from last_error

    @staticmethod
    def _response_identity(response: Any, fallback_url: str) -> str:
        geturl = getattr(response, "geturl", None)
        final_url = geturl() if callable(geturl) else fallback_url
        return _safe_remote_identity(final_url)

    @staticmethod
    def _response_etag(response: Any) -> str | None:
        raw = response.headers.get("ETag")
        if raw is None:
            return None
        if not isinstance(raw, str):
            raise FrontierScanError("range response has a non-text ETag")
        value = raw.strip()
        if (
            len(value) < 2
            or value.startswith("W/")
            or not (value.startswith('"') and value.endswith('"'))
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
        ):
            raise FrontierScanError("range response lacks a valid strong ETag")
        return value

    @staticmethod
    def _require_identity_encoding(response: Any, path: str) -> None:
        encoding = response.headers.get("Content-Encoding")
        if encoding is not None and encoding.strip().lower() not in {"", "identity"}:
            raise FrontierScanError(f"compressed transfer encoding is forbidden for {path}")

    def _record_receipt(self, receipt: TransferReceipt) -> None:
        with self._stats_lock:
            self._requests += 1
            self._bytes += receipt.bytes_received
            self._receipts.append(receipt)

    def get_json(self, path: str, *, max_bytes: int = MAX_JSON_BYTES) -> Any:
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise FrontierScanError("max_bytes must be a positive integer")
        request = urllib.request.Request(self._url(path), headers=self._headers())
        response = self._open(request)
        try:
            self._require_identity_encoding(response, path)
            final_url = self._response_identity(response, request.full_url)
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except (TypeError, ValueError) as error:
                    raise FrontierScanError(f"invalid Content-Length for {path}") from error
                if declared_length < 0 or declared_length > max_bytes:
                    raise FrontierScanError(f"remote JSON exceeds {max_bytes} bytes: {path}")
            payload = response.read(max_bytes + 1)
            etag = response.headers.get("ETag")
            etag = etag.strip() if isinstance(etag, str) and etag.strip() else None
        finally:
            response.close()
        if len(payload) > max_bytes:
            raise FrontierScanError(f"remote JSON exceeds {max_bytes} bytes: {path}")
        self._record_receipt(
            TransferReceipt(
                path=path,
                kind="json",
                start=None,
                end_inclusive=None,
                total_file_bytes=len(payload),
                bytes_received=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                etag=etag,
                final_url=final_url,
            )
        )
        return _strict_json_loads(payload, path)

    def get_range(self, path: str, start: int, end: int) -> RangeChunk:
        validate_repository_path(path, suffix=".safetensors")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in (start, end)):
            raise FrontierScanError("range offsets must be integers")
        if start < 0 or end < start:
            raise FrontierScanError(f"invalid byte range: {start}-{end}")
        length = end - start + 1
        if length > MAX_SAFETENSORS_HEADER_BYTES:
            raise FrontierScanError("single metadata range exceeds the header safety limit")
        with self._bindings_lock:
            binding = self._bindings.get(path)
        extra_headers = {"Range": f"bytes={start}-{end}"}
        if binding is not None:
            extra_headers["If-Range"] = binding.etag
        request = urllib.request.Request(self._url(path), headers=self._headers(extra_headers))
        response = self._open(request)
        try:
            if response.status != 206:
                raise FrontierScanError(
                    f"server ignored byte range for {path}; refusing full-shard response"
                )
            self._require_identity_encoding(response, path)
            final_url = self._response_identity(response, request.full_url)
            etag = self._response_etag(response)
            if etag is None:
                raise FrontierScanError(f"range response lacks a strong ETag for {path}")
            match = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
            if match is None:
                raise FrontierScanError(f"missing or invalid Content-Range for {path}")
            actual_start, actual_end, total = (int(value) for value in match.groups())
            if (actual_start, actual_end) != (start, end) or total <= end:
                raise FrontierScanError(f"server returned the wrong byte range for {path}")
            if binding is not None and etag != binding.etag:
                raise FrontierScanError(f"remote shard ETag changed during scan: {path}")
            if binding is not None and total != binding.total_file_bytes:
                raise FrontierScanError(f"remote shard size changed during scan: {path}")
            payload = response.read(length + 1)
        finally:
            response.close()
        if len(payload) != length:
            raise FrontierScanError(
                f"range length mismatch for {path}: expected {length}, received {len(payload)}"
            )
        # The Hub may route consecutive ranges for the same immutable object through
        # different trusted CDN/Xet endpoints.  Endpoint is provenance, not object
        # identity; bind the representation to its strong ETag and total byte size.
        candidate = _RangeBinding(etag=etag, total_file_bytes=total)
        with self._bindings_lock:
            previous = self._bindings.setdefault(path, candidate)
            if previous != candidate:
                if previous.etag != candidate.etag:
                    raise FrontierScanError(f"remote shard ETag changed during scan: {path}")
                raise FrontierScanError(f"remote shard size changed during scan: {path}")
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        self._record_receipt(
            TransferReceipt(
                path=path,
                kind="range",
                start=start,
                end_inclusive=end,
                total_file_bytes=total,
                bytes_received=len(payload),
                sha256=payload_sha256,
                etag=etag,
                final_url=final_url,
            )
        )
        return RangeChunk(
            data=payload,
            total_file_bytes=total,
            etag=etag,
            final_url=final_url,
            sha256=payload_sha256,
        )

    def stats(self) -> TransferStats:
        with self._stats_lock:
            return TransferStats(self._requests, self._bytes)

    def receipts(self) -> tuple[TransferReceipt, ...]:
        with self._stats_lock:
            return tuple(self._receipts)


@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int

    @property
    def stored_bytes(self) -> int:
        return self.data_end - self.data_start

    @property
    def stored_elements(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class RepositoryInspection:
    repo_id: str
    revision: str
    license: str
    config: Mapping[str, Any]
    source_payload_bytes: int
    source_file_bytes: int
    tensor_count: int
    shard_count: int
    category_bytes: Mapping[str, int]
    category_tensors: Mapping[str, int]
    dtype_bytes: Mapping[str, int]
    routed_expert_bytes: Mapping[int, int]
    layer_bytes: Mapping[int, int]
    transfer_stats: TransferStats
    index_sha256: str
    config_sha256: str
    tensor_inventory_sha256: str
    index_raw_sha256: str | None = None
    config_raw_sha256: str | None = None
    transfer_receipts: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": {
                "repo_id": self.repo_id,
                "revision": self.revision,
                "license": self.license,
                "model_type": self.config.get("model_type"),
                "architecture": list(self.config.get("architectures", [])),
                "config_sha256": self.config_sha256,
                "index_sha256": self.index_sha256,
                "config_raw_sha256": self.config_raw_sha256,
                "index_raw_sha256": self.index_raw_sha256,
                "json_hash_scope": {
                    "config_sha256": "canonical_parsed_json",
                    "index_sha256": "canonical_parsed_json",
                    "config_raw_sha256": "exact_received_bytes",
                    "index_raw_sha256": "exact_received_bytes",
                },
                "tensor_inventory_sha256": self.tensor_inventory_sha256,
            },
            "source_geometry": {
                "hidden_size": self.config.get("hidden_size"),
                "num_hidden_layers": self.config.get("num_hidden_layers"),
                "n_routed_experts": self.config.get("n_routed_experts"),
                "n_shared_experts": self.config.get("n_shared_experts"),
                "num_experts_per_tok": self.config.get("num_experts_per_tok"),
                "max_position_embeddings": self.config.get("max_position_embeddings"),
                "expert_dtype": self.config.get("expert_dtype"),
                "quantization_config": self.config.get("quantization_config"),
            },
            "inventory": {
                "shard_count": self.shard_count,
                "tensor_count": self.tensor_count,
                "source_payload_bytes": self.source_payload_bytes,
                "source_file_bytes": self.source_file_bytes,
                "categories": _counter_rows(self.category_bytes, self.category_tensors),
                "dtypes": [
                    {"dtype": key, "stored_bytes": value}
                    for key, value in sorted(self.dtype_bytes.items())
                ],
                "routed_expert_count_observed": len(self.routed_expert_bytes),
                "routed_expert_stored_bytes_min": min(self.routed_expert_bytes.values(), default=0),
                "routed_expert_stored_bytes_max": max(self.routed_expert_bytes.values(), default=0),
                "layer_count_observed": len(self.layer_bytes),
            },
            "scan": {
                "http_requests": self.transfer_stats.requests,
                "bytes_received": self.transfer_stats.bytes_received,
                "full_shards_downloaded": 0,
                "donor_model_instantiated": False,
                "donor_forward_calls": 0,
                "donor_logits_or_generations_used": False,
                "accepted_transfer_receipts": list(self.transfer_receipts),
            },
        }


@dataclass(frozen=True)
class TensorLocation:
    descriptor: TensorDescriptor
    data_origin: int
    shard_file_bytes: int

    @property
    def absolute_data_start(self) -> int:
        return self.data_origin + self.descriptor.data_start

    @property
    def absolute_data_end(self) -> int:
        return self.data_origin + self.descriptor.data_end


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _tensor_inventory_sha256(descriptors: Sequence[TensorDescriptor]) -> str:
    digest = hashlib.sha256()
    for item in sorted(descriptors, key=lambda value: value.name):
        digest.update(
            _canonical_json_bytes(
                {
                    "name": item.name,
                    "shard": item.shard,
                    "dtype": item.dtype,
                    "shape": list(item.shape),
                    "data_offsets": [item.data_start, item.data_end],
                }
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _counter_rows(byte_counts: Mapping[str, int], tensor_counts: Mapping[str, int]) -> list[dict[str, Any]]:
    total = sum(byte_counts.values())
    return [
        {
            "category": category,
            "tensor_count": tensor_counts.get(category, 0),
            "stored_bytes": stored,
            "stored_fraction": (stored / total) if total else 0.0,
        }
        for category, stored in sorted(byte_counts.items())
    ]


def transport_receipts(transport: RepositoryTransport) -> tuple[Mapping[str, Any], ...]:
    getter = getattr(transport, "receipts", None)
    if not callable(getter):
        return ()
    raw = getter()
    if not isinstance(raw, (tuple, list)) or len(raw) > (2 * MAX_SHARDS + 32):
        raise FrontierScanError("transport provenance receipt count is outside safety bounds")
    result: list[Mapping[str, Any]] = []
    for item in raw:
        if isinstance(item, TransferReceipt):
            result.append(item.to_dict())
        elif isinstance(item, Mapping):
            result.append(dict(item))
        else:
            raise FrontierScanError("transport returned an invalid provenance receipt")
    return tuple(result)


def _raw_json_hash(receipts: Sequence[Mapping[str, Any]], path: str) -> str | None:
    matches = [
        item.get("sha256")
        for item in receipts
        if item.get("kind") == "json" and item.get("path") == path
    ]
    if not matches:
        return None
    if len(matches) != 1 or not isinstance(matches[0], str) or not re.fullmatch(r"[0-9a-f]{64}", matches[0]):
        raise FrontierScanError(f"ambiguous raw JSON provenance for {path}")
    return matches[0]


def _parse_index(raw: Any) -> tuple[dict[str, str], int]:
    if not isinstance(raw, Mapping):
        raise FrontierScanError("safetensors index must be a JSON object")
    weight_map = raw.get("weight_map")
    metadata = raw.get("metadata")
    if not isinstance(weight_map, Mapping) or not isinstance(metadata, Mapping):
        raise FrontierScanError("safetensors index lacks weight_map or metadata")
    if not 1 <= len(weight_map) <= MAX_TENSORS:
        raise FrontierScanError("safetensors index tensor count is outside safety bounds")
    declared_total = metadata.get("total_size")
    if not isinstance(declared_total, int) or isinstance(declared_total, bool) or declared_total <= 0:
        raise FrontierScanError("safetensors index metadata.total_size must be positive")
    clean: dict[str, str] = {}
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not name or len(name) > 1_024:
            raise FrontierScanError("invalid tensor name in safetensors index")
        if not isinstance(shard, str):
            raise FrontierScanError(f"invalid shard mapping for tensor {name!r}")
        clean[name] = validate_repository_path(shard, suffix=".safetensors")
    if len(set(clean.values())) > MAX_SHARDS:
        raise FrontierScanError("safetensors shard count exceeds the safety limit")
    return clean, declared_total


def _parse_tensor(name: str, shard: str, metadata: Any) -> TensorDescriptor:
    if not isinstance(metadata, Mapping):
        raise FrontierScanError(f"invalid metadata for tensor {name!r}")
    dtype = metadata.get("dtype")
    shape = metadata.get("shape")
    offsets = metadata.get("data_offsets")
    if not isinstance(dtype, str) or dtype not in _DTYPE_BITS:
        raise FrontierScanError(f"unknown or invalid dtype for tensor {name!r}: {dtype!r}")
    if not isinstance(shape, list) or not 0 <= len(shape) <= MAX_DIMENSIONS:
        raise FrontierScanError(f"invalid shape for tensor {name!r}")
    dimensions: list[int] = []
    for dimension in shape:
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 0
            or dimension > 2**40
        ):
            raise FrontierScanError(f"invalid dimension for tensor {name!r}")
        dimensions.append(dimension)
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise FrontierScanError(f"invalid offsets for tensor {name!r}")
    start, end = offsets
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in offsets):
        raise FrontierScanError(f"non-integer offsets for tensor {name!r}")
    if start < 0 or end < start:
        raise FrontierScanError(f"invalid offsets for tensor {name!r}")
    element_count = math.prod(dimensions)
    expected_bits = element_count * _DTYPE_BITS[dtype]
    if expected_bits % 8:
        raise FrontierScanError(f"sub-byte tensor is not byte-aligned: {name!r}")
    expected_bytes = expected_bits // 8
    if end - start != expected_bytes:
        raise FrontierScanError(
            f"shape/dtype byte size mismatch for tensor {name!r}: "
            f"expected {expected_bytes}, observed {end - start}"
        )
    return TensorDescriptor(name, shard, dtype, tuple(dimensions), start, end)


def _read_shard_header(
    transport: RepositoryTransport,
    shard: str,
    expected_names: frozenset[str],
) -> tuple[list[TensorDescriptor], int, int]:
    prefix = transport.get_range(shard, 0, 7)
    if len(prefix.data) != 8:
        raise FrontierScanError(f"safetensors prefix is truncated: {shard}")
    if prefix.sha256 is not None and prefix.sha256 != hashlib.sha256(prefix.data).hexdigest():
        raise FrontierScanError(f"range provenance hash mismatch: {shard}")
    header_length = struct.unpack("<Q", prefix.data)[0]
    if not 2 <= header_length <= MAX_SAFETENSORS_HEADER_BYTES:
        raise FrontierScanError(f"safetensors header length is unsafe: {shard}")
    header_chunk = transport.get_range(shard, 8, 7 + header_length)
    if header_chunk.total_file_bytes != prefix.total_file_bytes:
        raise FrontierScanError(f"remote shard size changed during scan: {shard}")
    if prefix.etag is not None and header_chunk.etag != prefix.etag:
        raise FrontierScanError(f"remote shard ETag changed during scan: {shard}")
    if (
        header_chunk.sha256 is not None
        and header_chunk.sha256 != hashlib.sha256(header_chunk.data).hexdigest()
    ):
        raise FrontierScanError(f"range provenance hash mismatch: {shard}")
    header = _strict_json_loads(header_chunk.data, shard)
    if not isinstance(header, Mapping):
        raise FrontierScanError(f"safetensors header must be an object: {shard}")
    observed_names = frozenset(name for name in header if name != "__metadata__")
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)[:3]
        extra = sorted(observed_names - expected_names)[:3]
        raise FrontierScanError(
            f"index/header mismatch in {shard}; missing={missing}, extra={extra}"
        )
    descriptors = [_parse_tensor(name, shard, header[name]) for name in sorted(observed_names)]
    data_origin = 8 + header_length
    data_capacity = prefix.total_file_bytes - data_origin
    previous_end = 0
    for descriptor in sorted(descriptors, key=lambda item: item.data_start):
        if descriptor.data_start != previous_end:
            relation = "overlapping" if descriptor.data_start < previous_end else "gapped"
            raise FrontierScanError(f"{relation} tensor offsets in {shard}")
        if descriptor.data_end > data_capacity:
            raise FrontierScanError(f"tensor offset exceeds shard size in {shard}")
        previous_end = descriptor.data_end
    if previous_end != data_capacity:
        raise FrontierScanError(f"unmapped trailing tensor data in {shard}")
    return descriptors, prefix.total_file_bytes, data_origin


def _category(name: str) -> tuple[str, int | None]:
    expert = _ROUTED_EXPERT.search(name)
    if expert is not None:
        return "routed_experts", int(expert.group(1))
    if ".shared_experts." in name:
        return "shared_experts", None
    if name.startswith(("embed.", "head.", "lm_head.", "model.embed_tokens.")):
        return "embedding_or_head", None
    if "router" in name or ".gate." in name or name.endswith(".gate.weight"):
        return "router_or_gate", None
    return "other_core", None


def locate_tensors(
    transport: RepositoryTransport,
    tensor_names: Sequence[str],
) -> tuple[dict[str, TensorLocation], str]:
    """Locate selected tensors by reading only the index and relevant headers."""

    if not 1 <= len(tensor_names) <= 16:
        raise FrontierScanError("tensor_names must contain between 1 and 16 entries")
    requested: list[str] = []
    seen: set[str] = set()
    for name in tensor_names:
        if not isinstance(name, str) or not name or len(name) > 1_024:
            raise FrontierScanError("invalid requested tensor name")
        if name in seen:
            raise FrontierScanError(f"duplicate requested tensor name: {name}")
        seen.add(name)
        requested.append(name)

    index_raw = transport.get_json("model.safetensors.index.json")
    weight_map, _declared_total = _parse_index(index_raw)
    missing = sorted(set(requested) - set(weight_map))
    if missing:
        raise FrontierScanError(f"requested tensor is absent from the index: {missing[0]}")
    relevant_shards = {weight_map[name] for name in requested}
    expected_by_shard: dict[str, set[str]] = defaultdict(set)
    for name, shard in weight_map.items():
        if shard in relevant_shards:
            expected_by_shard[shard].add(name)

    locations: dict[str, TensorLocation] = {}
    for shard in sorted(relevant_shards):
        descriptors, file_bytes, data_origin = _read_shard_header(
            transport,
            shard,
            frozenset(expected_by_shard[shard]),
        )
        for descriptor in descriptors:
            if descriptor.name in seen:
                locations[descriptor.name] = TensorLocation(descriptor, data_origin, file_bytes)
    if set(locations) != set(requested):
        raise FrontierScanError("not every requested tensor was located")
    return locations, _sha256_json(index_raw)


_STORAGE_WIDTH = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2FNUZ": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "C64": 8,
}


def _sample_windows(size: int, budget: int, alignment: int) -> list[tuple[int, int]]:
    if size <= 0:
        return []
    usable = min(size, budget)
    usable -= usable % alignment
    if usable <= 0:
        usable = min(size, alignment)
    if usable >= size:
        return [(0, size)]
    window = max(alignment, (usable // 3 // alignment) * alignment)
    window = min(window, size)
    starts = [0, max(0, (size - window) // 2), max(0, size - window)]
    starts = [start - (start % alignment) for start in starts]
    windows: list[tuple[int, int]] = []
    for start in starts:
        end = min(size, start + window)
        end -= (end - start) % alignment
        if end <= start:
            continue
        if windows and start < windows[-1][1]:
            continue
        windows.append((start, end))
    return windows


def _storage_values(dtype: str, payload: bytes) -> tuple[list[float], str]:
    width = _STORAGE_WIDTH.get(dtype, 1)
    usable = payload[: len(payload) - (len(payload) % width)]
    if not usable:
        return [], "no_aligned_values"
    if dtype == "I8":
        return [float(value if value < 128 else value - 256) for value in usable], "signed_int8_storage"
    if dtype in {"U8", "BOOL"}:
        return [float(value) for value in usable], "unsigned_byte_storage"
    formats = {
        "I16": "h",
        "U16": "H",
        "F16": "e",
        "I32": "i",
        "U32": "I",
        "F32": "f",
        "I64": "q",
        "U64": "Q",
        "F64": "d",
    }
    if dtype == "BF16":
        values = []
        for (bits,) in struct.iter_unpack("<H", usable):
            values.append(struct.unpack("<f", struct.pack("<I", bits << 16))[0])
        return values, "decoded_bfloat16_sample"
    if dtype in formats:
        return [float(value[0]) for value in struct.iter_unpack("<" + formats[dtype], usable)], (
            f"decoded_{dtype.lower()}_sample"
        )
    return [float(value) for value in usable], "raw_quantized_bytes_not_dequantized"


def _sample_statistics(dtype: str, payload: bytes, name: str) -> dict[str, Any]:
    counts = Counter(payload)
    total = len(payload)
    entropy = 0.0
    if total:
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log2(probability)
    values, interpretation = _storage_values(dtype, payload)
    finite_values = [value for value in values if math.isfinite(value)]
    result: dict[str, Any] = {
        "raw_byte_min": min(payload) if payload else None,
        "raw_byte_max": max(payload) if payload else None,
        "raw_zero_fraction": (counts.get(0, 0) / total) if total else None,
        "raw_byte_entropy_bits": entropy,
        "storage_interpretation": interpretation,
        "decoded_or_storage_value_count": len(values),
        "finite_value_fraction": (len(finite_values) / len(values)) if values else None,
    }
    if finite_values:
        result.update(
            {
                "finite_value_min": min(finite_values),
                "finite_value_max": max(finite_values),
                "finite_value_mean": statistics.fmean(finite_values),
                "finite_value_population_stddev": statistics.pstdev(finite_values),
            }
        )
    if dtype == "I8" and ".ffn.experts." in name:
        nibbles = Counter()
        for value in payload:
            nibbles[value & 0x0F] += 1
            nibbles[(value >> 4) & 0x0F] += 1
        result["packed_nibble_histogram"] = {str(key): nibbles[key] for key in range(16)}
        result["storage_interpretation"] = "possible_packed_fp4_container_not_dequantized"
    return result


def fingerprint_tensors(
    transport: RepositoryTransport,
    locations: Mapping[str, TensorLocation],
    *,
    sample_bytes_per_tensor: int = 4_096,
) -> list[dict[str, Any]]:
    """Fetch bounded tensor micro-samples and return hashes/statistics, never raw bytes."""

    if not isinstance(locations, Mapping) or not 1 <= len(locations) <= 16:
        raise FrontierScanError("locations must contain between 1 and 16 tensors")
    for name, location in locations.items():
        if not isinstance(name, str) or not isinstance(location, TensorLocation):
            raise FrontierScanError("locations must map tensor names to TensorLocation values")
        if name != location.descriptor.name:
            raise FrontierScanError("location key must match descriptor name")
    if (
        not isinstance(sample_bytes_per_tensor, int)
        or isinstance(sample_bytes_per_tensor, bool)
        or not 256 <= sample_bytes_per_tensor <= 65_536
    ):
        raise FrontierScanError("sample_bytes_per_tensor must be within [256, 65536]")
    results: list[dict[str, Any]] = []
    for name in sorted(locations):
        location = locations[name]
        descriptor = location.descriptor
        alignment = _STORAGE_WIDTH.get(descriptor.dtype, 1)
        windows = _sample_windows(descriptor.stored_bytes, sample_bytes_per_tensor, alignment)
        digest = hashlib.sha256()
        combined = bytearray()
        receipts: list[dict[str, int | str]] = []
        for relative_start, relative_end in windows:
            absolute_start = location.absolute_data_start + relative_start
            absolute_end = location.absolute_data_start + relative_end - 1
            chunk = transport.get_range(descriptor.shard, absolute_start, absolute_end)
            if chunk.total_file_bytes != location.shard_file_bytes:
                raise FrontierScanError(f"remote shard size changed while sampling {name}")
            raw_sha256 = hashlib.sha256(chunk.data).hexdigest()
            if chunk.sha256 is not None and chunk.sha256 != raw_sha256:
                raise FrontierScanError(f"range provenance hash mismatch while sampling {name}")
            digest.update(struct.pack("<QQ", relative_start, relative_end))
            digest.update(chunk.data)
            combined.extend(chunk.data)
            receipts.append(
                {
                    "relative_start": relative_start,
                    "relative_end_exclusive": relative_end,
                    "absolute_http_start": absolute_start,
                    "absolute_http_end_inclusive": absolute_end,
                    "bytes": len(chunk.data),
                    "sha256": raw_sha256,
                    "etag": chunk.etag,
                    "final_url": chunk.final_url,
                }
            )
        category, expert_id = _category(name)
        sample = bytes(combined)
        results.append(
            {
                "tensor": name,
                "structural_category": category,
                "expert_id": expert_id,
                "shard": descriptor.shard,
                "dtype": descriptor.dtype,
                "shape": list(descriptor.shape),
                "tensor_stored_bytes": descriptor.stored_bytes,
                "sampled_bytes": len(sample),
                "sampled_fraction": len(sample) / descriptor.stored_bytes if descriptor.stored_bytes else 0.0,
                "sample_fingerprint_sha256": digest.hexdigest(),
                "range_receipts": receipts,
                "statistics": _sample_statistics(descriptor.dtype, sample, name),
                "semantic_label": None,
                "semantic_limit": "storage statistics do not reveal a fact, sentence, or fixed concept",
            }
        )
    return results


def inspect_repository(
    *,
    repo_id: str,
    revision: str,
    license_name: str,
    transport: RepositoryTransport,
    workers: int = 6,
) -> RepositoryInspection:
    """Inspect an immutable safetensors repository without fetching payloads."""

    validate_repo_id(repo_id)
    validate_revision(revision)
    if not isinstance(license_name, str) or not license_name.strip():
        raise FrontierScanError("license_name must be a non-empty string")
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 16:
        raise FrontierScanError("workers must be an integer within [1, 16]")

    config = transport.get_json("config.json")
    index_raw = transport.get_json("model.safetensors.index.json")
    if not isinstance(config, Mapping):
        raise FrontierScanError("config.json must contain an object")
    weight_map, declared_total = _parse_index(index_raw)
    groups: dict[str, set[str]] = defaultdict(set)
    for name, shard in weight_map.items():
        groups[shard].add(name)

    all_descriptors: list[TensorDescriptor] = []
    shard_sizes: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_read_shard_header, transport, shard, frozenset(names)): shard
            for shard, names in sorted(groups.items())
        }
        try:
            for future in concurrent.futures.as_completed(futures):
                shard = futures[future]
                descriptors, file_bytes, _data_origin = future.result()
                all_descriptors.extend(descriptors)
                shard_sizes[shard] = file_bytes
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    if len(all_descriptors) != len(weight_map):
        raise FrontierScanError("scanned tensor count differs from the index")

    category_bytes: Counter[str] = Counter()
    category_tensors: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    routed_expert_bytes: Counter[int] = Counter()
    layer_bytes: Counter[int] = Counter()
    routed_ids_by_layer: dict[int, set[int]] = defaultdict(set)
    routed_components: dict[tuple[int, int], set[str]] = defaultdict(set)
    for descriptor in all_descriptors:
        category, expert_id = _category(descriptor.name)
        stored = descriptor.stored_bytes
        category_bytes[category] += stored
        category_tensors[category] += 1
        dtype_bytes[descriptor.dtype] += stored
        if expert_id is not None:
            routed_expert_bytes[expert_id] += stored
        layer = _LAYER.search(descriptor.name)
        if layer is not None:
            layer_bytes[int(layer.group(1))] += stored
        routed_site = _LAYER_ROUTED_EXPERT.search(descriptor.name)
        if routed_site is not None:
            layer_id, routed_id = (int(value) for value in routed_site.groups())
            routed_ids_by_layer[layer_id].add(routed_id)
            routed_components[(layer_id, routed_id)].add(
                descriptor.name[routed_site.end() :]
            )

    observed_total = sum(category_bytes.values())
    if observed_total != declared_total:
        raise FrontierScanError(
            f"payload total mismatch: index={declared_total}, headers={observed_total}"
        )
    configured_experts = config.get("n_routed_experts")
    if isinstance(configured_experts, int) and not isinstance(configured_experts, bool):
        if configured_experts < 0 or configured_experts > MAX_TENSORS:
            raise FrontierScanError("config.json n_routed_experts is outside safety bounds")
        expected_ids = set(range(configured_experts))
        if set(routed_expert_bytes) != expected_ids:
            raise FrontierScanError("observed routed expert IDs disagree with config.json")
        for layer_id, observed_ids in sorted(routed_ids_by_layer.items()):
            if observed_ids != expected_ids:
                raise FrontierScanError(
                    f"routed expert IDs in layer {layer_id} disagree with config.json"
                )
        component_reference: frozenset[str] | None = None
        for site, components in sorted(routed_components.items()):
            frozen = frozenset(components)
            if component_reference is None:
                component_reference = frozen
            elif frozen != component_reference:
                raise FrontierScanError(
                    "routed expert component sets are inconsistent across layer/expert sites"
                )

    configured_layers = config.get("num_hidden_layers")
    if isinstance(configured_layers, int) and not isinstance(configured_layers, bool):
        if configured_layers <= 0 or configured_layers > MAX_TENSORS:
            raise FrontierScanError("config.json num_hidden_layers is outside safety bounds")
        invalid_layers = sorted(layer_id for layer_id in layer_bytes if layer_id >= configured_layers)
        if invalid_layers:
            raise FrontierScanError(
                f"observed layer ID exceeds config.json num_hidden_layers: {invalid_layers[0]}"
            )

    receipts = transport_receipts(transport)
    return RepositoryInspection(
        repo_id=repo_id,
        revision=revision,
        license=license_name.strip(),
        config=dict(config),
        source_payload_bytes=observed_total,
        source_file_bytes=sum(shard_sizes.values()),
        tensor_count=len(all_descriptors),
        shard_count=len(shard_sizes),
        category_bytes=dict(category_bytes),
        category_tensors=dict(category_tensors),
        dtype_bytes=dict(dtype_bytes),
        routed_expert_bytes=dict(routed_expert_bytes),
        layer_bytes=dict(layer_bytes),
        transfer_stats=transport.stats(),
        index_sha256=_sha256_json(index_raw),
        config_sha256=_sha256_json(config),
        tensor_inventory_sha256=_tensor_inventory_sha256(all_descriptors),
        index_raw_sha256=_raw_json_hash(receipts, "model.safetensors.index.json"),
        config_raw_sha256=_raw_json_hash(receipts, "config.json"),
        transfer_receipts=receipts,
    )


@dataclass(frozen=True)
class HardwareBudget:
    ram_gib: float = 12.7
    vram_gib: float = 14.5
    disk_gib: float = 63.0
    accelerator: str = "NVIDIA T4"
    native_tensor_formats: tuple[str, ...] = ("FP16", "INT8", "INT4")

    def __post_init__(self) -> None:
        for name in ("ram_gib", "vram_gib", "disk_gib"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise FrontierScanError(f"{name} must be finite and positive")


def build_frontier_plan(
    inspection: RepositoryInspection,
    *,
    hardware: HardwareBudget = HardwareBudget(),
    expert_options: Sequence[int] = (1, 2, 4, 8, 16),
) -> dict[str, Any]:
    """Build a conservative feasibility plan; this never creates a checkpoint."""

    routed_total = inspection.category_bytes.get("routed_experts", 0)
    expert_count = len(inspection.routed_expert_bytes)
    core_bytes = inspection.source_payload_bytes - routed_total
    average_expert_bytes = (routed_total / expert_count) if expert_count else 0.0
    configured_top_k = inspection.config.get("num_experts_per_tok")
    source_formats = set(inspection.dtype_bytes)
    has_fp8 = any(dtype.startswith("F8_") for dtype in source_formats)
    has_packed_experts = inspection.config.get("expert_dtype") == "fp4"
    t4_native = not has_fp8 and not has_packed_experts

    estimates: list[dict[str, Any]] = []
    for option in expert_options:
        if not isinstance(option, int) or isinstance(option, bool) or option <= 0:
            raise FrontierScanError("expert_options must contain positive integers")
        if expert_count and option > expert_count:
            raise FrontierScanError("expert option exceeds the observed expert count")
        stored = int(round(core_bytes + average_expert_bytes * option))
        reduction = 1.0 - (stored / inspection.source_payload_bytes)
        reasons: list[str] = []
        if isinstance(configured_top_k, int) and option < configured_top_k:
            reasons.append(
                f"original router activates top-{configured_top_k}; {option} experts changes its function"
            )
        if reduction > 0.50:
            reasons.append(
                "compression exceeds the conservative reductions explored by the cited "
                "weight-only MoE methods"
            )
        if not t4_native:
            reasons.append("stored FP4/FP8 checkpoint is not a directly executable T4 representation")
        reasons.append("size arithmetic does not demonstrate retained knowledge or language quality")
        estimates.append(
            {
                "routed_experts_retained_or_merged": option,
                "estimated_stored_bytes": stored,
                "estimated_stored_gib": stored / (2**30),
                "stored_reduction_fraction": reduction,
                "fits_colab_disk_as_files": stored < hardware.disk_gib * (2**30),
                "fits_t4_vram_by_file_size_only": stored < hardware.vram_gib * (2**30),
                "executable_checkpoint_proven": False,
                "quality_retention_proven": False,
                "blocking_reasons": reasons,
            }
        )

    return {
        "format": FRONTIER_MANIFEST_FORMAT,
        "format_version": FRONTIER_MANIFEST_VERSION,
        "canonical_lira": False,
        "status": "metadata_scan_complete_checkpoint_build_blocked",
        "inspection": inspection.to_dict(),
        "hardware_assumption": dataclasses.asdict(hardware),
        "candidate_size_estimates": estimates,
        "decision": {
            "remote_range_scanning_passed": True,
            "full_source_materialization_required": False,
            "source_weight_payload_processed": False,
            "standalone_candidate_created": False,
            "standalone_candidate_inference_passed": False,
            "frontier_to_t4_recipe_authorized": False,
            "reason": (
                "No tested weight-only method justifies collapsing this frontier MoE to a T4-sized "
                "checkpoint while retaining its general knowledge; mixed FP4/FP8 also requires a "
                "compatible conversion and runtime."
            ),
            "next_safe_stage": (
                "Implement and validate a factorized MoE backend on a smaller open MoE fixture, "
                "then scale only if reconstruction and behavioural gates pass."
            ),
        },
        "scientific_contract": {
            "knowledge_is_not_a_separable_tensor_field": True,
            "hidden_chain_of_thought_accessed": False,
            "hidden_chain_of_thought_transfer_claimed": False,
            "model_identity_isolated_from_source_weights": False,
            "identity_note": (
                "Source weights can influence behaviour. DragonBRX identity must be supplied by a "
                "separate controller or trained later with owned data and independently evaluated."
            ),
            "a_tensor_sketch_is_an_executable_model": False,
            "smaller_model_can_outperform_source_generally": "not_established",
            "targeted_outperformance": (
                "possible only as an empirical result with tools, retrieval, memory, verification, "
                "specialized data, or additional optimization"
            ),
        },
    }


def token_from_environment(variable: str = "HF_TOKEN") -> str | None:
    """Read an optional Hub token without ever placing it in a manifest."""

    value = os.environ.get(variable)
    return value.strip() if value and value.strip() else None


__all__ = [
    "FRONTIER_MANIFEST_FORMAT",
    "FRONTIER_MANIFEST_VERSION",
    "FrontierScanError",
    "HardwareBudget",
    "HuggingFaceRangeTransport",
    "RangeChunk",
    "RepositoryInspection",
    "RepositoryTransport",
    "TensorDescriptor",
    "TensorLocation",
    "TransferStats",
    "build_frontier_plan",
    "fingerprint_tensors",
    "inspect_repository",
    "locate_tensors",
    "token_from_environment",
    "transport_receipts",
    "validate_repo_id",
    "validate_repository_path",
    "validate_revision",
]
