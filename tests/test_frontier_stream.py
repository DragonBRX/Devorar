from __future__ import annotations

import json
import hashlib
import struct
import types
import unittest
import urllib.request
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import src.frontier_stream as frontier_stream
from src.frontier_stream import (
    FrontierScanError,
    HuggingFaceRangeTransport,
    RangeChunk,
    TransferStats,
    build_frontier_plan,
    inspect_repository,
    validate_repo_id,
    validate_repository_path,
    validate_revision,
)


REPO = "owner/model"
REVISION = "a" * 40


def make_safetensors(tensors: list[tuple[str, str, list[int], int]]) -> bytes:
    """Build a valid metadata fixture; payload values are intentionally meaningless."""
    offset = 0
    header = {}
    payload = bytearray()
    for name, dtype, shape, byte_count in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + byte_count],
        }
        payload.extend(bytes([len(payload) % 251]) * byte_count)
        offset += byte_count
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + bytes(payload)


class MemoryTransport:
    def __init__(self, json_files, binary_files):
        self.json_files = json_files
        self.binary_files = binary_files
        self.ranges = []
        self._requests = 0
        self._bytes = 0

    def get_json(self, path, *, max_bytes=64 * 1024 * 1024):
        payload = json.dumps(self.json_files[path]).encode("utf-8")
        if len(payload) > max_bytes:
            raise FrontierScanError("fixture JSON too large")
        self._requests += 1
        self._bytes += len(payload)
        return json.loads(payload)

    def get_range(self, path, start, end):
        payload = self.binary_files[path]
        if start < 0 or end >= len(payload):
            raise FrontierScanError("fixture range outside file")
        result = payload[start : end + 1]
        self.ranges.append((path, start, end))
        self._requests += 1
        self._bytes += len(result)
        return RangeChunk(result, len(payload))

    def stats(self):
        return TransferStats(self._requests, self._bytes)


def fixture_transport(*, mismatch=False):
    shard_a = make_safetensors(
        [
            ("embed.weight", "BF16", [4, 2], 16),
            ("layers.0.ffn.shared_experts.w1.weight", "F8_E4M3", [2, 2], 4),
            ("layers.0.ffn.experts.0.w1.weight", "I8", [2, 2], 4),
        ]
    )
    shard_b = make_safetensors(
        [
            ("layers.0.ffn.experts.1.w1.weight", "I8", [2, 2], 4),
            ("layers.0.ffn.gate.weight", "F32", [2, 2], 16),
            ("head.weight", "BF16", [4, 2], 16),
        ]
    )
    mapping = {
        "embed.weight": "model-1.safetensors",
        "layers.0.ffn.shared_experts.w1.weight": "model-1.safetensors",
        "layers.0.ffn.experts.0.w1.weight": "model-1.safetensors",
        "layers.0.ffn.experts.1.w1.weight": "model-2.safetensors",
        "layers.0.ffn.gate.weight": "model-2.safetensors",
        "head.weight": "model-2.safetensors",
    }
    if mismatch:
        mapping["ghost.weight"] = "model-2.safetensors"
    payload_total = 16 + 4 + 4 + 4 + 16 + 16
    return MemoryTransport(
        {
            "config.json": {
                "model_type": "tiny_moe",
                "architectures": ["TinyMoeForCausalLM"],
                "hidden_size": 2,
                "num_hidden_layers": 1,
                "n_routed_experts": 2,
                "n_shared_experts": 1,
                "num_experts_per_tok": 1,
                "expert_dtype": "fp4",
            },
            "model.safetensors.index.json": {
                "metadata": {"total_size": payload_total},
                "weight_map": mapping,
            },
        },
        {
            "model-1.safetensors": shard_a,
            "model-2.safetensors": shard_b,
        },
    )


class FrontierInspectionTests(unittest.TestCase):
    def test_scans_only_prefixes_and_headers(self):
        transport = fixture_transport()
        result = inspect_repository(
            repo_id=REPO,
            revision=REVISION,
            license_name="MIT",
            transport=transport,
            workers=2,
        )
        self.assertEqual(result.tensor_count, 6)
        self.assertEqual(result.shard_count, 2)
        self.assertEqual(result.source_payload_bytes, 60)
        self.assertEqual(result.category_bytes["routed_experts"], 8)
        self.assertEqual(result.category_bytes["shared_experts"], 4)
        self.assertEqual(result.category_bytes["embedding_or_head"], 32)
        self.assertEqual(result.category_bytes["router_or_gate"], 16)
        self.assertEqual(result.routed_expert_bytes, {0: 4, 1: 4})
        self.assertEqual(result.transfer_stats.requests, 6)

        per_file = Counter(path for path, _, _ in transport.ranges)
        self.assertEqual(per_file, {"model-1.safetensors": 2, "model-2.safetensors": 2})
        for path, start, end in transport.ranges:
            if start == 0:
                self.assertEqual(end, 7)
            else:
                header_length = struct.unpack("<Q", transport.binary_files[path][:8])[0]
                self.assertEqual((start, end), (8, 7 + header_length))

    def test_plan_marks_size_math_as_non_executable(self):
        inspection = inspect_repository(
            repo_id=REPO,
            revision=REVISION,
            license_name="MIT",
            transport=fixture_transport(),
            workers=1,
        )
        plan = build_frontier_plan(inspection, expert_options=(1, 2))
        self.assertEqual(plan["status"], "metadata_scan_complete_checkpoint_build_blocked")
        self.assertTrue(plan["decision"]["remote_range_scanning_passed"])
        self.assertFalse(plan["decision"]["frontier_to_t4_recipe_authorized"])
        self.assertFalse(plan["candidate_size_estimates"][0]["executable_checkpoint_proven"])
        self.assertFalse(plan["scientific_contract"]["hidden_chain_of_thought_transfer_claimed"])

    def test_rejects_index_header_mismatch(self):
        with self.assertRaisesRegex(FrontierScanError, "index/header mismatch"):
            inspect_repository(
                repo_id=REPO,
                revision=REVISION,
                license_name="MIT",
                transport=fixture_transport(mismatch=True),
                workers=1,
            )

    def test_rejects_overlapping_offsets(self):
        transport = fixture_transport()
        payload = transport.binary_files["model-1.safetensors"]
        length = struct.unpack("<Q", payload[:8])[0]
        header = json.loads(payload[8 : 8 + length])
        header["layers.0.ffn.shared_experts.w1.weight"]["data_offsets"] = [15, 19]
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        transport.binary_files["model-1.safetensors"] = (
            struct.pack("<Q", len(encoded)) + encoded + payload[8 + length :]
        )
        with self.assertRaisesRegex(FrontierScanError, "overlapping tensor offsets"):
            inspect_repository(
                repo_id=REPO,
                revision=REVISION,
                license_name="MIT",
                transport=transport,
                workers=1,
            )

    def test_rejects_gaps_and_unmapped_trailing_bytes(self):
        transport = fixture_transport()
        payload = transport.binary_files["model-1.safetensors"]
        length = struct.unpack("<Q", payload[:8])[0]
        header = json.loads(payload[8 : 8 + length])
        header["layers.0.ffn.shared_experts.w1.weight"]["data_offsets"] = [17, 21]
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        transport.binary_files["model-1.safetensors"] = (
            struct.pack("<Q", len(encoded)) + encoded + payload[8 + length :]
        )
        with self.assertRaisesRegex(FrontierScanError, "gapped tensor offsets"):
            inspect_repository(
                repo_id=REPO,
                revision=REVISION,
                license_name="MIT",
                transport=transport,
                workers=1,
            )

    def test_rejects_expert_ids_that_are_complete_only_after_global_aggregation(self):
        shard_a = make_safetensors(
            [("layers.0.ffn.experts.0.w1.weight", "I8", [2, 2], 4)]
        )
        shard_b = make_safetensors(
            [("layers.1.ffn.experts.1.w1.weight", "I8", [2, 2], 4)]
        )
        transport = MemoryTransport(
            {
                "config.json": {
                    "num_hidden_layers": 2,
                    "n_routed_experts": 2,
                },
                "model.safetensors.index.json": {
                    "metadata": {"total_size": 8},
                    "weight_map": {
                        "layers.0.ffn.experts.0.w1.weight": "model-1.safetensors",
                        "layers.1.ffn.experts.1.w1.weight": "model-2.safetensors",
                    },
                },
            },
            {
                "model-1.safetensors": shard_a,
                "model-2.safetensors": shard_b,
            },
        )
        with self.assertRaisesRegex(FrontierScanError, "expert IDs in layer"):
            inspect_repository(
                repo_id=REPO,
                revision=REVISION,
                license_name="MIT",
                transport=transport,
                workers=1,
            )

    def test_rejects_inconsistent_components_between_experts(self):
        tensors = [
            ("layers.0.ffn.experts.0.w1.weight", "I8", [2, 2], 4),
            ("layers.0.ffn.experts.0.w2.weight", "I8", [2, 2], 4),
            ("layers.0.ffn.experts.1.w1.weight", "I8", [2, 2], 4),
        ]
        shard = make_safetensors(tensors)
        mapping = {name: "model.safetensors" for name, *_rest in tensors}
        transport = MemoryTransport(
            {
                "config.json": {
                    "num_hidden_layers": 1,
                    "n_routed_experts": 2,
                },
                "model.safetensors.index.json": {
                    "metadata": {"total_size": 12},
                    "weight_map": mapping,
                },
            },
            {"model.safetensors": shard},
        )
        with self.assertRaisesRegex(FrontierScanError, "component sets are inconsistent"):
            inspect_repository(
                repo_id=REPO,
                revision=REVISION,
                license_name="MIT",
                transport=transport,
                workers=1,
            )


class InputAndHttpSafetyTests(unittest.TestCase):
    def test_requires_safe_repo_commit_and_path(self):
        self.assertEqual(validate_repo_id(REPO), REPO)
        self.assertEqual(validate_revision(REVISION), REVISION)
        self.assertEqual(validate_repository_path("nested/model.safetensors"), "nested/model.safetensors")
        for invalid in ("one-part", "../owner/model", "owner/model/extra", "owner/$model"):
            with self.assertRaises(FrontierScanError):
                validate_repo_id(invalid)
        for invalid in ("main", "A" * 40, "a" * 39, "a" * 41):
            with self.assertRaises(FrontierScanError):
                validate_revision(invalid)
        for invalid in ("../model.safetensors", "/model.safetensors", "a\\b.safetensors"):
            with self.assertRaises(FrontierScanError):
                validate_repository_path(invalid)

    def test_http_range_reader_rejects_full_response_without_reading_body(self):
        class Response:
            status = 200
            headers = {"Content-Length": "999999999"}
            read_called = False

            def read(self, *_args):
                self.read_called = True
                raise AssertionError("full shard body must not be read")

            def close(self):
                pass

        response = Response()
        transport = HuggingFaceRangeTransport(REPO, REVISION, retries=1)
        with patch.object(transport, "_open", return_value=response):
            with self.assertRaisesRegex(FrontierScanError, "ignored byte range"):
                transport.get_range("model.safetensors", 0, 7)
        self.assertFalse(response.read_called)

    def test_redirect_strips_token_cross_origin_and_rejects_untrusted_hosts(self):
        handler = frontier_stream._SafeHuggingFaceRedirectHandler()
        request = urllib.request.Request(
            "https://huggingface.co/owner/model/resolve/" + REVISION + "/model.safetensors",
            headers={"Authorization": "Bearer SECRET", "Range": "bytes=0-7"},
        )
        redirected = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://cas-bridge.xethub.hf.co/blob/model.safetensors",
        )
        self.assertIsNotNone(redirected)
        self.assertIsNone(redirected.get_header("Authorization"))
        self.assertEqual(redirected.get_header("Range"), "bytes=0-7")
        for url in (
            "http://huggingface.co/model.safetensors",
            "https://attacker.invalid/model.safetensors",
        ):
            with self.subTest(url=url), self.assertRaises(FrontierScanError):
                handler.redirect_request(request, None, 302, "Found", {}, url)

    def test_rejects_unknown_dtype_and_shape_byte_mismatch(self):
        for dtype, shape, message in (
            ("HAX", [1], "unknown or invalid dtype"),
            ("I8", [999], "shape/dtype byte size mismatch"),
        ):
            with self.subTest(dtype=dtype, shape=shape):
                blob = make_safetensors([("x", dtype, shape, 1)])
                transport = MemoryTransport(
                    {
                        "config.json": {"n_routed_experts": 0},
                        "model.safetensors.index.json": {
                            "metadata": {"total_size": 1},
                            "weight_map": {"x": "model.safetensors"},
                        },
                    },
                    {"model.safetensors": blob},
                )
                with self.assertRaisesRegex(FrontierScanError, message):
                    inspect_repository(
                        repo_id=REPO,
                        revision=REVISION,
                        license_name="MIT",
                        transport=transport,
                        workers=1,
                    )

    def test_range_requests_are_bound_to_one_strong_etag(self):
        class Response:
            status = 206

            def __init__(self, start, end, etag, payload):
                self.headers = {
                    "Content-Range": f"bytes {start}-{end}/100",
                    "Content-Encoding": "identity",
                    "ETag": etag,
                }
                self.payload = payload

            def read(self, _limit):
                return self.payload

            def close(self):
                pass

        responses = [
            Response(0, 7, '"first"', b"12345678"),
            Response(8, 15, '"changed"', b"abcdefgh"),
        ]
        requests = []

        def fake_open(request):
            requests.append(request)
            return responses.pop(0)

        transport = HuggingFaceRangeTransport(REPO, REVISION, retries=1)
        with patch.object(transport, "_open", side_effect=fake_open):
            first = transport.get_range("model.safetensors", 0, 7)
            self.assertEqual(first.etag, '"first"')
            with self.assertRaisesRegex(FrontierScanError, "identity changed"):
                transport.get_range("model.safetensors", 8, 15)
        self.assertEqual(requests[1].get_header("If-range"), '"first"')

    def test_json_receipt_hashes_exact_bytes_and_redacts_signed_query(self):
        payload = b'{"model_type":"tiny"}'

        class Response:
            headers = {
                "Content-Length": str(len(payload)),
                "Content-Encoding": "identity",
                "ETag": '"json"',
            }

            def read(self, _limit):
                return payload

            def close(self):
                pass

            def geturl(self):
                return "https://cdn-lfs.huggingface.co/config.json?X-Amz-Signature=SECRET"

        transport = HuggingFaceRangeTransport(REPO, REVISION, retries=1)
        with patch.object(transport, "_open", return_value=Response()):
            self.assertEqual(transport.get_json("config.json"), {"model_type": "tiny"})
        receipt = transport.receipts()[0]
        self.assertEqual(receipt.sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(receipt.final_url, "https://cdn-lfs.huggingface.co/config.json")
        self.assertNotIn("SECRET", json.dumps(receipt.to_dict()))

    def test_strict_json_rejects_duplicate_tensor_keys(self):
        header = b'{"x":{"dtype":"I8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"I8","shape":[1],"data_offsets":[1,2]}}'
        blob = struct.pack("<Q", len(header)) + header + b"xx"
        transport = MemoryTransport(
            {
                "config.json": {"n_routed_experts": 0},
                "model.safetensors.index.json": {
                    "metadata": {"total_size": 1},
                    "weight_map": {"x": "model.safetensors"},
                },
            },
            {"model.safetensors": blob},
        )
        with self.assertRaisesRegex(FrontierScanError, "duplicate JSON key"):
            inspect_repository(
                repo_id=REPO,
                revision=REVISION,
                license_name="MIT",
                transport=transport,
                workers=1,
            )

    def test_strict_json_rejects_nonfinite_numbers(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value), self.assertRaisesRegex(
                FrontierScanError,
                "non-finite JSON number",
            ):
                frontier_stream._strict_json_loads(
                    f'{{"value":{value}}}'.encode("ascii"),
                    "fixture.json",
                )


if __name__ == "__main__":
    unittest.main()
