from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import frontier_tensor_probe
from src.frontier_stream import (
    FrontierScanError,
    RangeChunk,
    TensorDescriptor,
    TensorLocation,
    TransferStats,
    fingerprint_tensors,
    locate_tensors,
)


PINNED_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"


def make_safetensors(tensors: list[tuple[str, str, list[int], bytes]]) -> bytes:
    header: dict[str, Any] = {}
    payload = bytearray()
    for name, dtype, shape, values in tensors:
        start = len(payload)
        payload.extend(values)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + bytes(payload)


class MemoryTransport:
    def __init__(self, json_files: dict[str, Any], binary_files: dict[str, bytes]):
        self.json_files = json_files
        self.binary_files = binary_files
        self.ranges: list[tuple[str, int, int]] = []
        self._requests = 0
        self._bytes = 0

    def get_json(self, path: str, *, max_bytes: int = 64 * 1024 * 1024) -> Any:
        payload = json.dumps(self.json_files[path]).encode("utf-8")
        if len(payload) > max_bytes:
            raise FrontierScanError("fixture JSON too large")
        self._requests += 1
        self._bytes += len(payload)
        return json.loads(payload)

    def get_range(self, path: str, start: int, end: int) -> RangeChunk:
        payload = self.binary_files[path]
        if start < 0 or end < start or end >= len(payload):
            raise FrontierScanError("fixture range outside file")
        result = payload[start : end + 1]
        self.ranges.append((path, start, end))
        self._requests += 1
        self._bytes += len(result)
        return RangeChunk(result, len(payload))

    def stats(self) -> TransferStats:
        return TransferStats(self._requests, self._bytes)


def fixture_transport() -> MemoryTransport:
    # Distinct, deterministic payloads make accidental raw-byte disclosure easy to spot.
    shard_a = make_safetensors(
        [
            ("embed.weight", "BF16", [400, 4], bytes(range(256)) * 12 + bytes(range(128))),
            (
                "layers.0.ffn.experts.0.w1.weight",
                "I8",
                [800, 4],
                bytes(reversed(range(256))) * 12 + bytes(reversed(range(128))),
            ),
            ("unused.same_shard", "F32", [256], b"SAME" * 256),
        ]
    )
    shard_b = make_safetensors(
        [("head.weight", "BF16", [512], b"HEAD" * 256)]
    )
    shard_irrelevant = make_safetensors(
        [("unused.other_shard", "F32", [256], b"NOPE" * 256)]
    )
    mapping = {
        "embed.weight": "model-1.safetensors",
        "layers.0.ffn.experts.0.w1.weight": "model-1.safetensors",
        "unused.same_shard": "model-1.safetensors",
        "head.weight": "model-2.safetensors",
        "unused.other_shard": "model-3.safetensors",
    }
    return MemoryTransport(
        {
            "model.safetensors.index.json": {
                "metadata": {
                    "total_size": sum(len(blob) for blob in (shard_a, shard_b, shard_irrelevant))
                },
                "weight_map": mapping,
            }
        },
        {
            "model-1.safetensors": shard_a,
            "model-2.safetensors": shard_b,
            "model-3.safetensors": shard_irrelevant,
        },
    )


def assert_no_binary_value(test: unittest.TestCase, value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            assert_no_binary_value(test, item)
    elif isinstance(value, list):
        for item in value:
            assert_no_binary_value(test, item)
    else:
        test.assertNotIsInstance(value, (bytes, bytearray, memoryview))


class FrontierTensorLibraryTests(unittest.TestCase):
    def test_locate_reads_only_index_and_headers_of_relevant_shards(self):
        transport = fixture_transport()
        locations, index_sha256 = locate_tensors(
            transport,
            ("embed.weight", "head.weight"),
        )

        self.assertEqual(set(locations), {"embed.weight", "head.weight"})
        self.assertRegex(index_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(transport.stats().requests, 5)  # index + two ranges per shard
        self.assertNotIn("model-3.safetensors", {path for path, _, _ in transport.ranges})
        for path, start, end in transport.ranges:
            if start == 0:
                self.assertEqual(end, 7)
            else:
                header_length = struct.unpack("<Q", transport.binary_files[path][:8])[0]
                self.assertEqual((start, end), (8, 7 + header_length))

    def test_fingerprint_uses_bounded_disjoint_ranges_and_returns_no_raw_bytes(self):
        transport = fixture_transport()
        locations, _ = locate_tensors(
            transport,
            ("embed.weight", "layers.0.ffn.experts.0.w1.weight"),
        )
        transport.ranges.clear()

        samples = fingerprint_tensors(
            transport,
            locations,
            sample_bytes_per_tensor=300,
        )

        self.assertEqual(len(samples), 2)
        self.assertEqual(len(transport.ranges), 6)  # start, middle, and end per tensor
        for sample in samples:
            self.assertLessEqual(sample["sampled_bytes"], 300)
            self.assertLess(sample["sampled_fraction"], 1.0)
            self.assertEqual(sum(row["bytes"] for row in sample["range_receipts"]), sample["sampled_bytes"])
            self.assertRegex(sample["sample_fingerprint_sha256"], r"^[0-9a-f]{64}$")
            self.assertIsNone(sample["semantic_label"])
            self.assertIn("do not reveal", sample["semantic_limit"])
            for receipt in sample["range_receipts"]:
                self.assertLessEqual(receipt["bytes"], 300)
                self.assertLess(receipt["absolute_http_end_inclusive"] - receipt["absolute_http_start"] + 1, len(transport.binary_files[sample["shard"]]))
        expert = next(row for row in samples if ".experts." in row["tensor"])
        self.assertEqual(
            expert["statistics"]["storage_interpretation"],
            "possible_packed_fp4_container_not_dequantized",
        )
        self.assertEqual(set(expert["statistics"]["packed_nibble_histogram"]), {str(i) for i in range(16)})
        assert_no_binary_value(self, samples)
        json.dumps(samples, allow_nan=False)

    def test_rejects_duplicates_missing_names_and_unbounded_sample_budget(self):
        with self.assertRaisesRegex(FrontierScanError, "duplicate requested tensor"):
            locate_tensors(fixture_transport(), ("embed.weight", "embed.weight"))
        with self.assertRaisesRegex(FrontierScanError, "absent from the index"):
            locate_tensors(fixture_transport(), ("does.not.exist",))

        locations, _ = locate_tensors(fixture_transport(), ("embed.weight",))
        for invalid in (255, 65_537, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(FrontierScanError, "within"):
                    fingerprint_tensors(
                        fixture_transport(),
                        locations,
                        sample_bytes_per_tensor=invalid,
                    )

    def test_fingerprint_rejects_unbounded_or_incoherent_location_mappings(self):
        descriptor = TensorDescriptor(
            name="tensor.weight",
            shard="model.safetensors",
            dtype="I8",
            shape=(1_024,),
            data_start=0,
            data_end=1_024,
        )
        location = TensorLocation(
            descriptor=descriptor,
            data_origin=128,
            shard_file_bytes=1_152,
        )
        transport = MemoryTransport({}, {"model.safetensors": bytes(1_152)})

        for locations in ({}, {f"tensor.{index}": location for index in range(17)}):
            with self.subTest(count=len(locations)):
                with self.assertRaisesRegex(FrontierScanError, "between 1 and 16"):
                    fingerprint_tensors(transport, locations, sample_bytes_per_tensor=256)
        with self.assertRaisesRegex(FrontierScanError, "descriptor name"):
            fingerprint_tensors(
                transport,
                {"different.weight": location},
                sample_bytes_per_tensor=256,
            )

    def test_rejects_shard_size_change_during_payload_sampling(self):
        transport = fixture_transport()
        locations, _ = locate_tensors(transport, ("embed.weight",))

        class ChangedSizeTransport(MemoryTransport):
            def get_range(self, path: str, start: int, end: int) -> RangeChunk:
                result = super().get_range(path, start, end)
                return RangeChunk(result.data, result.total_file_bytes + 1)

        changed = ChangedSizeTransport(transport.json_files, transport.binary_files)
        with self.assertRaisesRegex(FrontierScanError, "shard size changed"):
            fingerprint_tensors(changed, locations, sample_bytes_per_tensor=256)


class FrontierTensorCliTests(unittest.TestCase):
    def test_frontier_source_is_immutable_and_truthful_contract_is_written(self):
        self.assertEqual(frontier_tensor_probe.SOURCE_MODEL, "deepseek-ai/DeepSeek-V4-Flash")
        self.assertEqual(frontier_tensor_probe.SOURCE_REVISION, PINNED_REVISION)
        self.assertRegex(frontier_tensor_probe.SOURCE_REVISION, r"^[0-9a-f]{40}$")
        self.assertEqual(frontier_tensor_probe.SOURCE_LICENSE, "MIT")

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "probe.lira.json"
            transport = fixture_transport()
            with patch.object(
                frontier_tensor_probe,
                "HuggingFaceRangeTransport",
                return_value=transport,
            ):
                exit_code = frontier_tensor_probe.main(
                    [
                        "--tensor",
                        "embed.weight",
                        "--sample-bytes",
                        "256",
                        "--json-output",
                        str(destination),
                    ]
                )

            self.assertEqual(exit_code, 0)
            report = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(report["source"]["revision"], PINNED_REVISION)
            self.assertFalse(report["canonical_lira"])
            self.assertEqual(report["transfer"]["full_shards_downloaded"], 0)
            self.assertFalse(report["transfer"]["donor_model_instantiated"])
            self.assertEqual(report["transfer"]["donor_forward_calls"], 0)
            limits = report["scientific_limits"]
            self.assertFalse(limits["standalone_parameter_can_generate_text"])
            self.assertTrue(limits["sample_classifies_structure_and_storage_only"])
            self.assertFalse(limits["sample_decodes_semantic_knowledge"])
            self.assertFalse(limits["sample_decodes_hidden_chain_of_thought"])
            self.assertFalse(limits["whole_tensor_integrity_proven"])
            self.assertEqual(
                report["transfer"]["tensor_payload_bytes_sampled"],
                sum(row["sampled_bytes"] for row in report["samples"]),
            )
            assert_no_binary_value(self, report)

    def test_default_tensor_set_is_small_and_covers_structural_roles(self):
        self.assertEqual(len(frontier_tensor_probe.DEFAULT_TENSORS), 6)
        self.assertEqual(len(set(frontier_tensor_probe.DEFAULT_TENSORS)), 6)
        self.assertIn("embed.weight", frontier_tensor_probe.DEFAULT_TENSORS)
        self.assertIn("head.weight", frontier_tensor_probe.DEFAULT_TENSORS)
        self.assertTrue(any(".experts." in name for name in frontier_tensor_probe.DEFAULT_TENSORS))
        self.assertTrue(any("shared_experts" in name for name in frontier_tensor_probe.DEFAULT_TENSORS))


if __name__ == "__main__":
    unittest.main()
