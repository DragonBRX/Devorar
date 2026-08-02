from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import frontier_head_parity
from src.frontier_compute import (
    compare_logits,
    decode_bfloat16_le,
    deterministic_float32_hidden,
    load_complete_bf16_rows,
    reference_linear_logits,
    select_evenly_spaced_rows,
)
from src.frontier_stream import (
    FrontierScanError,
    RangeChunk,
    TensorDescriptor,
    TensorLocation,
    TransferStats,
)


PINNED_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"


def encode_bf16(values: list[float]) -> bytes:
    payload = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        payload.extend(struct.pack("<H", bits >> 16))
    return bytes(payload)


class MemoryRangeTransport:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.ranges: list[tuple[int, int]] = []
        self._bytes = 0

    def get_range(self, path: str, start: int, end: int) -> RangeChunk:
        if path != "model.safetensors" or start < 0 or end >= len(self.payload):
            raise AssertionError("unexpected fixture range")
        data = self.payload[start : end + 1]
        self.ranges.append((start, end))
        self._bytes += len(data)
        return RangeChunk(
            data=data,
            total_file_bytes=len(self.payload),
            etag='"fixture"',
            final_url="https://huggingface.co/fixture/model.safetensors",
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def stats(self) -> TransferStats:
        return TransferStats(len(self.ranges), self._bytes)


def fixture_location() -> tuple[MemoryRangeTransport, TensorLocation]:
    prefix = b"HEADER00"
    rows = [
        [1.0, 2.0, 3.0, 4.0],
        [-1.0, 0.5, 2.0, -2.0],
        [0.25, 0.5, 0.75, 1.0],
        [8.0, 4.0, 2.0, 1.0],
    ]
    raw = encode_bf16([value for row in rows for value in row])
    suffix = b"TAIL"
    payload = prefix + raw + suffix
    descriptor = TensorDescriptor(
        name="head.weight",
        shard="model.safetensors",
        dtype="BF16",
        shape=(4, 4),
        data_start=0,
        data_end=len(raw),
    )
    return MemoryRangeTransport(payload), TensorLocation(descriptor, len(prefix), len(payload))


class FrontierComputeLibraryTests(unittest.TestCase):
    def test_row_selection_is_deterministic_bounded_and_spans_tensor(self):
        self.assertEqual(select_evenly_spaced_rows(10, 4), (0, 3, 6, 9))
        self.assertEqual(select_evenly_spaced_rows(10, 1), (0,))
        for invalid in (0, 65, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(FrontierScanError):
                    select_evenly_spaced_rows(100, invalid)

    def test_bfloat16_decoder_and_complete_row_loader(self):
        self.assertEqual(decode_bfloat16_le(encode_bf16([1.0, -2.0])), (1.0, -2.0))
        transport, location = fixture_location()
        loaded = load_complete_bf16_rows(transport, location, (1, 3))

        self.assertEqual(loaded.row_ids, (1, 3))
        self.assertEqual(loaded.values[0], (-1.0, 0.5, 2.0, -2.0))
        self.assertEqual(loaded.values[1], (8.0, 4.0, 2.0, 1.0))
        self.assertEqual(loaded.payload_bytes, 16)
        self.assertRegex(loaded.payload_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(transport.ranges, [(16, 23), (32, 39)])
        self.assertTrue(all(row["bytes"] == 8 for row in loaded.range_receipts))

    def test_reference_executes_linear_operator_and_comparison_gates_errors(self):
        logits = reference_linear_logits(((1.0, 2.0), (3.0, 4.0)), (0.5, -1.0))
        self.assertEqual(logits, (-1.5, -2.5))
        passed = compare_logits(logits, (-1.50001, -2.50001), atol=2e-4, rtol=2e-4)
        failed = compare_logits(logits, (-1.4, -2.5), atol=2e-4, rtol=2e-4)
        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])

    def test_hidden_input_is_reproducible_float32_and_rms_normalized(self):
        first = deterministic_float32_hidden(128)
        second = deterministic_float32_hidden(128)
        self.assertEqual(first, second)
        rms = math.sqrt(sum(value * value for value in first) / len(first))
        self.assertAlmostEqual(rms, 1.0, places=6)

    def test_loader_rejects_duplicate_rows_quantized_data_and_size_changes(self):
        transport, location = fixture_location()
        with self.assertRaisesRegex(FrontierScanError, "duplicate row"):
            load_complete_bf16_rows(transport, location, (0, 0))

        quantized = TensorLocation(
            TensorDescriptor("head.weight", "model.safetensors", "I8", (4, 4), 0, 16),
            8,
            len(transport.payload),
        )
        with self.assertRaisesRegex(FrontierScanError, "only BF16"):
            load_complete_bf16_rows(transport, quantized, (0,))

        class ChangedSizeTransport(MemoryRangeTransport):
            def get_range(self, path: str, start: int, end: int) -> RangeChunk:
                result = super().get_range(path, start, end)
                return RangeChunk(result.data, result.total_file_bytes + 1)

        with self.assertRaisesRegex(FrontierScanError, "size changed"):
            load_complete_bf16_rows(ChangedSizeTransport(transport.payload), location, (0,))


class FrontierHeadParityCliTests(unittest.TestCase):
    def test_source_is_pinned_and_defaults_are_bounded(self):
        self.assertEqual(frontier_head_parity.SOURCE_MODEL, "deepseek-ai/DeepSeek-V4-Flash")
        self.assertEqual(frontier_head_parity.SOURCE_REVISION, PINNED_REVISION)
        self.assertEqual(frontier_head_parity.SOURCE_TENSOR, "head.weight")
        args = frontier_head_parity.build_parser().parse_args([])
        self.assertEqual(args.row_count, 16)
        self.assertEqual(args.device, "auto")
        self.assertFalse(args.require_cuda)

    def test_report_separates_slice_parity_from_full_model_logits(self):
        transport, location = fixture_location()
        args = argparse.Namespace(
            row_count=2,
            device="cuda",
            require_cuda=True,
            atol=2e-4,
            rtol=2e-4,
            hf_token_env="HF_TOKEN",
            timeout_seconds=60.0,
        )
        with (
            patch.object(frontier_head_parity, "HuggingFaceRangeTransport", return_value=transport),
            patch.object(
                frontier_head_parity,
                "locate_tensors",
                return_value=({"head.weight": location}, "a" * 64),
            ),
            patch.object(
                frontier_head_parity,
                "_candidate_logits",
                side_effect=lambda weights, hidden, **kwargs: (
                    reference_linear_logits(weights, hidden),
                    {
                        "framework": "fixture",
                        "selected_device": "cuda",
                        "device_name": "Fixture T4",
                        "dtype": "float32",
                    },
                ),
            ),
            patch.object(frontier_head_parity, "transport_receipts", return_value=()),
        ):
            report = frontier_head_parity.run(args)

        self.assertEqual(report["status"], "passed_bounded_head_slice_parity")
        self.assertTrue(report["evidence"]["remote_checkpoint_bytes_became_numeric_weights"])
        self.assertTrue(report["evidence"]["cuda_operator_executed"])
        limits = report["scientific_limits"]
        self.assertFalse(limits["partial_slice_logits_are_full_model_logits"])
        self.assertFalse(limits["official_deepseek_runtime_logits_compared"])
        self.assertFalse(limits["next_token_or_phrase_attributable_to_deepseek"])
        json.dumps(report, allow_nan=False)

    def test_main_writes_report_and_returns_distinct_parity_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "report.json"
            failed = {
                "parity": {"passed": False},
                "source": {
                    "repo_id": "fixture",
                    "revision": "a",
                    "tensor": "head.weight",
                    "dtype": "BF16",
                    "shape": [2, 2],
                },
                "executed_slice": {"selected_row_count": 2, "weight_payload_bytes": 8},
                "candidate": {"selected_device": "cpu", "device_name": "fixture"},
            }
            with (
                patch.object(frontier_head_parity, "run", return_value=failed),
                patch.object(frontier_head_parity, "print_report"),
            ):
                status = frontier_head_parity.main(["--json-output", str(destination)])
            self.assertEqual(status, 3)
            written = json.loads(destination.read_text(encoding="utf-8"))
            self.assertFalse(written["parity"]["passed"])


if __name__ == "__main__":
    unittest.main()
