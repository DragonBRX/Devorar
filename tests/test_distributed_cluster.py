from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import distributed_server
import distributed_worker
import distributed_train_head
from src.distributed_cluster import (
    ClusterError,
    ClusterState,
    evenly_spaced_indices,
    sign_request,
    verify_request_signature,
)
from src.frontier_compute import deterministic_float32_hidden_seeded
from src.frontier_stream import RangeChunk, TransferStats


def encode_bf16(values: list[float]) -> bytes:
    import struct

    payload = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        payload.extend(struct.pack("<H", bits >> 16))
    return bytes(payload)


class MemoryTransport:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.requests = 0
        self.bytes = 0

    def get_range(self, path: str, start: int, end: int) -> RangeChunk:
        data = self.payload[start : end + 1]
        self.requests += 1
        self.bytes += len(data)
        return RangeChunk(
            data=data,
            total_file_bytes=len(self.payload),
            etag='"fixture"',
            final_url="https://huggingface.co/fixture/model.safetensors",
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def stats(self) -> TransferStats:
        return TransferStats(self.requests, self.bytes)

    def receipts(self):
        return ()


class DistributedClusterTests(unittest.TestCase):
    def test_signature_roundtrip_and_tamper_detection(self):
        token = "x" * 32
        body = b'{"a":1}'
        timestamp = "1000.0"
        nonce = "a" * 32
        signature = sign_request(token, "POST", "/v1/test", body, timestamp, nonce)
        verify_request_signature(
            token,
            "POST",
            "/v1/test",
            body,
            timestamp,
            nonce,
            signature,
            now=1000.0,
        )
        with self.assertRaises(ClusterError):
            verify_request_signature(
                token,
                "POST",
                "/v1/test",
                body + b"x",
                timestamp,
                nonce,
                signature,
                now=1000.0,
            )

    def test_state_claim_result_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "cluster.sqlite3"
            state = ClusterState(database, lease_seconds=30, max_attempts=3)
            specs = [
                {"operation": "x", "value": 1},
                {"operation": "x", "value": 2},
            ]
            self.assertEqual(state.add_jobs(specs), 2)
            self.assertEqual(state.add_jobs(specs), 0)
            worker = state.register_worker("phone", {"cpu": 8})
            job = state.claim_job(worker)
            self.assertIsNotNone(job)
            assert job is not None
            state.submit_result(
                worker,
                job.job_id,
                job.spec_sha256,
                ok=True,
                result={"answer": 7},
            )
            status = state.status()
            self.assertEqual(status["jobs"]["done"], 1)
            self.assertEqual(status["jobs"]["queued"], 1)
            reopened = ClusterState(database, lease_seconds=30, max_attempts=3)
            self.assertEqual(len(reopened.results()), 1)

    def test_termux_install_block_is_self_contained(self):
        token = "t" * 32
        block = distributed_server.build_termux_install_block("http://192.168.1.10:8765", token, 2)
        self.assertIn("pkg install -y python git tmux", block)
        self.assertIn("git clone --depth 1 https://github.com/DragonBRX/Devorar.git", block)
        self.assertIn("DEVORAR_SERVER=http://192.168.1.10:8765", block)
        self.assertIn(f"DEVORAR_CLUSTER_TOKEN={token}", block)
        self.assertIn("DEVORAR_PROCESSES=2", block)
        self.assertIn("./termux_install.sh", block)

    def test_even_spacing_and_seeded_hidden(self):
        self.assertEqual(evenly_spaced_indices(10, 4), (0, 3, 6, 9))
        first = deterministic_float32_hidden_seeded(64, 123)
        second = deterministic_float32_hidden_seeded(64, 123)
        other = deterministic_float32_hidden_seeded(64, 124)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        rms = math.sqrt(sum(value * value for value in first) / len(first))
        self.assertAlmostEqual(rms, 1.0, places=6)

    def test_training_matrix_collects_rectangular_teacher_targets(self):
        records = [
            {
                "result": {
                    "operation": "remote_bf16_head_teacher",
                    "source": {"repo_id": "owner/model", "revision": "a" * 40},
                    "tensor": "head.weight",
                    "shape": [4, 4],
                    "samples": [
                        {"seed": 1, "row_logits": [{"row_id": 0, "value": 1.0}, {"row_id": 1, "value": 2.0}]},
                        {"seed": 2, "row_logits": [{"row_id": 0, "value": 3.0}, {"row_id": 1, "value": 4.0}]},
                    ],
                }
            }
        ]
        matrix = distributed_train_head.collect_training_matrix(records)
        self.assertEqual(matrix["seeds"], [1, 2])
        self.assertEqual(matrix["row_ids"], [0, 1])
        self.assertEqual(matrix["targets"], [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(len(matrix["inputs"][0]), 4)

    def test_worker_executes_remote_bf16_rows_and_returns_teacher_samples(self):
        prefix = b"HEADER00"
        raw = encode_bf16([
            1.0, 2.0, 3.0, 4.0,
            -1.0, 0.5, 2.0, -2.0,
            0.25, 0.5, 0.75, 1.0,
            8.0, 4.0, 2.0, 1.0,
        ])
        payload = prefix + raw + b"TAIL"
        transport = MemoryTransport(payload)
        spec = {
            "operation": "remote_bf16_head_teacher",
            "source": {
                "repo_id": "owner/model",
                "revision": "a" * 40,
                "index_sha256": "b" * 64,
            },
            "tensor": {
                "name": "head.weight",
                "shard": "model.safetensors",
                "dtype": "BF16",
                "shape": [4, 4],
                "data_start": 0,
                "data_end": len(raw),
                "data_origin": len(prefix),
                "shard_file_bytes": len(payload),
            },
            "row_ids": [1, 3],
            "sample_seeds": [1, 2],
        }
        with patch.object(distributed_worker, "HuggingFaceRangeTransport", return_value=transport):
            result = distributed_worker.execute_remote_head_job(spec, "HF_TOKEN")
        self.assertEqual(result["row_ids"], [1, 3])
        self.assertEqual(len(result["samples"]), 2)
        self.assertEqual(len(result["samples"][0]["row_logits"]), 2)
        self.assertGreater(result["timing_seconds"]["compute"], 0.0)
        self.assertEqual(result["weight_payload_bytes"], 16)


if __name__ == "__main__":
    unittest.main()
