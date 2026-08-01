from __future__ import annotations

import argparse
import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import frontier_scan
import start_frontier_colab
from src.frontier_stream import TransferStats


def fake_inspection():
    return SimpleNamespace(
        source_payload_bytes=1_000,
        transfer_stats=TransferStats(requests=4, bytes_received=200),
    )


def fake_plan():
    return {
        "format": "lira.experimental.frontier-weight-map",
        "format_version": 1,
        "canonical_lira": False,
        "status": "metadata_scan_complete_checkpoint_build_blocked",
        "inspection": {
            "inventory": {
                "tensor_count": 2,
                "shard_count": 1,
                "source_payload_bytes": 1_000,
                "categories": [
                    {"category": "routed_experts", "stored_bytes": 600},
                ],
            },
            "scan": {"bytes_received": 200, "full_shards_downloaded": 0},
        },
        "decision": {"frontier_to_t4_recipe_authorized": False},
    }


class FrontierCliTests(unittest.TestCase):
    def test_sources_parse_and_pin_immutable_frontier_model(self):
        for path in (Path("frontier_scan.py"), Path("start_frontier_colab.py")):
            ast.parse(path.read_text(encoding="utf-8"))
        self.assertEqual(frontier_scan.SOURCE_MODEL, "deepseek-ai/DeepSeek-V4-Flash")
        self.assertEqual(
            frontier_scan.SOURCE_REVISION,
            "60d8d70770c6776ff598c94bb586a859a38244f1",
        )
        self.assertEqual(frontier_scan.SOURCE_LICENSE, "MIT")

    def test_runner_writes_inspection_only_manifest_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "frontier"
            args = frontier_scan.build_parser().parse_args(
                ["--output-dir", str(output), "--overwrite-output"]
            )
            with (
                patch.object(frontier_scan, "HuggingFaceRangeTransport", return_value=object()),
                patch.object(frontier_scan, "inspect_repository", return_value=fake_inspection()),
                patch.object(frontier_scan, "build_frontier_plan", return_value=fake_plan()),
            ):
                result = frontier_scan.run(args)

            self.assertEqual(result["status"], "metadata_scan_complete_checkpoint_build_blocked")
            self.assertEqual(
                (output / frontier_scan.OUTPUT_SENTINEL_NAME).read_text(encoding="utf-8"),
                frontier_scan.OUTPUT_SENTINEL_CONTENT,
            )
            manifest = json.loads((output / frontier_scan.MANIFEST_NAME).read_text(encoding="utf-8"))
            summary = json.loads((output / frontier_scan.SUMMARY_NAME).read_text(encoding="utf-8"))
            self.assertFalse(manifest["decision"]["frontier_to_t4_recipe_authorized"])
            self.assertFalse(summary["standalone_candidate_created"])
            self.assertEqual(summary["full_shards_downloaded"], 0)

    def test_failed_scan_preserves_existing_managed_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "frontier"
            output.mkdir()
            (output / frontier_scan.OUTPUT_SENTINEL_NAME).write_text(
                frontier_scan.OUTPUT_SENTINEL_CONTENT,
                encoding="utf-8",
            )
            (output / "old.txt").write_text("preserve", encoding="utf-8")
            args = frontier_scan.build_parser().parse_args(
                ["--output-dir", str(output), "--overwrite-output"]
            )
            with patch.object(frontier_scan, "_run_staged", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    frontier_scan.run(args)
            self.assertEqual((output / "old.txt").read_text(encoding="utf-8"), "preserve")

    def test_refuses_to_overwrite_unmanaged_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "foreign"
            output.mkdir()
            args = frontier_scan.build_parser().parse_args(
                ["--output-dir", str(output), "--overwrite-output"]
            )
            with self.assertRaisesRegex(RuntimeError, "unmanaged"):
                frontier_scan.run(args)

    def test_backup_cleanup_error_does_not_misreport_successful_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            target = parent / "frontier"
            target.mkdir()
            (target / frontier_scan.OUTPUT_SENTINEL_NAME).write_text(
                frontier_scan.OUTPUT_SENTINEL_CONTENT,
                encoding="utf-8",
            )
            (target / "old.txt").write_text("old", encoding="utf-8")
            ready = parent / "ready"
            ready.mkdir()
            (ready / frontier_scan.OUTPUT_SENTINEL_NAME).write_text(
                frontier_scan.OUTPUT_SENTINEL_CONTENT,
                encoding="utf-8",
            )
            (ready / "new.txt").write_text("new", encoding="utf-8")

            with (
                patch.object(frontier_scan.shutil, "rmtree", side_effect=OSError("locked")),
                self.assertWarnsRegex(UserWarning, "published plan"),
            ):
                frontier_scan._commit(ready, target, overwrite=True)

            self.assertTrue((target / "new.txt").is_file())
            self.assertFalse((target / "old.txt").exists())

    def test_starter_passes_unknown_bounded_scanner_options(self):
        parsed, remaining = start_frontier_colab.build_parser().parse_known_args(
            ["--output", "result", "--workers", "3"]
        )
        self.assertEqual(parsed.output_dir, Path("result"))
        self.assertEqual(remaining, ["--workers", "3"])

    def test_starter_reports_archive_os_error_without_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "frontier"
            with (
                patch.object(start_frontier_colab.subprocess, "run", return_value=None),
                patch.object(start_frontier_colab, "_archive", side_effect=OSError("disk full")),
            ):
                status = start_frontier_colab.main(["--output-dir", str(output)])
            self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
