from __future__ import annotations

import ast
import importlib.util
import math
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_colab.py"
STARTER = ROOT / "start_colab.py"


def _load_runner_without_ml_dependencies():
    """Import the orchestration helpers with a mocked engine, never PyTorch."""
    fake_devorar = types.ModuleType("src.devorar")

    class FakeAssimilationError(RuntimeError):
        pass

    fake_devorar.AssimilationError = FakeAssimilationError
    fake_src = types.ModuleType("src")
    fake_src.devorar = fake_devorar
    spec = importlib.util.spec_from_file_location("_devorar_run_colab_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_src = sys.modules.get("src")
    original_devorar = sys.modules.get("src.devorar")
    try:
        sys.modules["src"] = fake_src
        sys.modules["src.devorar"] = fake_devorar
        spec.loader.exec_module(module)
    finally:
        if original_src is None:
            sys.modules.pop("src", None)
        else:
            sys.modules["src"] = original_src
        if original_devorar is None:
            sys.modules.pop("src.devorar", None)
        else:
            sys.modules["src.devorar"] = original_devorar
    return module


run_colab = _load_runner_without_ml_dependencies()


def _load_starter():
    spec = importlib.util.spec_from_file_location("_devorar_start_colab_test", STARTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


start_colab = _load_starter()


class OneFileLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = STARTER.read_text(encoding="utf-8")
        ast.parse(cls.source)

    def test_launcher_installs_runs_and_archives_automatically(self) -> None:
        for fragment in (
            "requirements-colab.txt",
            "run_colab.py",
            "pip",
            "--overwrite-output",
            "shutil.make_archive",
            "if __name__ == \"__main__\"",
        ):
            self.assertIn(fragment, self.source)

    def test_launcher_has_bounded_colab_defaults(self) -> None:
        self.assertIn('"devorar-output"', self.source)
        self.assertIn('"huggingface-cache"', self.source)
        self.assertEqual(
            start_colab._resolve_runtime_path(Path("relative-output")),
            (start_colab.REPOSITORY_ROOT / "relative-output").resolve(),
        )
        alias = start_colab.build_parser().parse_args(["--output", "alias-output"])
        self.assertEqual(alias.output_dir, Path("alias-output"))

    def test_archive_refuses_to_replace_foreign_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            output.mkdir()
            (output / ".devorar-output-v1").write_text(
                "DragonBRX/Devorar managed output v1\n",
                encoding="utf-8",
            )
            Path(f"{output}.zip").write_text("not a Devorar ZIP", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not owned by Devorar"):
                start_colab._create_archive(output)

    def test_one_file_launcher_invokes_runner_and_creates_zip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result"
            cache = root / "cache"
            calls: list[list[str]] = []

            def fake_run(command, **kwargs):
                calls.append(list(command))
                output.mkdir()
                (output / ".devorar-output-v1").write_text(
                    "DragonBRX/Devorar managed output v1\n",
                    encoding="utf-8",
                )
                (output / "run-summary.json").write_text('{"status":"ok"}\n', encoding="utf-8")
                return SimpleNamespace(returncode=0)

            with patch.object(start_colab.subprocess, "run", side_effect=fake_run):
                code = start_colab.main(
                    [
                        "--no-install",
                        "--output-dir",
                        str(output),
                        "--cache-dir",
                        str(cache),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 1)
            self.assertIn("--overwrite-output", calls[0])
            archive = Path(f"{output}.zip")
            self.assertTrue(archive.is_file())
            with zipfile.ZipFile(archive) as bundle:
                self.assertIn("run-summary.json", bundle.namelist())


class RunnerStaticSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_runner_has_immutable_model_revisions(self) -> None:
        self.assertEqual(run_colab.BASE_MODEL, "HuggingFaceTB/SmolLM2-360M")
        self.assertEqual(
            run_colab.BASE_REVISION,
            "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        )
        self.assertEqual(run_colab.DONOR_MODEL, "HuggingFaceTB/SmolLM2-360M-Instruct")
        self.assertEqual(
            run_colab.DONOR_REVISION,
            "a10cc1512eabd3dde888204e902eca88bddb4951",
        )
        self.assertEqual(run_colab.SOURCE_LICENSE, "Apache-2.0")

    def test_runner_imports_engine_and_uses_direct_assimilation(self) -> None:
        imported = False
        assimilates = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom) and node.module == "src":
                imported |= any(alias.name == "devorar" for alias in node.names)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assimilates |= (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "devorar"
                    and node.func.attr == "assimilate_models"
                )
        self.assertTrue(imported)
        self.assertTrue(assimilates)

    def test_runner_never_invokes_donor_as_a_model(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotEqual(node.func.id, "donor_model")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "donor_model":
                    self.assertNotIn(node.func.attr, {"forward", "generate", "__call__"})

    def test_sequence_baseline_then_build_then_release_then_candidate(self) -> None:
        baseline_at = self.source.index("baseline = evaluate_model(")
        donor_load_at = self.source.index("donor_model = load_causal_lm(")
        build_at = self.source.index("assimilation = devorar.assimilate_models(")
        release_at = self.source.index("del donor_model")
        candidate_at = self.source.index("candidate = evaluate_model(")
        self.assertLess(baseline_at, donor_load_at)
        self.assertLess(donor_load_at, build_at)
        self.assertLess(build_at, release_at)
        self.assertLess(release_at, candidate_at)

    def test_weight_allowlist_rejects_executable_and_pickle_formats(self) -> None:
        self.assertIn("*.safetensors", run_colab.SNAPSHOT_ALLOW_PATTERNS)
        for pattern in ("*.bin", "*.pt", "*.pth", "*.ckpt", "*.py"):
            self.assertIn(pattern, run_colab.SNAPSHOT_DENY_PATTERNS)

    def test_default_limits_are_bounded(self) -> None:
        args = run_colab.build_parser().parse_args([])
        self.assertLessEqual(args.max_prompts, 4)
        self.assertLessEqual(args.max_input_tokens, 256)
        self.assertLessEqual(args.max_new_tokens, 64)
        self.assertLessEqual(args.perplexity_tokens, 1024)
        self.assertLessEqual(args.max_process_ram_gib, 10.0)
        alias = run_colab.build_parser().parse_args(["--output", "alias-output"])
        documented = run_colab.build_parser().parse_args(["--output-dir", "documented-output"])
        self.assertEqual(alias.output_dir, Path("alias-output"))
        self.assertEqual(documented.output_dir, Path("documented-output"))


class RunnerMockTests(unittest.TestCase):
    def test_overwrite_rejects_unowned_directory_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "existing"
            target.mkdir()
            args = SimpleNamespace(
                output_dir=target,
                cache_dir=Path(temporary) / "cache",
                overwrite_output=True,
            )
            with patch.object(run_colab, "_run_staged") as staged:
                with self.assertRaisesRegex(RuntimeError, "not owned by Devorar"):
                    run_colab.run(args)
                staged.assert_not_called()

    def test_managed_output_is_replaced_only_after_staged_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "managed"
            target.mkdir()
            (target / run_colab.OUTPUT_SENTINEL_NAME).write_text(
                run_colab.OUTPUT_SENTINEL_CONTENT,
                encoding="utf-8",
            )
            (target / "old.txt").write_text("old", encoding="utf-8")
            args = SimpleNamespace(
                output_dir=target,
                cache_dir=Path(temporary) / "cache",
                overwrite_output=True,
            )

            def fake_run(staged_args, *, final_output_dir):
                self.assertEqual(final_output_dir, target.resolve())
                (staged_args.output_dir / run_colab.OUTPUT_SENTINEL_NAME).write_text(
                    run_colab.OUTPUT_SENTINEL_CONTENT,
                    encoding="utf-8",
                )
                (staged_args.output_dir / "new.txt").write_text("new", encoding="utf-8")
                return {"status": "ok"}

            with patch.object(run_colab, "_run_staged", side_effect=fake_run):
                result = run_colab.run(args)

            self.assertEqual(result, {"status": "ok"})
            self.assertFalse((target / "old.txt").exists())
            self.assertEqual((target / "new.txt").read_text(encoding="utf-8"), "new")

    def test_failed_staging_preserves_previous_managed_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "managed"
            target.mkdir()
            (target / run_colab.OUTPUT_SENTINEL_NAME).write_text(
                run_colab.OUTPUT_SENTINEL_CONTENT,
                encoding="utf-8",
            )
            (target / "old.txt").write_text("old", encoding="utf-8")
            args = SimpleNamespace(
                output_dir=target,
                cache_dir=Path(temporary) / "cache",
                overwrite_output=True,
            )

            def fail_run(staged_args, *, final_output_dir):
                (staged_args.output_dir / "partial.txt").write_text("partial", encoding="utf-8")
                raise RuntimeError("simulated failure")

            with patch.object(run_colab, "_run_staged", side_effect=fail_run):
                with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                    run_colab.run(args)

            self.assertEqual((target / "old.txt").read_text(encoding="utf-8"), "old")
            self.assertFalse((target / "partial.txt").exists())

    def test_broad_output_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "broad output directory"):
            run_colab._validate_output_target(Path.cwd())

    def test_download_uses_revision_and_safe_patterns_without_network(self) -> None:
        calls: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"
            snapshot.mkdir()
            (snapshot / "model.safetensors").write_bytes(b"safe-test-placeholder")

            def fake_snapshot_download(**kwargs: object) -> str:
                calls.append(kwargs)
                return str(snapshot)

            result = run_colab.download_snapshot(
                run_colab.BASE_MODEL,
                run_colab.BASE_REVISION,
                cache_dir=Path(temporary) / "cache",
                snapshot_download_func=fake_snapshot_download,
            )

        self.assertEqual(result, snapshot)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["repo_id"], run_colab.BASE_MODEL)
        self.assertEqual(calls[0]["revision"], run_colab.BASE_REVISION)
        self.assertIn("*.safetensors", calls[0]["allow_patterns"])
        self.assertIn("*.bin", calls[0]["ignore_patterns"])

    def test_loader_disables_remote_code_and_forces_safetensors(self) -> None:
        calls: dict[str, list[tuple[str, dict[str, object]]]] = {
            "config": [],
            "tokenizer": [],
            "model": [],
        }
        config = object()

        class FakeTokenizer:
            pad_token_id = None
            pad_token = None
            eos_token = "<eos>"

        tokenizer = FakeTokenizer()

        class FakeModel:
            def eval(self) -> None:
                calls["model"].append(("eval", {}))

        class ConfigFactory:
            @staticmethod
            def from_pretrained(path: str, **kwargs: object) -> object:
                calls["config"].append((path, kwargs))
                return config

        class TokenizerFactory:
            @staticmethod
            def from_pretrained(path: str, **kwargs: object) -> FakeTokenizer:
                calls["tokenizer"].append((path, kwargs))
                return tokenizer

        class ModelFactory:
            @staticmethod
            def from_pretrained(path: str, **kwargs: object) -> FakeModel:
                calls["model"].append((path, kwargs))
                return FakeModel()

        fake_transformers = SimpleNamespace(
            AutoConfig=ConfigFactory,
            AutoTokenizer=TokenizerFactory,
            AutoModelForCausalLM=ModelFactory,
        )
        model, loaded_tokenizer, loaded_config = run_colab.load_model_assets(
            Path("/immutable/snapshot"),
            dtype="float16-test",
            transformers_module=fake_transformers,
        )

        self.assertIsInstance(model, FakeModel)
        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertIs(loaded_config, config)
        for group in ("config", "tokenizer"):
            kwargs = calls[group][0][1]
            self.assertFalse(kwargs["trust_remote_code"])
            self.assertTrue(kwargs["local_files_only"])
        model_kwargs = calls["model"][0][1]
        self.assertFalse(model_kwargs["trust_remote_code"])
        self.assertTrue(model_kwargs["local_files_only"])
        self.assertTrue(model_kwargs["use_safetensors"])
        self.assertEqual(model_kwargs["torch_dtype"], "float16-test")
        self.assertEqual(tokenizer.pad_token, "<eos>")

    def test_deterministic_rules_and_retention_gate_are_pure(self) -> None:
        self.assertTrue(
            run_colab.evaluate_rule(
                "The answer is Paris.",
                {"rule": "contains_word", "expected": "PARIS"},
            )
        )
        self.assertFalse(
            run_colab.evaluate_rule(
                "fifty",
                {"rule": "contains_word", "expected": "5"},
            )
        )
        baseline = {
            "retention_corpus": {
                "negative_log_likelihood": math.log(10.0),
                "perplexity": 10.0,
            },
            "deterministic_rules": {"score": 0.75},
        }
        candidate = {
            "retention_corpus": {
                "negative_log_likelihood": math.log(12.0),
                "perplexity": 12.0,
            },
            "deterministic_rules": {"score": 0.75},
        }
        comparison = run_colab.compare_evaluations(
            baseline,
            candidate,
            max_perplexity_ratio=1.5,
            max_rule_score_drop=0.25,
        )
        self.assertTrue(comparison["retention_passed"])
        self.assertTrue(comparison["rules_passed"])
        self.assertFalse(comparison["behavioral_transfer_observed"])
        self.assertTrue(comparison["quality_gates_passed"])
        self.assertFalse(comparison["all_gates_passed"])
        self.assertEqual(comparison["promotion"], "none_experimental_only")

        extreme = {
            "retention_corpus": {
                "negative_log_likelihood": 1_000.0,
                "perplexity": None,
            },
            "deterministic_rules": {"score": 1.0},
        }
        extreme_comparison = run_colab.compare_evaluations(
            baseline,
            extreme,
            max_perplexity_ratio=1.5,
            max_rule_score_drop=0.25,
        )
        self.assertFalse(extreme_comparison["retention_passed"])
        self.assertIsNone(extreme_comparison["perplexity_ratio_candidate_over_host"])

    def test_chat_template_is_applied_without_model_output(self) -> None:
        calls: list[tuple[object, bool, bool]] = []

        class ChatTokenizer:
            chat_template = "configured"

            @staticmethod
            def apply_chat_template(messages, *, tokenize, add_generation_prompt):
                calls.append((messages, tokenize, add_generation_prompt))
                return "<user>ping</user><assistant>"

        rendered = run_colab.render_prompt(ChatTokenizer(), "ping")
        self.assertEqual(rendered, "<user>ping</user><assistant>")
        self.assertEqual(
            calls,
            [([{"role": "user", "content": "ping"}], False, True)],
        )

    def test_build_audit_must_explicitly_prove_zero_donor_calls(self) -> None:
        accepted = SimpleNamespace(audit=SimpleNamespace(donor_forward_calls_build=0))
        rejected = SimpleNamespace(audit=SimpleNamespace(donor_forward_calls_build=1))
        missing = SimpleNamespace(audit=SimpleNamespace())
        self.assertEqual(run_colab.audited_donor_forward_calls(accepted), 0)
        with self.assertRaises(RuntimeError):
            run_colab.audited_donor_forward_calls(rejected)
        with self.assertRaises(RuntimeError):
            run_colab.audited_donor_forward_calls(missing)


if __name__ == "__main__":
    unittest.main()
