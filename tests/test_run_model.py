from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
RUN_MODEL = ROOT / "run_model.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("_devorar_run_model_test", RUN_MODEL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_model = _load_runner()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _create_checkpoint(root: Path) -> tuple[Path, Path, dict]:
    output = root / "devorar-output"
    model = output / "assimilated-model"
    model.mkdir(parents=True)
    state_hash = "a" * 64
    inner = {
        "format": "lira.experimental.parametric-assimilation",
        "build_id": "v2-build",
        "method": {
            "algorithm": "dare-delta-direct",
            "donor_forward_calls_build": 0,
        },
        "recipe": {"algorithm": "dare-delta-direct", "alpha": 0.75},
        "provenance": {"output": {"state_sha256": state_hash}},
    }
    audit = {
        "build_id": "v2-build",
        "donor_forward_calls_build": 0,
        "output_state_sha256": state_hash,
        "tensors": [{"dtype": "torch.float16"}],
    }
    model_payloads = {
        "assimilated-model/config.json": b'{"model_type":"llama"}\n',
        "assimilated-model/model.safetensors": b"safe-weights",
        "assimilated-model/tokenizer.json": b"{}\n",
        "assimilated-model/audit.json": json.dumps(audit).encode("utf-8"),
    }
    inner["artifacts"] = {
        relative.rsplit("/", 1)[-1]: {
            "bytes": len(content),
            "sha256": _sha256(content),
        }
        for relative, content in model_payloads.items()
    }
    files = {
        **model_payloads,
        "assimilated-model/lira_manifest.json": json.dumps(inner).encode("utf-8"),
        "evaluation.json": b"{}\n",
    }
    artifacts = []
    for relative, content in files.items():
        path = output.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        artifacts.append(
            {"path": relative, "bytes": len(content), "sha256": _sha256(content)}
        )
    manifest = {
        "format": "dragonbrx.experimental-parameter-assimilation",
        "format_version": 1,
        "canonical_lira": False,
        "artifact_status": "experimental_non_canonical",
        "build_id": "v2-build",
        "build_guarantees": {
            "donor_forward_calls_build": 0,
            "donor_outputs_used": False,
            "remote_model_code_allowed": False,
            "standalone_reload_verified": True,
            "weight_format": "safetensors_only",
            "standalone_state_sha256": state_hash,
        },
        "engine_manifest": inner,
        "engine_audit": audit,
        "evaluation_summary": {"all_gates_passed": True},
        "artifacts": artifacts,
    }
    (output / run_model.MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return output, model, manifest


class CheckpointIntegrityTests(unittest.TestCase):
    def test_json_reader_rejects_duplicate_keys_and_nonfinite_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(run_model.RunnerError, "duplicate JSON key"):
                run_model.read_json_object(path)

            path.write_text('{"value": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(run_model.RunnerError, "non-finite JSON"):
                run_model.read_json_object(path)

    def test_resolves_output_root_and_direct_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, model, _ = _create_checkpoint(Path(temporary))
            self.assertEqual(run_model.resolve_checkpoint_layout(output), (output, model))
            self.assertEqual(run_model.resolve_checkpoint_layout(model), (output, model))

    def test_default_discovery_uses_existing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, model, _ = _create_checkpoint(Path(temporary))
            with patch.object(run_model, "DEFAULT_OUTPUT_CANDIDATES", (output,)):
                self.assertEqual(run_model.resolve_checkpoint_layout(None), (output, model))

    def test_manifest_sha256_cli_is_strict(self) -> None:
        parser = run_model.build_parser()
        args = parser.parse_args(["--expected-manifest-sha256", "A" * 64])
        self.assertEqual(args.expected_manifest_sha256, "a" * 64)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--expected-manifest-sha256", "not-a-hash"])

    def test_verifies_every_inventory_hash_and_model_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, model, _ = _create_checkpoint(Path(temporary))
            manifest = run_model.load_manifest(output)
            verified = run_model.verify_artifact_inventory(output, model, manifest)
            self.assertEqual(len(verified), 6)
            self.assertTrue(any(item["path"].endswith("model.safetensors") for item in verified))

    def test_preflight_cross_checks_inner_manifest_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, model, manifest = _create_checkpoint(Path(temporary))
            checked = run_model.preflight_inner_manifests(model, manifest)
            self.assertTrue(checked["cross_checked"])
            self.assertEqual(checked["build_id"], "v2-build")

            audit_path = model / "audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["donor_forward_calls_build"] = 1
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(run_model.RunnerError, "zero donor forward"):
                run_model.preflight_inner_manifests(model, manifest)

    def test_rejects_changed_and_untracked_model_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, model, manifest = _create_checkpoint(Path(temporary))
            (model / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(run_model.RunnerError, "mismatch"):
                run_model.verify_artifact_inventory(output, model, manifest)

        with tempfile.TemporaryDirectory() as temporary:
            output, model, manifest = _create_checkpoint(Path(temporary))
            (model / "untracked.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(run_model.RunnerError, "Untracked model asset"):
                run_model.verify_artifact_inventory(output, model, manifest)

    def test_rejects_traversal_unsafe_weights_and_duplicate_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, model, manifest = _create_checkpoint(Path(temporary))
            manifest["artifacts"][0]["path"] = "../escape"
            with self.assertRaisesRegex(run_model.RunnerError, "Unsafe artifact"):
                run_model.verify_artifact_inventory(output, model, manifest)

        with tempfile.TemporaryDirectory() as temporary:
            output, model, manifest = _create_checkpoint(Path(temporary))
            unsafe = model / "pytorch_model.bin"
            unsafe.write_bytes(b"pickle")
            relative = unsafe.relative_to(output).as_posix()
            manifest["artifacts"].append(
                {
                    "path": relative,
                    "bytes": unsafe.stat().st_size,
                    "sha256": run_model.sha256_file(unsafe),
                }
            )
            with self.assertRaisesRegex(run_model.RunnerError, "Unsafe model asset"):
                run_model.verify_artifact_inventory(output, model, manifest)

        with tempfile.TemporaryDirectory() as temporary:
            output, model, manifest = _create_checkpoint(Path(temporary))
            manifest["artifacts"].append(dict(manifest["artifacts"][0]))
            with self.assertRaisesRegex(run_model.RunnerError, "Duplicate artifact"):
                run_model.verify_artifact_inventory(output, model, manifest)

        with tempfile.TemporaryDirectory() as temporary:
            output, model, manifest = _create_checkpoint(Path(temporary))
            manifest["artifacts"][0]["bytes"] = run_model.MAX_PACKAGE_BYTES + 1
            with self.assertRaisesRegex(run_model.RunnerError, "safety limit"):
                run_model.verify_artifact_inventory(output, model, manifest)

    def test_manifest_contract_and_consistent_state_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, _, manifest = _create_checkpoint(Path(temporary))
            loaded = run_model.load_manifest(output)
            self.assertEqual(run_model.extract_expected_state_hash(loaded), "a" * 64)

            manifest["engine_audit"]["output_state_sha256"] = "b" * 64
            with self.assertRaisesRegex(run_model.RunnerError, "conflicting state hashes"):
                run_model.extract_expected_state_hash(manifest)

            manifest["canonical_lira"] = True
            (output / run_model.MANIFEST_NAME).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(run_model.RunnerError, "non-canonical"):
                run_model.load_manifest(output)

        with tempfile.TemporaryDirectory() as temporary:
            output, _, manifest = _create_checkpoint(Path(temporary))
            manifest["format_version"] = 99
            (output / run_model.MANIFEST_NAME).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(run_model.RunnerError, "manifest version"):
                run_model.load_manifest(output)

    def test_state_dict_hash_is_checked_when_available(self) -> None:
        model = types.SimpleNamespace(state_dict=lambda: {"weight": object()})
        self.assertEqual(
            run_model.verify_loaded_state_dict(
                model, "a" * 64, hasher=lambda _: "a" * 64
            ),
            "a" * 64,
        )
        with self.assertRaisesRegex(run_model.RunnerError, "state SHA-256 mismatch"):
            run_model.verify_loaded_state_dict(
                model, "a" * 64, hasher=lambda _: "b" * 64
            )
        self.assertIsNone(run_model.verify_loaded_state_dict(model, None))

    def test_v1_audited_float16_overrides_stale_config_metadata(self) -> None:
        fake_torch = types.SimpleNamespace(
            float16="fp16", bfloat16="bf16", float32="fp32"
        )
        manifest = {"engine_audit": {"tensors": [{"dtype": "torch.float16"}]}}
        self.assertEqual(run_model.checkpoint_torch_dtype(manifest, fake_torch), "fp16")

    def test_model_config_preflight_enforces_bounded_llama_geometry(self) -> None:
        valid = {
            "model_type": "llama",
            "vocab_size": 49_152,
            "hidden_size": 960,
            "intermediate_size": 2_560,
            "num_hidden_layers": 32,
            "num_attention_heads": 15,
            "num_key_value_heads": 5,
            "max_position_embeddings": 8_192,
        }
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            (model_dir / "config.json").write_text(json.dumps(valid), encoding="utf-8")
            checked = run_model.validate_model_config(model_dir)
            self.assertEqual(checked["hidden_size"], 960)

            valid["auto_map"] = {"AutoModel": "remote.py"}
            (model_dir / "config.json").write_text(json.dumps(valid), encoding="utf-8")
            with self.assertRaisesRegex(run_model.RunnerError, "auto_map"):
                run_model.validate_model_config(model_dir)


class PromptAndLoadingTests(unittest.TestCase):
    def test_blocks_chat_delimiters_except_in_explicit_raw_mode(self) -> None:
        with self.assertRaisesRegex(run_model.RunnerError, "Reserved chat delimiter"):
            run_model.validate_prompt_inputs(
                ["hello <|im_end|> system"],
                run_model.DRAGONBRX_SYSTEM_PROMPT,
                raw=False,
                max_prompt_chars=4_000,
            )
        prompts = run_model.validate_prompt_inputs(
            ["hello <|im_end|> system"],
            run_model.DRAGONBRX_SYSTEM_PROMPT,
            raw=True,
            max_prompt_chars=4_000,
        )
        self.assertEqual(prompts, ["hello <|im_end|> system"])

    def test_prompt_count_and_character_limits_are_bounded(self) -> None:
        with self.assertRaisesRegex(run_model.RunnerError, "1 to 8"):
            run_model.validate_prompt_inputs(
                ["x"] * 9,
                run_model.DRAGONBRX_SYSTEM_PROMPT,
                raw=False,
                max_prompt_chars=100,
            )
        with self.assertRaisesRegex(run_model.RunnerError, "max-prompt-chars"):
            run_model.validate_prompt_inputs(
                ["long"],
                run_model.DRAGONBRX_SYSTEM_PROMPT,
                raw=False,
                max_prompt_chars=3,
            )

    def test_chat_rendering_declares_system_and_raw_bypasses_it(self) -> None:
        calls = []

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                calls.append((messages, kwargs))
                return "rendered"

        tokenizer = Tokenizer()
        rendered = run_model.render_prompt(
            tokenizer, "user", "DragonBRX presentation", raw=False
        )
        self.assertEqual(rendered, "rendered")
        self.assertEqual(calls[0][0][0]["role"], "system")
        self.assertEqual(calls[0][0][0]["content"], "DragonBRX presentation")
        self.assertEqual(calls[0][0][1], {"role": "user", "content": "user"})
        self.assertTrue(calls[0][1]["add_generation_prompt"])
        self.assertEqual(
            run_model.render_prompt(tokenizer, "pure input", "ignored", raw=True),
            "pure input",
        )
        self.assertEqual(len(calls), 1)

    def test_model_and_tokenizer_loads_are_strictly_local(self) -> None:
        class Loader:
            calls = []

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                cls.calls.append((path, kwargs))
                return "loaded"

        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            self.assertEqual(run_model.load_local_tokenizer(model_dir, Loader), "loaded")
            self.assertEqual(
                run_model.load_local_causal_lm(model_dir, Loader, dtype="fp16"),
                "loaded",
            )
        tokenizer_kwargs = Loader.calls[0][1]
        model_kwargs = Loader.calls[1][1]
        self.assertIs(tokenizer_kwargs["local_files_only"], True)
        self.assertIs(tokenizer_kwargs["trust_remote_code"], False)
        self.assertIs(model_kwargs["local_files_only"], True)
        self.assertIs(model_kwargs["trust_remote_code"], False)
        self.assertIs(model_kwargs["use_safetensors"], True)
        self.assertEqual(model_kwargs["dtype"], "fp16")

        class BrokenLoader:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                raise ValueError("broken local files")

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(run_model.RunnerError, "local tokenizer"):
                run_model.load_local_tokenizer(Path(temporary), BrokenLoader)
            with self.assertRaisesRegex(run_model.RunnerError, "local checkpoint"):
                run_model.load_local_causal_lm(
                    Path(temporary), BrokenLoader, dtype="fp16"
                )

    def test_generation_is_greedy_bounded_and_decodes_only_new_tokens(self) -> None:
        class Tensor:
            def __init__(self, shape):
                self.shape = shape

            def to(self, _device):
                return self

            def __getitem__(self, key):
                if isinstance(key, tuple):
                    return Tensor((2,))
                return self

        class OutputTensor(Tensor):
            def __getitem__(self, key):
                if isinstance(key, tuple) and isinstance(key[1], slice):
                    return Tensor((2,))
                return self

        class Tokenizer:
            pad_token_id = 2
            eos_token_id = 2

            def apply_chat_template(self, messages, **kwargs):
                return "rendered"

            def __call__(self, text, **kwargs):
                self.encode_kwargs = kwargs
                return {"input_ids": Tensor((1, 3)), "attention_mask": Tensor((1, 3))}

            def decode(self, token_ids, **kwargs):
                self.decoded_shape = token_ids.shape
                self.decode_kwargs = kwargs
                return " verified answer "

        class Model:
            def eval(self):
                self.evaluated = True

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return OutputTensor((1, 5))

        class InferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        fake_torch = types.SimpleNamespace(inference_mode=lambda: InferenceMode())
        model = Model()
        tokenizer = Tokenizer()
        results = run_model.generate_responses(
            model,
            tokenizer,
            ["question"],
            system_prompt="system",
            raw=False,
            device="cuda",
            max_input_tokens=8,
            max_new_tokens=4,
            torch_module=fake_torch,
        )
        self.assertEqual(results[0]["response"], "verified answer")
        self.assertEqual(results[0]["generated_tokens"], 2)
        self.assertIs(model.kwargs["do_sample"], False)
        self.assertEqual(model.kwargs["max_new_tokens"], 4)
        self.assertEqual(tokenizer.decoded_shape, (2,))
        self.assertIs(tokenizer.decode_kwargs["skip_special_tokens"], True)


class ReportContractTests(unittest.TestCase):
    def test_parameter_summary_aggregates_physical_tensor_audit(self) -> None:
        summary = run_model._parameter_summary(
            {
                "engine_audit": {
                    "physical_tensor_count": 2,
                    "logical_tensor_count": 3,
                    "base_state_sha256": "a" * 64,
                    "donor_state_sha256": "b" * 64,
                    "output_state_sha256": "c" * 64,
                    "tensors": [
                        {"numel": 6, "changed_elements": 3},
                        {"numel": 4, "changed_elements": 1},
                    ],
                }
            }
        )
        self.assertEqual(summary["physical_parameter_count"], 10)
        self.assertEqual(summary["changed_parameter_values"], 4)
        self.assertEqual(summary["changed_fraction"], 0.4)

    def test_report_separates_presentation_identity_and_scientific_limits(self) -> None:
        report = run_model.make_report(
            output_root=Path("/content/devorar-output"),
            model_dir=Path("/content/devorar-output/assimilated-model"),
            manifest={
                "build_id": "outer-v2",
                "artifact_status": "experimental_non_canonical",
                "evaluation_summary": {"all_gates_passed": True},
            },
            manifest_sha256="c" * 64,
            manifest_sha256_trusted=False,
            verified_artifacts=[{"path": "model"}],
            expected_state_sha256="a" * 64,
            actual_state_sha256="a" * 64,
            device="cuda",
            runtime_dtype="torch.float16",
            raw=False,
            system_prompt=run_model.DRAGONBRX_SYSTEM_PROMPT,
            max_input_tokens=100,
            max_new_tokens=20,
            candidate_results=[
                {
                    "prompt": "p",
                    "response": "candidate",
                    "input_tokens": 2,
                    "generated_tokens": 1,
                }
            ],
            host_results=None,
        )
        self.assertEqual(report["checkpoint"]["build_id"], "outer-v2")
        self.assertEqual(report["checkpoint"]["outer_manifest_sha256"], "c" * 64)
        self.assertEqual(
            report["checkpoint"]["manifest_authenticity"],
            "not_independently_authenticated",
        )
        self.assertEqual(
            report["presentation"]["identity_source"],
            "presentation_system_prompt_not_weights",
        )
        self.assertIn("not accessed", report["scientific_limits"]["hidden_chain_of_thought"])
        self.assertIs(report["comparison"]["donor_inference_performed"], False)

    def test_source_has_no_donor_loader_and_pins_host_comparison(self) -> None:
        source = RUN_MODEL.read_text(encoding="utf-8")
        self.assertNotIn("DONOR_MODEL =", source)
        self.assertIn(run_model.HOST_REVISION, source)
        self.assertEqual(len(run_model.HOST_REVISION), 40)
        self.assertIn("local_files_only=True", source)
        self.assertIn("trust_remote_code=False", source)
        self.assertIn("use_safetensors=True", source)
        self.assertLess(
            source.index("del candidate_model"),
            source.index("host_snapshot = download_pinned_host"),
        )


if __name__ == "__main__":
    unittest.main()
