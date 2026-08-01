#!/usr/bin/env python3
"""Run and verify a locally built Devorar checkpoint.

This is intentionally separate from the assimilation builder.  It validates
the build manifest and its artifact inventory before loading any model, then
checks the logical state-dict hash when the build recorded one.  The optional
host comparison downloads only the immutable, pinned host snapshot and never
loads or runs the donor model.

The default DragonBRX identity is a declared presentation-system prompt.  It
is not represented as evidence that the identity was learned in the weights.
Use ``--raw`` to bypass that presentation layer and test the checkpoint alone.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


MANIFEST_NAME = "assimilation-manifest.lira.json"
MODEL_DIRECTORY_NAME = "assimilated-model"
HOST_MODEL = "HuggingFaceTB/SmolLM2-360M"
HOST_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CANDIDATES = (
    Path("/content/devorar-output"),
    REPOSITORY_ROOT / "devorar-output",
    Path.cwd() / "devorar-output",
)

DRAGONBRX_SYSTEM_PROMPT = (
    "You are DragonBRX Assimilated, an experimental AI model produced by "
    "direct parameter-space assimilation. Give clear final answers and concise, "
    "verifiable explanations. Do not claim access to or reveal hidden "
    "chain-of-thought."
)
DEMO_PROMPTS = (
    "Responda em uma frase: o que torna um experimento verificável?",
)
BLOCKED_CHAT_DELIMITERS = ("<|im_start|>", "<|im_end|>")
MAX_PROMPTS = 8
DEFAULT_MAX_PROMPT_CHARS = 4_000
HASH_CHUNK_BYTES = 4 * 1024 * 1024
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_ARTIFACT_COUNT = 128
MAX_PACKAGE_BYTES = 4 * 1024 * 1024 * 1024
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
UNSAFE_MODEL_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".h5",
    ".msgpack",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".tflite",
}

HOST_ALLOW_PATTERNS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "LICENSE*",
)
HOST_DENY_PATTERNS = (
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "*.tflite",
    "*.py",
)


class RunnerError(RuntimeError):
    """Raised when a checkpoint cannot be verified or safely executed."""


def bounded_int(minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        parsed = int(value)
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"expected an integer from {minimum} to {maximum}"
            )
        return parsed

    return parse


def sha256_argument(value: str) -> str:
    normalized = value.strip().lower()
    if not HEX_SHA256.fullmatch(normalized):
        raise argparse.ArgumentTypeError("expected exactly 64 hexadecimal SHA-256 characters")
    return normalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        if path.is_symlink():
            raise RunnerError(f"Symbolic-link JSON file rejected: {path}")
        size = path.stat().st_size
        if size > MAX_JSON_BYTES:
            raise RunnerError(f"JSON file exceeds {MAX_JSON_BYTES} bytes: {path}")
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except FileNotFoundError as exc:
        raise RunnerError(f"Required file not found: {path}") from exc
    except RunnerError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RunnerError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunnerError(f"Expected a JSON object in {path}")
    return payload


def encode_json(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RunnerError(f"Inference report is not strict JSON: {exc}") from exc


def write_json_atomic(path: Path, encoded: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_artifact_path(output_root: Path, path_text: Any) -> Path:
    if not isinstance(path_text, str) or not path_text or "\\" in path_text:
        raise RunnerError(f"Invalid artifact path in manifest: {path_text!r}")
    relative = PurePosixPath(path_text)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise RunnerError(f"Unsafe artifact path in manifest: {path_text!r}")
    unresolved = output_root.joinpath(*relative.parts)
    current = output_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RunnerError(f"Symbolic-link artifact rejected: {path_text!r}")
    resolved = unresolved.resolve()
    if not _is_inside(resolved, output_root):
        raise RunnerError(f"Artifact escapes the output directory: {path_text!r}")
    return resolved


def resolve_checkpoint_layout(output_dir: Path | None) -> tuple[Path, Path]:
    """Return ``(output_root, model_directory)`` from a root or model path."""

    if output_dir is None:
        candidates: list[Path] = []
        seen: set[Path] = set()
        for candidate in DEFAULT_OUTPUT_CANDIDATES:
            expanded = candidate.expanduser().absolute()
            if expanded.is_symlink():
                continue
            resolved = expanded.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)
        selected = next(
            (candidate for candidate in candidates if (candidate / MANIFEST_NAME).is_file()),
            None,
        )
        if selected is None:
            searched = ", ".join(str(path) for path in candidates)
            raise RunnerError(
                "Devorar output was not found. Pass --output-dir. "
                f"Searched: {searched}"
            )
        output_root = selected
        model_dir = output_root / MODEL_DIRECTORY_NAME
    else:
        expanded = output_dir.expanduser().absolute()
        if expanded.is_symlink():
            raise RunnerError(f"Symbolic-link checkpoint path rejected: {expanded}")
        selected = expanded.resolve()
        if (selected / MANIFEST_NAME).is_file():
            output_root = selected
            model_dir = output_root / MODEL_DIRECTORY_NAME
        elif (selected / "config.json").is_file() and (
            selected.parent / MANIFEST_NAME
        ).is_file():
            output_root = selected.parent
            model_dir = selected
        else:
            raise RunnerError(
                f"{selected} is neither a Devorar output directory nor its model directory"
            )

    if model_dir.is_symlink():
        raise RunnerError(f"Symbolic-link model directory rejected: {model_dir}")
    if not model_dir.is_dir():
        raise RunnerError(f"Assimilated model directory not found: {model_dir}")
    return output_root.resolve(), model_dir.resolve()


def load_manifest(output_root: Path) -> dict[str, Any]:
    manifest = read_json_object(output_root / MANIFEST_NAME)
    format_name = manifest.get("format")
    if format_name != "dragonbrx.experimental-parameter-assimilation":
        raise RunnerError(f"Unsupported Devorar manifest format: {format_name!r}")
    format_version = manifest.get("format_version")
    if isinstance(format_version, bool) or format_version not in (1, 2):
        raise RunnerError(f"Unsupported Devorar manifest version: {format_version!r}")
    if manifest.get("canonical_lira") is not False:
        raise RunnerError("The runner requires an explicitly non-canonical experimental build")
    guarantees = manifest.get("build_guarantees")
    if not isinstance(guarantees, dict):
        raise RunnerError("Manifest has no build_guarantees object")
    if guarantees.get("remote_model_code_allowed") is not False:
        raise RunnerError("Manifest does not prove that remote model code was disabled")
    if guarantees.get("weight_format") != "safetensors_only":
        raise RunnerError("Manifest does not declare safetensors-only weights")
    donor_calls = guarantees.get("donor_forward_calls_build")
    if not isinstance(donor_calls, int) or isinstance(donor_calls, bool) or donor_calls != 0:
        raise RunnerError("Manifest does not prove zero donor forward calls during the build")
    if guarantees.get("donor_outputs_used") is not False:
        raise RunnerError("Manifest does not prove that donor outputs were unused")
    if guarantees.get("standalone_reload_verified") is not True:
        raise RunnerError("Manifest does not prove a successful standalone reload")
    evaluation = manifest.get("evaluation_summary")
    if not isinstance(evaluation, dict) or evaluation.get("all_gates_passed") is not True:
        raise RunnerError("The checkpoint did not pass every recorded evaluation gate")
    return manifest


def verify_artifact_inventory(
    output_root: Path,
    model_dir: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Verify every inventoried file and reject untracked model assets."""

    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise RunnerError("Manifest artifact inventory is empty or invalid")
    if len(raw_artifacts) > MAX_ARTIFACT_COUNT:
        raise RunnerError(
            f"Artifact inventory exceeds the limit of {MAX_ARTIFACT_COUNT} files"
        )

    verified: list[dict[str, Any]] = []
    inventoried_paths: set[str] = set()
    declared_package_bytes = 0
    for index, raw_entry in enumerate(raw_artifacts):
        if not isinstance(raw_entry, dict):
            raise RunnerError(f"Artifact entry {index} is not an object")
        path_text = raw_entry.get("path")
        artifact_path = _safe_artifact_path(output_root, path_text)
        if path_text in inventoried_paths:
            raise RunnerError(f"Duplicate artifact path in manifest: {path_text!r}")
        inventoried_paths.add(path_text)

        expected_hash = raw_entry.get("sha256")
        expected_bytes = raw_entry.get("bytes")
        if not isinstance(expected_hash, str) or not HEX_SHA256.fullmatch(expected_hash):
            raise RunnerError(f"Invalid SHA-256 for artifact {path_text!r}")
        if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes < 0:
            raise RunnerError(f"Invalid byte count for artifact {path_text!r}")
        declared_package_bytes += expected_bytes
        if declared_package_bytes > MAX_PACKAGE_BYTES:
            raise RunnerError(
                f"Declared package exceeds the {MAX_PACKAGE_BYTES}-byte safety limit"
            )
        if not artifact_path.is_file():
            raise RunnerError(f"Manifest artifact is missing: {artifact_path}")
        actual_bytes = artifact_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RunnerError(
                f"Artifact size mismatch for {path_text}: "
                f"expected={expected_bytes}, actual={actual_bytes}"
            )
        actual_hash = sha256_file(artifact_path)
        if actual_hash != expected_hash:
            raise RunnerError(
                f"Artifact SHA-256 mismatch for {path_text}: "
                f"expected={expected_hash}, actual={actual_hash}"
            )
        verified.append(
            {"path": path_text, "bytes": actual_bytes, "sha256": actual_hash}
        )

    model_entries = list(model_dir.rglob("*"))
    symlinks = [path for path in model_entries if path.is_symlink()]
    if symlinks:
        raise RunnerError(f"Symbolic link rejected in model directory: {symlinks[0]}")
    model_files = [path for path in model_entries if path.is_file()]
    if not model_files:
        raise RunnerError(f"Model directory is empty: {model_dir}")
    for path in model_files:
        relative = path.resolve().relative_to(output_root).as_posix()
        if relative not in inventoried_paths:
            raise RunnerError(f"Untracked model asset rejected: {relative}")
        if path.suffix.lower() in UNSAFE_MODEL_SUFFIXES or path.suffix.lower() == ".py":
            raise RunnerError(f"Unsafe model asset rejected: {relative}")

    weight_files = [path for path in model_files if path.suffix.lower() == ".safetensors"]
    if not weight_files:
        raise RunnerError("No safetensors weight file was found in the model directory")
    return verified


def preflight_inner_manifests(
    model_dir: Path,
    outer_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check the engine manifest and audit saved beside the weights."""

    inner_path = model_dir / "lira_manifest.json"
    audit_path = model_dir / "audit.json"
    if not inner_path.is_file() or not audit_path.is_file():
        raise RunnerError("Model package must contain lira_manifest.json and audit.json")
    inner = read_json_object(inner_path)
    audit = read_json_object(audit_path)
    if inner.get("format") != "lira.experimental.parametric-assimilation":
        raise RunnerError(f"Unsupported inner Lira manifest format: {inner.get('format')!r}")

    outer_engine = outer_manifest.get("engine_manifest")
    outer_audit = outer_manifest.get("engine_audit")
    if not isinstance(outer_engine, Mapping) or not isinstance(outer_audit, Mapping):
        raise RunnerError("Outer manifest is missing its engine manifest or audit")

    outer_build_id = _build_id(outer_manifest)
    inner_build_id = inner.get("build_id")
    audit_build_id = audit.get("build_id")
    if not outer_build_id or inner_build_id != outer_build_id or audit_build_id != outer_build_id:
        raise RunnerError(
            "Build ID mismatch across outer manifest, inner Lira manifest, and audit"
        )

    expected_state = extract_expected_state_hash(outer_manifest)
    inner_output = inner.get("provenance", {}).get("output", {})
    inner_state = inner_output.get("state_sha256") if isinstance(inner_output, Mapping) else None
    audit_state = audit.get("output_state_sha256")
    if not expected_state or inner_state != expected_state or audit_state != expected_state:
        raise RunnerError("Output state hash mismatch across package manifests")

    inner_method = inner.get("method")
    if not isinstance(inner_method, Mapping) or inner_method.get("donor_forward_calls_build") != 0:
        raise RunnerError("Inner Lira manifest does not prove zero donor forward calls")
    if audit.get("donor_forward_calls_build") != 0:
        raise RunnerError("Inner audit does not prove zero donor forward calls")

    # save_standalone computes the inner artifact table on a deep copy. Some
    # V1/V2 outer manifests therefore retain an empty pre-save `artifacts`
    # field. Compare every other semantic field exactly, then validate the
    # inner artifact table against both disk and the outer package inventory.
    if dict(outer_audit) != audit:
        raise RunnerError("Outer engine_audit does not match assimilated-model/audit.json")
    outer_semantic = dict(outer_engine)
    inner_semantic = dict(inner)
    outer_engine_artifacts = outer_semantic.pop("artifacts", None)
    inner_artifacts = inner_semantic.pop("artifacts", None)
    if outer_semantic != inner_semantic:
        raise RunnerError(
            "Outer engine_manifest does not match assimilated-model/lira_manifest.json"
        )

    if not isinstance(inner_artifacts, Mapping) or not inner_artifacts:
        raise RunnerError("Inner Lira artifact inventory is empty or invalid")
    if outer_engine_artifacts not in (None, {}) and outer_engine_artifacts != inner_artifacts:
        raise RunnerError("Outer and inner engine artifact inventories conflict")

    outer_inventory: dict[str, Mapping[str, Any]] = {}
    for entry in outer_manifest.get("artifacts", []):
        if isinstance(entry, Mapping) and isinstance(entry.get("path"), str):
            outer_inventory[str(entry["path"])] = entry
    for name, metadata in inner_artifacts.items():
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or name in (".", "..")
            or not isinstance(metadata, Mapping)
        ):
            raise RunnerError(f"Invalid inner Lira artifact entry: {name!r}")
        path = model_dir / name
        if path.is_symlink() or not path.is_file():
            raise RunnerError(f"Inner Lira artifact is missing or unsafe: {name}")
        expected_bytes = metadata.get("bytes")
        expected_hash = metadata.get("sha256")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
            or not isinstance(expected_hash, str)
            or not HEX_SHA256.fullmatch(expected_hash)
        ):
            raise RunnerError(f"Invalid inner Lira artifact metadata: {name}")
        if path.stat().st_size != expected_bytes or sha256_file(path) != expected_hash:
            raise RunnerError(f"Inner Lira artifact hash/size mismatch: {name}")
        outer_entry = outer_inventory.get(f"{MODEL_DIRECTORY_NAME}/{name}")
        if (
            outer_entry is None
            or outer_entry.get("bytes") != expected_bytes
            or outer_entry.get("sha256") != expected_hash
        ):
            raise RunnerError(f"Inner artifact is inconsistent with outer inventory: {name}")
    return {
        "build_id": outer_build_id,
        "state_sha256": expected_state,
        "donor_forward_calls_build": 0,
        "cross_checked": True,
    }


def validate_model_config(model_dir: Path) -> dict[str, Any]:
    """Reject remote hooks and architectures outside this bounded Class-A runner."""
    config = read_json_object(model_dir / "config.json")
    if config.get("model_type") != "llama":
        raise RunnerError(f"Unsupported model_type: {config.get('model_type')!r}")
    if config.get("auto_map") not in (None, {}):
        raise RunnerError("Remote/custom auto_map entries are not allowed")
    bounded_fields = {
        "vocab_size": 65_536,
        "hidden_size": 2_048,
        "intermediate_size": 8_192,
        "num_hidden_layers": 48,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "max_position_embeddings": 65_536,
    }
    values: dict[str, int] = {}
    for name, maximum in bounded_fields.items():
        value = config.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or value > maximum
        ):
            raise RunnerError(
                f"Config field {name!r} must be an integer from 1 to {maximum}"
            )
        values[name] = value
    heads = values["num_attention_heads"]
    key_value_heads = values["num_key_value_heads"]
    if values["hidden_size"] % heads != 0 or heads % key_value_heads != 0:
        raise RunnerError("Config has an invalid attention/GQA geometry")
    approximate_parameters = (
        values["vocab_size"] * values["hidden_size"]
        + values["num_hidden_layers"]
        * (
            4 * values["hidden_size"] * values["hidden_size"]
            + 3 * values["hidden_size"] * values["intermediate_size"]
        )
    )
    if approximate_parameters > 1_000_000_000:
        raise RunnerError(
            "Config exceeds the one-billion-parameter Class-A safety budget"
        )
    return {
        "model_type": "llama",
        "approximate_parameter_upper_bound": approximate_parameters,
        **values,
    }


def extract_expected_state_hash(manifest: Mapping[str, Any]) -> str | None:
    candidates: list[str] = []
    guarantees = manifest.get("build_guarantees")
    if isinstance(guarantees, Mapping):
        value = guarantees.get("standalone_state_sha256")
        if value:
            candidates.append(str(value))
    audit = manifest.get("engine_audit")
    if isinstance(audit, Mapping):
        value = audit.get("output_state_sha256")
        if value:
            candidates.append(str(value))
    engine = manifest.get("engine_manifest")
    if isinstance(engine, Mapping):
        provenance = engine.get("provenance")
        if isinstance(provenance, Mapping):
            output = provenance.get("output")
            if isinstance(output, Mapping) and output.get("state_sha256"):
                candidates.append(str(output["state_sha256"]))

    if not candidates:
        return None
    if any(not HEX_SHA256.fullmatch(value) for value in candidates):
        raise RunnerError("Manifest contains an invalid standalone state SHA-256")
    if len(set(candidates)) != 1:
        raise RunnerError(f"Manifest contains conflicting state hashes: {candidates}")
    return candidates[0]


def checkpoint_torch_dtype(manifest: Mapping[str, Any], torch_module: Any) -> Any:
    """Resolve the audited checkpoint dtype before runtime conversion."""

    audit = manifest.get("engine_audit")
    tensors = audit.get("tensors") if isinstance(audit, Mapping) else None
    values = {
        entry.get("dtype")
        for entry in tensors or []
        if isinstance(entry, Mapping) and entry.get("dtype")
    }
    known = {
        "torch.float16": torch_module.float16,
        "torch.bfloat16": torch_module.bfloat16,
        "torch.float32": torch_module.float32,
    }
    if len(values) == 1 and next(iter(values)) in known:
        return known[next(iter(values))]
    # Current Transformers accepts "auto". This is a fallback for older or
    # future manifests that did not record a uniform tensor dtype.
    return "auto"


def verify_loaded_state_dict(
    model: Any,
    expected_sha256: str | None,
    *,
    hasher: Callable[[Mapping[str, Any]], str] | None = None,
) -> str | None:
    if expected_sha256 is None:
        return None
    if hasher is None:
        from src import devorar

        hasher = devorar.state_dict_sha256
    actual = hasher(model.state_dict())
    if actual != expected_sha256:
        raise RunnerError(
            "Loaded standalone state SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={actual}"
        )
    return actual


def validate_prompt_inputs(
    prompts: Sequence[str] | None,
    system_prompt: str,
    *,
    raw: bool,
    max_prompt_chars: int,
) -> list[str]:
    resolved = list(prompts or DEMO_PROMPTS)
    if not resolved or len(resolved) > MAX_PROMPTS:
        raise RunnerError(f"Expected from 1 to {MAX_PROMPTS} prompts")
    for index, prompt in enumerate(resolved, start=1):
        if not isinstance(prompt, str) or not prompt.strip():
            raise RunnerError(f"Prompt {index} is empty")
        if len(prompt) > max_prompt_chars:
            raise RunnerError(
                f"Prompt {index} exceeds --max-prompt-chars ({max_prompt_chars})"
            )
    if len(system_prompt) > max_prompt_chars:
        raise RunnerError(f"System prompt exceeds --max-prompt-chars ({max_prompt_chars})")
    if not raw:
        for label, text in (("system prompt", system_prompt),) + tuple(
            (f"prompt {index}", prompt) for index, prompt in enumerate(resolved, start=1)
        ):
            token = next((item for item in BLOCKED_CHAT_DELIMITERS if item in text), None)
            if token:
                raise RunnerError(
                    f"Reserved chat delimiter {token!r} is not allowed in {label}; "
                    "use --raw only for an intentional raw-token test"
                )
    return resolved


def render_prompt(tokenizer: Any, prompt: str, system_prompt: str, *, raw: bool) -> str:
    if raw:
        return prompt
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise RunnerError("The local tokenizer has no usable chat template")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
        rendered = apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise RunnerError(f"Local chat template rejected the prompt: {exc}") from exc
    if not isinstance(rendered, str) or not rendered:
        raise RunnerError("Local chat template returned an empty prompt")
    return rendered


def load_local_tokenizer(model_dir: Path, auto_tokenizer_cls: Any) -> Any:
    try:
        return auto_tokenizer_cls.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise RunnerError(f"Could not load the verified local tokenizer: {exc}") from exc


def load_local_causal_lm(model_dir: Path, auto_model_cls: Any, *, dtype: Any) -> Any:
    try:
        return auto_model_cls.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        raise RunnerError(f"Could not load the verified local checkpoint: {exc}") from exc


def select_runtime(torch_module: Any) -> tuple[str, Any]:
    if torch_module.cuda.is_available():
        return "cuda", torch_module.float16
    return "cpu", torch_module.float32


def _move_inputs(encoded: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {name: value.to(device) for name, value in encoded.items()}


def generate_responses(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    system_prompt: str,
    raw: bool,
    device: str,
    max_input_tokens: int,
    max_new_tokens: int,
    torch_module: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if pad_token_id is None:
        raise RunnerError("Tokenizer defines neither pad_token_id nor eos_token_id")

    model.eval()
    for prompt in prompts:
        rendered = render_prompt(tokenizer, prompt, system_prompt, raw=raw)
        try:
            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                add_special_tokens=raw,
            )
        except Exception as exc:
            raise RunnerError(f"Local tokenizer could not encode a prompt: {exc}") from exc
        input_ids = encoded.get("input_ids")
        if input_ids is None or not hasattr(input_ids, "shape"):
            raise RunnerError("Local tokenizer did not return input_ids")
        input_tokens = int(input_ids.shape[-1])
        if input_tokens > max_input_tokens:
            raise RunnerError(
                f"Encoded prompt has {input_tokens} tokens; limit is {max_input_tokens}"
            )
        inputs = _move_inputs(encoded, device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": pad_token_id,
        }
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        try:
            with torch_module.inference_mode():
                output_ids = model.generate(**inputs, **generation_kwargs)
            generated_ids = output_ids[0, input_tokens:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        except Exception as exc:
            raise RunnerError(f"Deterministic local generation failed: {exc}") from exc
        results.append(
            {
                "prompt": prompt,
                "response": response,
                "input_tokens": input_tokens,
                "generated_tokens": int(generated_ids.shape[-1]),
            }
        )
    return results


def download_pinned_host(cache_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RunnerError(
            "--compare-host requires huggingface_hub from requirements-colab.txt"
        ) from exc
    try:
        snapshot = snapshot_download(
            repo_id=HOST_MODEL,
            revision=HOST_REVISION,
            cache_dir=str(cache_dir),
            allow_patterns=list(HOST_ALLOW_PATTERNS),
            ignore_patterns=list(HOST_DENY_PATTERNS),
        )
    except Exception as exc:
        raise RunnerError(f"Could not download the immutable pinned host: {exc}") from exc
    path = Path(snapshot).resolve()
    if not (path / "config.json").is_file():
        raise RunnerError(f"Pinned host snapshot is incomplete: {path}")
    unsafe = [
        item
        for item in path.rglob("*")
        if item.is_file()
        and (item.suffix.lower() in UNSAFE_MODEL_SUFFIXES or item.suffix.lower() == ".py")
    ]
    if unsafe:
        raise RunnerError(f"Pinned host snapshot contains a rejected file: {unsafe[0]}")
    if not any(path.glob("*.safetensors")):
        raise RunnerError("Pinned host snapshot contains no safetensors weights")
    return path


def release_runtime_memory(torch_module: Any) -> None:
    """Collect after the caller has deleted its final model reference."""
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def _build_id(manifest: Mapping[str, Any]) -> str | None:
    if manifest.get("build_id"):
        return str(manifest["build_id"])
    engine = manifest.get("engine_manifest")
    if isinstance(engine, Mapping) and engine.get("build_id"):
        return str(engine["build_id"])
    audit = manifest.get("engine_audit")
    if isinstance(audit, Mapping) and audit.get("build_id"):
        return str(audit["build_id"])
    return None


def _recipe(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    engine = manifest.get("engine_manifest")
    if isinstance(engine, Mapping) and isinstance(engine.get("recipe"), Mapping):
        return engine["recipe"]
    method = manifest.get("method")
    if isinstance(method, Mapping) and isinstance(method.get("recipe"), Mapping):
        return method["recipe"]
    return None


def _parameter_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    audit = manifest.get("engine_audit")
    if not isinstance(audit, Mapping):
        return {}
    tensors = audit.get("tensors")
    total = 0
    changed = 0
    if isinstance(tensors, list):
        for entry in tensors:
            if not isinstance(entry, Mapping):
                continue
            numel = entry.get("numel")
            changed_elements = entry.get("changed_elements")
            if isinstance(numel, int) and not isinstance(numel, bool) and numel >= 0:
                total += numel
            if (
                isinstance(changed_elements, int)
                and not isinstance(changed_elements, bool)
                and changed_elements >= 0
            ):
                changed += changed_elements
    return {
        "physical_tensor_count": audit.get("physical_tensor_count"),
        "logical_tensor_count": audit.get("logical_tensor_count"),
        "physical_parameter_count": total or None,
        "changed_parameter_values": changed if total else None,
        "changed_fraction": (changed / total) if total else None,
        "base_state_sha256": audit.get("base_state_sha256"),
        "donor_state_sha256": audit.get("donor_state_sha256"),
        "output_state_sha256": audit.get("output_state_sha256"),
    }


def make_report(
    *,
    output_root: Path,
    model_dir: Path,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    manifest_sha256_trusted: bool,
    verified_artifacts: Sequence[Mapping[str, Any]],
    expected_state_sha256: str | None,
    actual_state_sha256: str | None,
    device: str,
    runtime_dtype: Any,
    raw: bool,
    system_prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    candidate_results: Sequence[Mapping[str, Any]],
    host_results: Sequence[Mapping[str, Any]] | None,
    inner_preflight: Mapping[str, Any] | None = None,
    config_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidate_results):
        entry = {
            "prompt": candidate["prompt"],
            "candidate": candidate["response"],
            "input_tokens": candidate["input_tokens"],
            "candidate_generated_tokens": candidate["generated_tokens"],
        }
        if host_results is not None:
            host = host_results[index]
            entry["host"] = host["response"]
            entry["host_generated_tokens"] = host["generated_tokens"]
        results.append(entry)

    evaluation = manifest.get("evaluation_summary")
    parameter_summary = _parameter_summary(manifest)
    return {
        "schema": "dragonbrx.devorar.inference-report",
        "schema_version": 1,
        "status": "completed_verified_local_inference",
        "checkpoint": {
            "output_root": str(output_root),
            "model_directory": str(model_dir),
            "build_id": _build_id(manifest),
            "artifact_status": manifest.get("artifact_status"),
            "artifact_inventory_verified": True,
            "inner_manifest_audit_cross_checked": bool(
                inner_preflight and inner_preflight.get("cross_checked") is True
            ),
            "bounded_config_verified": bool(config_preflight),
            "config_preflight": dict(config_preflight or {}),
            "verified_artifact_count": len(verified_artifacts),
            "outer_manifest_sha256": manifest_sha256,
            "expected_state_sha256": expected_state_sha256,
            "actual_state_sha256": actual_state_sha256,
            "state_dict_verified": actual_state_sha256 is not None,
            "manifest_authenticity": (
                "matched_user_supplied_sha256"
                if manifest_sha256_trusted
                else "not_independently_authenticated"
            ),
            "recipe": _recipe(manifest),
            "build_evaluation": evaluation if isinstance(evaluation, Mapping) else None,
            "parameter_summary": parameter_summary,
        },
        "runtime": {
            "device": device,
            "dtype": str(runtime_dtype),
            "deterministic_greedy_decoding": True,
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": max_new_tokens,
            "local_checkpoint_loading_only": True,
            "trust_remote_code": False,
            "safetensors_only": True,
        },
        "presentation": {
            "raw_checkpoint_test": raw,
            "identity_source": (
                "none_raw_checkpoint" if raw else "presentation_system_prompt_not_weights"
            ),
            "system_prompt": None if raw else system_prompt,
        },
        "comparison": {
            "enabled": host_results is not None,
            "host": f"{HOST_MODEL}@{HOST_REVISION}" if host_results is not None else None,
            "same_prompt_and_candidate_tokenizer": host_results is not None,
            "donor_inference_performed": False,
        },
        "scientific_limits": {
            "hidden_chain_of_thought": "not accessed, extracted, or verified",
            "parameter_assimilation": (
                "produces a distinct checkpoint but does not by itself prove a new "
                "reasoning algorithm or retention of every donor capability"
            ),
        },
        "results": results,
    }


def print_readable_report(report: Mapping[str, Any]) -> None:
    checkpoint = report["checkpoint"]
    runtime = report["runtime"]
    presentation = report["presentation"]
    print("\nDragonBRX / Devorar — inferência local verificada")
    print(f"Checkpoint: {checkpoint['model_directory']}")
    print(f"Build ID: {checkpoint['build_id'] or 'não registrado'}")
    print(
        "Integridade: "
        f"{checkpoint['verified_artifact_count']} artefatos conferidos; "
        f"state_dict={'verificado' if checkpoint['state_dict_verified'] else 'sem hash no manifesto'}"
    )
    print(f"State SHA-256: {checkpoint['actual_state_sha256'] or 'não disponível'}")
    parameters = checkpoint.get("parameter_summary") or {}
    if parameters.get("physical_parameter_count"):
        changed = parameters.get("changed_parameter_values")
        total = parameters["physical_parameter_count"]
        fraction = parameters.get("changed_fraction")
        print(
            "Valores alterados: "
            f"{changed:,} de {total:,} ({fraction:.2%})".replace(",", ".")
        )
    evaluation = checkpoint.get("build_evaluation") or {}
    if evaluation:
        print(
            "Gates do build: "
            f"{'aprovados' if evaluation.get('all_gates_passed') else 'reprovados'}; "
            f"delta de regras={evaluation.get('deterministic_rule_score_delta')}; "
            "razão de perplexidade="
            f"{evaluation.get('perplexity_ratio_candidate_over_host')}"
        )
    print(f"Runtime: {runtime['device']} / {runtime['dtype']} / greedy determinístico")
    if presentation["raw_checkpoint_test"]:
        print("Identidade: nenhuma (teste --raw do checkpoint puro)")
    else:
        print("Identidade: prompt de sistema da camada de apresentação; não é prova nos pesos")
    print(
        "Limite científico: cadeia de pensamento oculta não foi acessada, "
        "extraída ou verificada."
    )
    for index, result in enumerate(report["results"], start=1):
        print(f"\n[{index}] Prompt: {result['prompt']}")
        print(f"Assimilado: {result['candidate']}")
        if "host" in result:
            print(f"Host pinado: {result['host']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and run the standalone Devorar checkpoint. By default the "
            "DragonBRX name is an explicit presentation prompt; use --raw to "
            "test only the checkpoint weights and tokenizer."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Devorar output root or its assimilated-model directory",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        type=sha256_argument,
        help="Optional trusted SHA-256 for the outer manifest",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help=f"Prompt to run; repeat up to {MAX_PROMPTS} times (default: built-in demo)",
    )
    parser.add_argument(
        "--system",
        default=DRAGONBRX_SYSTEM_PROMPT,
        help="Presentation-only system prompt (ignored by --raw)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Bypass chat/system presentation and tokenize each prompt as raw text",
    )
    parser.add_argument(
        "--compare-host",
        action="store_true",
        help="After releasing the candidate, compare with the immutable pinned host",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/content/huggingface-cache"),
        help="Hugging Face cache used only by --compare-host",
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=bounded_int(1, 16_000),
        default=DEFAULT_MAX_PROMPT_CHARS,
    )
    parser.add_argument(
        "--max-input-tokens",
        type=bounded_int(8, 4_096),
        default=1_024,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=bounded_int(1, 256),
        default=96,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_stdout",
        help="Print the final report as JSON instead of the readable view",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Also write the final JSON report to this file",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompts = validate_prompt_inputs(
        args.prompt,
        args.system,
        raw=args.raw,
        max_prompt_chars=args.max_prompt_chars,
    )
    output_root, model_dir = resolve_checkpoint_layout(args.output_dir)
    manifest_file = output_root / MANIFEST_NAME
    actual_manifest_sha256 = sha256_file(manifest_file)
    if args.expected_manifest_sha256 is not None:
        if actual_manifest_sha256 != args.expected_manifest_sha256:
            raise RunnerError(
                "Outer manifest SHA-256 mismatch: "
                f"expected={args.expected_manifest_sha256}, actual={actual_manifest_sha256}"
            )
    manifest = load_manifest(output_root)
    verified_artifacts = verify_artifact_inventory(output_root, model_dir, manifest)
    inner_preflight = preflight_inner_manifests(model_dir, manifest)
    config_preflight = validate_model_config(model_dir)
    expected_state_sha256 = extract_expected_state_hash(manifest)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RunnerError(
            "Missing inference dependencies; run start_colab.py or install "
            "requirements-colab.txt first"
        ) from exc

    device, runtime_dtype = select_runtime(torch)
    tokenizer = load_local_tokenizer(model_dir, AutoTokenizer)
    candidate_load_dtype = checkpoint_torch_dtype(manifest, torch)
    candidate_model = load_local_causal_lm(
        model_dir,
        AutoModelForCausalLM,
        dtype=candidate_load_dtype,
    )
    actual_state_sha256 = verify_loaded_state_dict(candidate_model, expected_state_sha256)
    candidate_model.to(device=device, dtype=runtime_dtype)
    candidate_results = generate_responses(
        candidate_model,
        tokenizer,
        prompts,
        system_prompt=args.system,
        raw=args.raw,
        device=device,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        torch_module=torch,
    )
    del candidate_model
    release_runtime_memory(torch)

    host_results = None
    if args.compare_host:
        host_snapshot = download_pinned_host(args.cache_dir.expanduser().resolve())
        host_model = load_local_causal_lm(
            host_snapshot,
            AutoModelForCausalLM,
            dtype=runtime_dtype,
        )
        host_model.to(device=device, dtype=runtime_dtype)
        host_results = generate_responses(
            host_model,
            tokenizer,
            prompts,
            system_prompt=args.system,
            raw=args.raw,
            device=device,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            torch_module=torch,
        )
        del host_model
        release_runtime_memory(torch)

    return make_report(
        output_root=output_root,
        model_dir=model_dir,
        manifest=manifest,
        manifest_sha256=actual_manifest_sha256,
        manifest_sha256_trusted=args.expected_manifest_sha256 is not None,
        verified_artifacts=verified_artifacts,
        expected_state_sha256=expected_state_sha256,
        actual_state_sha256=actual_state_sha256,
        device=device,
        runtime_dtype=runtime_dtype,
        raw=args.raw,
        system_prompt=args.system,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        candidate_results=candidate_results,
        host_results=host_results,
        inner_preflight=inner_preflight,
        config_preflight=config_preflight,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
        encoded = encode_json(report)
        if args.json_output is not None:
            destination = args.json_output.expanduser().resolve()
            write_json_atomic(destination, encoded)
        if args.json_stdout:
            print(encoded)
        else:
            print_readable_report(report)
        return 0
    except RunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
