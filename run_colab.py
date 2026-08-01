#!/usr/bin/env python3
"""Bounded Google Colab runner for the experimental Devorar prototype.

This program performs direct DARE-style parameter-delta assimilation.  It does
not train on, request, or inspect donor generations.  The donor is used only as
an explicitly licensed source of compatible tensors and is released before
candidate inference begins.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from src import devorar


BASE_MODEL = "HuggingFaceTB/SmolLM2-360M"
BASE_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
DONOR_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
DONOR_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
SOURCE_LICENSE = "Apache-2.0"
OUTPUT_SENTINEL_NAME = ".devorar-output-v1"
OUTPUT_SENTINEL_CONTENT = "DragonBRX/Devorar managed output v1\n"

# Only safetensors are accepted as model weights.  JSON/tokenizer assets and
# license text are required to instantiate and audit a Transformers checkpoint.
SNAPSHOT_ALLOW_PATTERNS = (
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
SNAPSHOT_DENY_PATTERNS = (
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

RULE_PROMPTS: tuple[dict[str, str], ...] = (
    {
        "id": "arithmetic",
        "prompt": "Answer with only the number. What is 2 + 3?",
        "rule": "exact",
        "expected": "5",
    },
    {
        "id": "capital",
        "prompt": "Complete with one word: The capital of France is",
        "rule": "exact",
        "expected": "Paris",
    },
    {
        "id": "identity_copy",
        "prompt": "Write exactly this identifier and nothing else: DragonBRX",
        "rule": "exact",
        "expected": "DragonBRX",
    },
    {
        "id": "binary_instruction",
        "prompt": "Reply only YES: Is water made of hydrogen and oxygen?",
        "rule": "exact",
        "expected": "YES",
    },
)

RETENTION_TEXTS: tuple[str, ...] = (
    "A careful experiment records its inputs, method, limits, and measured results.",
    "Knowledge becomes reliable when a claim can be checked against reproducible evidence.",
    "Software should preserve user data, reject unsafe inputs, and report failures clearly.",
    "O conhecimento cresce quando hipóteses são testadas e os resultados são comparados.",
)


def _jsonable(value: Any) -> Any:
    """Convert dataclasses and common model metadata into JSON-safe values."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def system_resources(torch_module: Any | None = None) -> dict[str, Any]:
    """Return a small, non-sensitive resource inventory for the manifest."""
    total_ram = available_ram = process_rss = None
    try:
        import psutil

        memory = psutil.virtual_memory()
        total_ram = round(memory.total / (1024**3), 3)
        available_ram = round(memory.available / (1024**3), 3)
        process_rss = round(psutil.Process(os.getpid()).memory_info().rss / (1024**3), 3)
    except ImportError:
        pass

    gpu: dict[str, Any] = {"available": False}
    if torch_module is not None and torch_module.cuda.is_available():
        index = torch_module.cuda.current_device()
        properties = torch_module.cuda.get_device_properties(index)
        gpu = {
            "available": True,
            "name": properties.name,
            "memory_gib": round(properties.total_memory / (1024**3), 3),
            "cuda_version": getattr(torch_module.version, "cuda", None),
        }
    packages: dict[str, str | None] = {}
    for package in (
        "torch",
        "transformers",
        "huggingface_hub",
        "safetensors",
        "accelerate",
        "psutil",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ram_total_gib": total_ram,
        "ram_available_gib": available_ram,
        "process_rss_gib": process_rss,
        "gpu": gpu,
        "packages": packages,
    }


def print_resources(label: str, torch_module: Any | None = None) -> dict[str, Any]:
    resources = system_resources(torch_module)
    print(f"\n[{label}]")
    print(json.dumps(resources, ensure_ascii=False, indent=2))
    return resources


def enforce_resource_budget(
    *,
    stage: str,
    max_process_ram_gib: float,
    min_available_ram_gib: float = 0.0,
) -> None:
    """Fail early when the bounded experiment no longer fits its RAM budget."""
    try:
        import psutil
    except ImportError:
        return

    process_gib = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    available_gib = psutil.virtual_memory().available / (1024**3)
    if process_gib > max_process_ram_gib:
        raise MemoryError(
            f"RAM guard at {stage}: process uses {process_gib:.2f} GiB, "
            f"above --max-process-ram-gib={max_process_ram_gib:.2f}."
        )
    if available_gib < min_available_ram_gib:
        raise MemoryError(
            f"RAM guard at {stage}: only {available_gib:.2f} GiB available, "
            f"below required {min_available_ram_gib:.2f} GiB."
        )


def download_snapshot(
    repo_id: str,
    revision: str,
    *,
    cache_dir: Path,
    snapshot_download_func: Callable[..., str] | None = None,
) -> Path:
    """Download an immutable, safetensors-only model snapshot."""
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("Model revision must be a full lowercase 40-character commit hash")
    if snapshot_download_func is None:
        from huggingface_hub import snapshot_download

        snapshot_download_func = snapshot_download

    snapshot = snapshot_download_func(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(cache_dir),
        allow_patterns=list(SNAPSHOT_ALLOW_PATTERNS),
        ignore_patterns=list(SNAPSHOT_DENY_PATTERNS),
    )
    snapshot_path = Path(snapshot)
    unsafe_weights = [
        path
        for pattern in ("*.bin", "*.pt", "*.pth", "*.ckpt", "*.h5", "*.msgpack")
        for path in snapshot_path.rglob(pattern)
    ]
    if unsafe_weights:
        raise RuntimeError(f"Rejected non-safetensors weight files: {unsafe_weights}")
    if not any(snapshot_path.rglob("*.safetensors")):
        raise RuntimeError(f"No safetensors weights found in pinned snapshot {repo_id}@{revision}")
    return snapshot_path


def load_config_and_tokenizer(
    snapshot_path: Path,
    *,
    transformers_module: Any | None = None,
) -> tuple[Any, Any]:
    """Load non-weight model metadata without permitting repository code."""
    if transformers_module is None:
        import transformers as transformers_module

    common = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    config = transformers_module.AutoConfig.from_pretrained(str(snapshot_path), **common)
    tokenizer = transformers_module.AutoTokenizer.from_pretrained(str(snapshot_path), **common)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, config


def load_causal_lm(
    snapshot_path: Path,
    *,
    config: Any,
    dtype: Any,
    transformers_module: Any | None = None,
) -> Any:
    """Load only safetensors weights from an already-vetted local snapshot."""
    if transformers_module is None:
        import transformers as transformers_module

    model = transformers_module.AutoModelForCausalLM.from_pretrained(
        str(snapshot_path),
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    return model


def load_model_assets(
    snapshot_path: Path,
    *,
    dtype: Any,
    transformers_module: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Load a complete local snapshot without permitting repository code."""
    tokenizer, config = load_config_and_tokenizer(
        snapshot_path,
        transformers_module=transformers_module,
    )
    model = load_causal_lm(
        snapshot_path,
        config=config,
        dtype=dtype,
        transformers_module=transformers_module,
    )
    return model, tokenizer, config


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def evaluate_rule(output: str, rule: Mapping[str, str]) -> bool:
    normalized_output = _normalize_text(output)
    normalized_expected = _normalize_text(rule["expected"])
    kind = rule["rule"]
    if kind == "contains_text":
        return normalized_expected in normalized_output
    if kind == "contains_word":
        words = "".join(
            character if character.isalnum() else " " for character in normalized_output
        ).split()
        return normalized_expected in words
    if kind == "exact":
        return normalized_output == normalized_expected
    raise ValueError(f"Unsupported deterministic rule: {kind}")


def render_prompt(tokenizer: Any, prompt: str) -> str:
    """Use an available Instruct chat template for both host and candidate."""
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    device: Any,
    max_input_tokens: int,
    max_new_tokens: int,
    torch_module: Any,
) -> str:
    rendered_prompt = render_prompt(tokenizer, prompt)
    encoded = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = encoded["input_ids"].shape[-1]
    with torch_module.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = generated[0, input_length:]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def evaluate_deterministic_rules(
    model: Any,
    tokenizer: Any,
    *,
    device: Any,
    max_prompts: int,
    max_input_tokens: int,
    max_new_tokens: int,
    torch_module: Any,
) -> dict[str, Any]:
    selected = RULE_PROMPTS[:max_prompts]
    results = []
    for rule in selected:
        output = generate_text(
            model,
            tokenizer,
            rule["prompt"],
            device=device,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            torch_module=torch_module,
        )
        passed = evaluate_rule(output, rule)
        results.append(
            {
                "id": rule["id"],
                "prompt": rule["prompt"],
                "expected": rule["expected"],
                "rule": rule["rule"],
                "output": output,
                "passed": passed,
            }
        )
    passed_count = sum(int(result["passed"]) for result in results)
    return {
        "score": passed_count / len(results) if results else 0.0,
        "passed": passed_count,
        "total": len(results),
        "results": results,
    }


def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    *,
    device: Any,
    max_tokens: int,
    torch_module: Any,
) -> dict[str, Any]:
    text = "\n\n".join(RETENTION_TEXTS)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch_module.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
    nll = float(output.loss.detach().float().cpu().item())
    if not math.isfinite(nll):
        raise RuntimeError("Retention evaluation produced a non-finite loss")
    try:
        perplexity: float | None = math.exp(nll)
    except OverflowError:
        perplexity = None
    return {
        "negative_log_likelihood": nll,
        "perplexity": perplexity,
        "perplexity_overflow": perplexity is None,
        "tokens": int(input_ids.numel()),
        "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def evaluate_model(
    model: Any,
    tokenizer: Any,
    *,
    device: Any,
    max_prompts: int,
    max_input_tokens: int,
    max_new_tokens: int,
    perplexity_tokens: int,
    torch_module: Any,
) -> dict[str, Any]:
    started = time.monotonic()
    model.to(device)
    model.eval()
    deterministic = evaluate_deterministic_rules(
        model,
        tokenizer,
        device=device,
        max_prompts=max_prompts,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        torch_module=torch_module,
    )
    perplexity = evaluate_perplexity(
        model,
        tokenizer,
        device=device,
        max_tokens=perplexity_tokens,
        torch_module=torch_module,
    )
    model.to("cpu")
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    return {
        "deterministic_rules": deterministic,
        "retention_corpus": perplexity,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def compare_evaluations(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    max_perplexity_ratio: float,
    max_rule_score_drop: float,
) -> dict[str, Any]:
    baseline_nll = float(baseline["retention_corpus"]["negative_log_likelihood"])
    candidate_nll = float(candidate["retention_corpus"]["negative_log_likelihood"])
    nll_delta = candidate_nll - baseline_nll
    finite_retention = all(math.isfinite(value) for value in (baseline_nll, candidate_nll, nll_delta))
    try:
        ratio: float | None = math.exp(nll_delta) if finite_retention else None
    except OverflowError:
        ratio = None
    baseline_rules = float(baseline["deterministic_rules"]["score"])
    candidate_rules = float(candidate["deterministic_rules"]["score"])
    rule_delta = candidate_rules - baseline_rules
    retention_passed = finite_retention and nll_delta <= math.log(max_perplexity_ratio)
    rules_passed = rule_delta >= -max_rule_score_drop
    behavioral_transfer_observed = rule_delta > 0.0
    return {
        "perplexity_ratio_candidate_over_host": ratio,
        "negative_log_likelihood_delta_candidate_minus_host": nll_delta,
        "perplexity_ratio_overflow": ratio is None,
        "max_perplexity_ratio": max_perplexity_ratio,
        "retention_passed": retention_passed,
        "deterministic_rule_score_delta": rule_delta,
        "max_rule_score_drop": max_rule_score_drop,
        "rules_passed": rules_passed,
        "behavioral_transfer_observed": behavioral_transfer_observed,
        "quality_gates_passed": retention_passed and rules_passed,
        "all_gates_passed": retention_passed and rules_passed and behavioral_transfer_observed,
        "promotion": "none_experimental_only",
    }


def _artifact_inventory(output_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and not path.name.endswith(".tmp"):
            artifacts.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return artifacts


def _save_with_engine(
    result: Any,
    output_dir: Path,
    *,
    config: Any,
    tokenizer: Any,
    overwrite: bool,
) -> None:
    """Call the engine's mandatory safetensors standalone serializer."""
    devorar.save_standalone(
        result,
        output_dir,
        config=config,
        tokenizer=tokenizer,
        overwrite=overwrite,
    )
    safetensors_files = list(output_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise RuntimeError("Assimilation engine did not produce a safetensors checkpoint")
    forbidden = [path for pattern in ("*.bin", "*.pt", "*.pth") for path in output_dir.glob(pattern)]
    if forbidden:
        raise RuntimeError(f"Unsafe checkpoint artifacts were produced: {forbidden}")


def audited_donor_forward_calls(assimilation: Any) -> int:
    """Read and enforce the engine's build-time donor-forward invariant."""
    audit = getattr(assimilation, "audit", None)
    if audit is None or not hasattr(audit, "donor_forward_calls_build"):
        raise RuntimeError("Assimilation audit is missing donor_forward_calls_build")
    count = getattr(audit, "donor_forward_calls_build")
    if isinstance(count, bool) or not isinstance(count, int):
        raise RuntimeError("Invalid donor_forward_calls_build audit value")
    if count != 0:
        raise RuntimeError(
            "Build audit rejected: donor model was invoked "
            f"{count} time(s); candidate will not be saved or evaluated."
        )
    return count


def _validate_output_target(target: Path) -> None:
    """Reject broad or ambiguous destinations before any replacement."""
    resolved = target.resolve()
    protected = {
        Path.cwd().resolve(),
        Path.home().resolve(),
        Path(__file__).resolve().parent,
    }
    contains_protected_path = any(path == resolved or path.is_relative_to(resolved) for path in protected)
    if resolved == Path(resolved.anchor) or contains_protected_path:
        raise RuntimeError(f"Refusing broad output directory: {resolved}")
    if target.exists() and (not target.is_dir() or target.is_symlink()):
        raise RuntimeError(f"Output target must be a real directory, not a file or symlink: {target}")


def _require_managed_output(target: Path) -> None:
    sentinel = target / OUTPUT_SENTINEL_NAME
    try:
        sentinel_content = sentinel.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            "Refusing to overwrite a directory not owned by Devorar; "
            f"missing readable sentinel {sentinel}"
        ) from error
    if sentinel_content != OUTPUT_SENTINEL_CONTENT:
        raise RuntimeError(f"Refusing invalid Devorar output sentinel: {sentinel}")


def _commit_output_directory(staging: Path, target: Path, *, overwrite: bool) -> None:
    """Atomically promote staging, restoring the previous directory on failure."""
    backup: Path | None = None
    if target.exists():
        if not overwrite:
            raise RuntimeError(
                f"Output directory already exists: {target}. Choose another path or pass --overwrite-output."
            )
        _require_managed_output(target)
        backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
        os.replace(target, backup)
    try:
        os.replace(staging, target)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    if backup is not None:
        expected_parent = target.parent.resolve()
        if backup.parent.resolve() != expected_parent or not backup.name.startswith(f".{target.name}.backup-"):
            raise RuntimeError(f"Refusing to remove unexpected backup path: {backup}")
        shutil.rmtree(backup)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Build in a sibling staging directory and promote only after full success."""
    raw_target = args.output_dir.expanduser().absolute()
    if raw_target.is_symlink():
        raise RuntimeError(f"Refusing symlink output directory: {raw_target}")
    target = raw_target.resolve()
    _validate_output_target(target)
    cache_target = args.cache_dir.expanduser().absolute().resolve()
    paths_overlap = (
        target == cache_target
        or target.is_relative_to(cache_target)
        or cache_target.is_relative_to(target)
    )
    if paths_overlap:
        raise RuntimeError(
            f"Output and cache directories must not overlap: output={target}, cache={cache_target}"
        )
    if target.exists():
        if not args.overwrite_output:
            raise RuntimeError(
                f"Output directory already exists: {target}. Choose another path or pass --overwrite-output."
            )
        _require_managed_output(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    staged_args = argparse.Namespace(**vars(args))
    staged_args.output_dir = staging
    try:
        manifest = _run_staged(staged_args, final_output_dir=target)
        _commit_output_directory(staging, target, overwrite=args.overwrite_output)
        return manifest
    except Exception:
        if staging.exists():
            expected_parent = target.parent.resolve()
            if staging.parent.resolve() == expected_parent and staging.name.startswith(f".{target.name}.staging-"):
                shutil.rmtree(staging, ignore_errors=True)
        raise


def _run_staged(args: argparse.Namespace, *, final_output_dir: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required. Run the dependency cell first.") from error

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / OUTPUT_SENTINEL_NAME).write_text(OUTPUT_SENTINEL_CONTENT, encoding="utf-8")
    model_dir = output_dir / "assimilated-model"
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    initial_resources = print_resources("resources: initial", torch)
    enforce_resource_budget(
        stage="startup",
        max_process_ram_gib=args.max_process_ram_gib,
        min_available_ram_gib=args.min_available_ram_gib,
    )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    print(f"Inference device: {device}; checkpoint build dtype: {dtype}")

    # Load host weights plus donor metadata.  The donor model itself does not
    # enter RAM until after the host baseline has been measured.
    print(f"\nDownloading pinned host: {BASE_MODEL}@{BASE_REVISION}")
    base_snapshot = download_snapshot(BASE_MODEL, BASE_REVISION, cache_dir=cache_dir)
    base_model, base_tokenizer, base_config = load_model_assets(base_snapshot, dtype=dtype)
    print(f"Downloading pinned donor files: {DONOR_MODEL}@{DONOR_REVISION}")
    donor_snapshot = download_snapshot(DONOR_MODEL, DONOR_REVISION, cache_dir=cache_dir)
    donor_tokenizer, donor_config = load_config_and_tokenizer(donor_snapshot)
    enforce_resource_budget(
        stage="host-loaded",
        max_process_ram_gib=args.max_process_ram_gib,
    )
    print("Measuring host baseline before loading donor...")
    baseline = evaluate_model(
        base_model,
        # Baseline and candidate use the same Instruct presentation layer.
        donor_tokenizer,
        device=device,
        max_prompts=args.max_prompts,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        perplexity_tokens=args.perplexity_tokens,
        torch_module=torch,
    )
    print("\nHost baseline complete. Loading donor weights for direct merge...")
    donor_model = load_causal_lm(
        donor_snapshot,
        config=donor_config,
        dtype=dtype,
    )
    enforce_resource_budget(
        stage="donor-loaded",
        max_process_ram_gib=args.max_process_ram_gib,
    )

    recipe = devorar.AssimilationRecipe(
        alpha=args.alpha,
        drop_rate=args.drop_rate,
        seed=args.seed,
        require_config_match=True,
        require_tokenizer_match=True,
    )
    base_metadata = {
        "model_id": BASE_MODEL,
        "revision": BASE_REVISION,
        "license": SOURCE_LICENSE,
        "source": "https://huggingface.co/HuggingFaceTB/SmolLM2-360M",
    }
    donor_metadata = {
        "model_id": DONOR_MODEL,
        "revision": DONOR_REVISION,
        "license": SOURCE_LICENSE,
        "source": "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct",
    }

    print("Applying direct DARE parameter deltas (no donor inference)...")
    assimilation = devorar.assimilate_models(
        base_model,
        donor_model,
        recipe,
        base_config=base_config,
        donor_config=donor_config,
        base_tokenizer=base_tokenizer,
        donor_tokenizer=donor_tokenizer,
        base_metadata=base_metadata,
        donor_metadata=donor_metadata,
    )
    donor_forward_calls = audited_donor_forward_calls(assimilation)
    enforce_resource_budget(
        stage="assimilation-built",
        max_process_ram_gib=args.max_process_ram_gib,
    )

    _save_with_engine(
        assimilation,
        model_dir,
        config=donor_config,
        # The compatible Instruct tokenizer carries the candidate's chat
        # template/special-token presentation without invoking the donor model.
        tokenizer=donor_tokenizer,
        overwrite=args.overwrite_output,
    )
    notices_source = Path(__file__).resolve().with_name("THIRD_PARTY_MODELS.md")
    if not notices_source.is_file():
        raise RuntimeError(f"Missing third-party model notice: {notices_source}")
    shutil.copy2(notices_source, output_dir / "THIRD_PARTY_MODELS.md")

    # Preserve only JSON-safe audit data, then release donor, host, and result.
    # Reloading from the saved directory proves that the candidate is standalone.
    engine_manifest = _jsonable(getattr(assimilation, "manifest", {}))
    engine_audit = _jsonable(getattr(assimilation, "audit", {}))
    del donor_model
    del base_model
    del base_tokenizer
    del base_config
    del assimilation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    enforce_resource_budget(
        stage="donor-released",
        max_process_ram_gib=args.max_process_ram_gib,
    )

    print("Donor released. Reloading standalone candidate for evaluation...")
    candidate_model, candidate_tokenizer, _candidate_config = load_model_assets(
        model_dir,
        dtype=dtype,
    )
    reloaded_state_sha256 = devorar.state_dict_sha256(candidate_model.state_dict())
    expected_state_sha256 = str(engine_audit.get("output_state_sha256", ""))
    if not expected_state_sha256 or reloaded_state_sha256 != expected_state_sha256:
        raise RuntimeError(
            "Standalone reload hash mismatch: "
            f"expected={expected_state_sha256!r}, actual={reloaded_state_sha256!r}"
        )
    standalone_reload_verified = True
    # The saved candidate now owns its reloaded config/tokenizer assets.
    del donor_tokenizer
    del donor_config
    candidate = evaluate_model(
        candidate_model,
        candidate_tokenizer,
        device=device,
        max_prompts=args.max_prompts,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        perplexity_tokens=args.perplexity_tokens,
        torch_module=torch,
    )
    del candidate_model
    del candidate_tokenizer
    del _candidate_config
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    comparison = compare_evaluations(
        baseline,
        candidate,
        max_perplexity_ratio=args.max_perplexity_ratio,
        max_rule_score_drop=args.max_rule_score_drop,
    )
    evaluation = {
        "host_baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
        "limits": {
            "max_prompts": args.max_prompts,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "perplexity_tokens": args.perplexity_tokens,
            "max_process_ram_gib": args.max_process_ram_gib,
        },
    }
    write_json(output_dir / "evaluation.json", evaluation)

    final_resources = print_resources("resources: final", torch)
    manifest_path = output_dir / "assimilation-manifest.lira.json"
    manifest = {
        "format": "dragonbrx.experimental-parameter-assimilation",
        "format_version": 1,
        "artifact_status": "experimental_non_canonical",
        "canonical_lira": False,
        "promotion": "forbidden_without_separate_review_and_explicit_action",
        "warning": (
            "Research prototype: direct parameter merging does not decode thoughts, "
            "recover hidden reasoning, or guarantee transferred capabilities."
        ),
        "method": {
            "name": "DARE direct delta assimilation",
            "recipe": _jsonable(recipe),
            "formula": "host + alpha * keep_mask/(1-drop_rate) * (donor-host)",
            "training": False,
            "teacher_student_distillation": False,
        },
        "sources": {"host": base_metadata, "donor": donor_metadata},
        "build_guarantees": {
            "donor_forward_calls_build": donor_forward_calls,
            "donor_outputs_used": False,
            "remote_model_code_allowed": False,
            "weight_format": "safetensors_only",
            "immutable_revisions": True,
            "donor_released_before_candidate_inference": True,
            "standalone_reload_verified": standalone_reload_verified,
            "standalone_state_sha256": reloaded_state_sha256,
        },
        "evaluation_summary": comparison,
        "evaluation_file": "evaluation.json",
        "engine_manifest": engine_manifest,
        "engine_audit": engine_audit,
        "resources": {"initial": initial_resources, "final": final_resources},
        "artifacts": [],
    }
    write_json(
        output_dir / "run-summary.json",
        {
            "status": "completed_experimental",
            "canonical_lira": False,
            "output_directory": str(final_output_dir),
            "model_directory": "assimilated-model",
            "host": f"{BASE_MODEL}@{BASE_REVISION}",
            "donor": f"{DONOR_MODEL}@{DONOR_REVISION}",
            "evaluation_gates_passed": comparison["all_gates_passed"],
            "donor_forward_calls_build": donor_forward_calls,
        },
    )
    manifest["artifacts"] = [
        artifact
        for artifact in _artifact_inventory(output_dir)
        if artifact["path"] != manifest_path.name
    ]
    write_json(manifest_path, manifest)
    print(f"\nExperimental checkpoint ready for atomic promotion to: {final_output_dir}")
    print(f"Evaluation gates passed: {comparison['all_gates_passed']}")
    print("No automatic canonical Lira promotion was performed.")
    return manifest


def bounded_int(minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        parsed = int(value)
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"expected an integer from {minimum} to {maximum}")
        return parsed

    return parse


def bounded_float(minimum: float, maximum: float) -> Callable[[str], float]:
    def parse(value: str) -> float:
        parsed = float(value)
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"expected a number from {minimum} to {maximum}")
        return parsed

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded, experimental DARE parameter assimilation on Colab."
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        default=Path("/content/devorar-output"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("/content/huggingface-cache"))
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Atomically replace the complete chosen output directory after a successful run",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference even when CUDA exists")
    parser.add_argument("--alpha", type=bounded_float(0.0, 1.0), default=0.75)
    parser.add_argument("--drop-rate", type=bounded_float(0.0, 0.99), default=0.50)
    parser.add_argument("--seed", type=int, default=24051996)
    parser.add_argument("--max-prompts", type=bounded_int(1, len(RULE_PROMPTS)), default=4)
    parser.add_argument("--max-input-tokens", type=bounded_int(16, 256), default=128)
    parser.add_argument("--max-new-tokens", type=bounded_int(1, 64), default=24)
    parser.add_argument("--perplexity-tokens", type=bounded_int(64, 1024), default=512)
    parser.add_argument("--max-process-ram-gib", type=bounded_float(3.0, 32.0), default=10.0)
    parser.add_argument("--min-available-ram-gib", type=bounded_float(0.0, 32.0), default=4.0)
    parser.add_argument("--max-perplexity-ratio", type=bounded_float(1.0, 5.0), default=1.50)
    parser.add_argument("--max-rule-score-drop", type=bounded_float(0.0, 1.0), default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (MemoryError, RuntimeError, ValueError, devorar.AssimilationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
