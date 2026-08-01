#!/usr/bin/env python3
"""Prompt-conditioned parameter microscope for a verified Devorar checkpoint.

The probe does not assign a permanent human meaning to individual scalars.
It computes gradient×weight *proxy scores* for one generated response, groups
them by tensor/layer/role, and directly measures a small number of reversible
whole-tensor attenuation interventions.  Coordinate rankings are not themselves
intervention results.  No donor model is loaded or queried.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_model


PROBE_FORMAT = "lira.experimental.prompt-conditioned-parameter-intervention-probe"
PROBE_VERSION = 2
DEFAULT_GRADIENT_SCALE = 1024.0
_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


class ProbeError(RuntimeError):
    """Raised when a parameter probe cannot be completed without ambiguity."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def bounded_int(minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be an integer") from error
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"must be within [{minimum}, {maximum}]")
        return parsed

    return parse


def bounded_float(minimum: float, maximum: float):
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be a number") from error
        if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"must be finite and within [{minimum}, {maximum}]")
        return parsed

    return parse


def parameter_role(name: str) -> str:
    lowered = name.lower()
    checks = (
        (("embed_tokens", "embed.weight", "wte.weight"), "token_embedding"),
        (("lm_head", "head.weight"), "output_vocabulary_head"),
        (("q_proj", "wq_"), "attention_query"),
        (("k_proj", "wk_"), "attention_key"),
        (("v_proj", "wv_", "wkv"), "attention_value_or_kv"),
        (("o_proj", "wo_"), "attention_output"),
        (("gate_proj", ".gate."), "mlp_gate_or_router"),
        (("up_proj", ".w3."), "mlp_expansion"),
        (("down_proj", ".w2."), "mlp_contraction"),
        (("layernorm", "layer_norm", "_norm", ".norm."), "normalization"),
        (("router", "experts."), "moe_routing_or_expert"),
    )
    for fragments, role in checks:
        if any(fragment in lowered for fragment in fragments):
            return role
    return "other_parameter"


def parameter_layer(name: str) -> int | None:
    match = _LAYER_PATTERN.search(name)
    return int(match.group(1)) if match is not None else None


def unravel_index(flat_index: int, shape: Sequence[int]) -> list[int]:
    if flat_index < 0:
        raise ProbeError("flat parameter index cannot be negative")
    if not shape:
        if flat_index != 0:
            raise ProbeError("scalar tensor has only index zero")
        return []
    total = math.prod(shape)
    if flat_index >= total:
        raise ProbeError("flat parameter index is outside tensor shape")
    coordinates = [0] * len(shape)
    remainder = flat_index
    for position in range(len(shape) - 1, -1, -1):
        dimension = int(shape[position])
        if dimension <= 0:
            raise ProbeError("cannot unravel an empty tensor")
        coordinates[position] = remainder % dimension
        remainder //= dimension
    return coordinates


def _named_parameter_groups(model: Any) -> list[tuple[str, list[str], Any]]:
    try:
        named = list(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - old PyTorch compatibility
        named = list(model.named_parameters())
    groups: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for name, parameter in named:
        key = id(parameter)
        if key not in groups:
            groups[key] = {"parameter": parameter, "aliases": []}
            order.append(key)
        groups[key]["aliases"].append(name)
    return [
        (groups[key]["aliases"][0], list(groups[key]["aliases"]), groups[key]["parameter"])
        for key in order
    ]


def _merge_top_weights(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    return sorted(candidates, key=lambda item: item["absolute_attribution"], reverse=True)[:limit]


def score_parameter(
    *,
    name: str,
    aliases: Sequence[str],
    parameter: Any,
    gradient_scale: float,
    chunk_elements: int,
    top_weights: int,
    torch_module: Any,
) -> dict[str, Any] | None:
    gradient = parameter.grad
    if gradient is None:
        return None
    flat_parameter = parameter.detach().reshape(-1)
    flat_gradient = gradient.detach().reshape(-1)
    numel = int(flat_parameter.numel())
    absolute_total = 0.0
    signed_total = 0.0
    gradient_square_total = 0.0
    candidates: list[dict[str, Any]] = []
    shape = [int(value) for value in parameter.shape]
    for start in range(0, numel, chunk_elements):
        end = min(start + chunk_elements, numel)
        parameter_chunk = flat_parameter[start:end].to(dtype=torch_module.float32)
        gradient_chunk = flat_gradient[start:end].to(dtype=torch_module.float32) / gradient_scale
        if not bool(torch_module.isfinite(parameter_chunk).all()) or not bool(
            torch_module.isfinite(gradient_chunk).all()
        ):
            raise ProbeError(f"non-finite parameter or gradient detected in {name}")
        attribution = parameter_chunk * gradient_chunk
        absolute = attribution.abs()
        absolute_total += float(absolute.sum().item())
        signed_total += float(attribution.sum().item())
        gradient_square_total += float(gradient_chunk.square().sum().item())
        if top_weights:
            count = min(top_weights, end - start)
            values, indices = torch_module.topk(absolute, count)
            local_indices = indices.detach().cpu().tolist()
            absolute_values = values.detach().cpu().tolist()
            signed_values = attribution[indices].detach().cpu().tolist()
            parameter_values = parameter_chunk[indices].detach().cpu().tolist()
            gradient_values = gradient_chunk[indices].detach().cpu().tolist()
            for local, absolute_value, signed_value, parameter_value, gradient_value in zip(
                local_indices,
                absolute_values,
                signed_values,
                parameter_values,
                gradient_values,
            ):
                flat_index = start + int(local)
                candidates.append(
                    {
                        "evidence_class": "unverified_first_order_coordinate_proxy",
                        "locally_intervened": False,
                        "flat_index": flat_index,
                        "coordinates": unravel_index(flat_index, shape),
                        "parameter_value": float(parameter_value),
                        "gradient": float(gradient_value),
                        "signed_attribution": float(signed_value),
                        "absolute_attribution": float(absolute_value),
                    }
                )
        del parameter_chunk, gradient_chunk, attribution, absolute
    return {
        "name": name,
        "aliases": list(aliases),
        "layer": parameter_layer(name),
        "role": parameter_role(name),
        "shape": shape,
        "dtype": str(parameter.dtype),
        "numel": numel,
        "absolute_attribution_total": absolute_total,
        "absolute_attribution_mean": absolute_total / numel if numel else 0.0,
        "signed_attribution_total": signed_total,
        "signed_attribution_mean": signed_total / numel if numel else 0.0,
        "gradient_l2": math.sqrt(max(gradient_square_total, 0.0)),
        "gradient_rms": math.sqrt(max(gradient_square_total / numel, 0.0)) if numel else 0.0,
        "ranking_evidence_class": "first_order_tensor_proxy",
        "locally_intervened": False,
        "top_coordinate_proxies": _merge_top_weights(candidates, top_weights),
    }


def aggregate_scores(scores: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layers: dict[str, dict[str, Any]] = {}
    roles: dict[str, dict[str, Any]] = {}

    def add(target: dict[str, dict[str, Any]], key: str, item: Mapping[str, Any]) -> None:
        row = target.setdefault(
            key,
            {
                "group": key,
                "tensor_count": 0,
                "parameter_values": 0,
                "absolute_attribution_total": 0.0,
                "signed_attribution_total": 0.0,
            },
        )
        row["tensor_count"] += 1
        row["parameter_values"] += int(item["numel"])
        row["absolute_attribution_total"] += float(item["absolute_attribution_total"])
        row["signed_attribution_total"] += float(item["signed_attribution_total"])

    for score in scores:
        layer = score.get("layer")
        add(layers, "global" if layer is None else f"layer_{layer}", score)
        add(roles, str(score["role"]), score)
    for collection in (layers, roles):
        for row in collection.values():
            count = row["parameter_values"]
            row["absolute_attribution_mean"] = (
                row["absolute_attribution_total"] / count if count else 0.0
            )
            row["signed_attribution_mean"] = (
                row["signed_attribution_total"] / count if count else 0.0
            )
    layer_rows = sorted(layers.values(), key=lambda item: item["absolute_attribution_total"], reverse=True)
    role_rows = sorted(roles.values(), key=lambda item: item["absolute_attribution_total"], reverse=True)
    return layer_rows, role_rows


def _normalise_shares(scores: list[dict[str, Any]]) -> None:
    total = sum(float(item["absolute_attribution_total"]) for item in scores)
    for item in scores:
        item["absolute_attribution_share"] = (
            float(item["absolute_attribution_total"]) / total if total else 0.0
        )


def validate_exact_targets(exact_targets: Sequence[str]) -> list[str]:
    targets = list(exact_targets)
    if len(targets) > 8:
        raise ProbeError("at most 8 exact target tensors may be selected")
    if any(not name or name.strip() != name for name in targets):
        raise ProbeError("target tensor names must be non-empty exact names without outer spaces")
    if len(set(targets)) != len(targets):
        raise ProbeError("target tensor names must not be duplicated")
    return targets


def select_intervention_scores(
    scores: Sequence[Mapping[str, Any]],
    *,
    exact_targets: Sequence[str],
    default_limit: int,
) -> list[Mapping[str, Any]]:
    """Select exact canonical tensor names, or fall back to the ranked prefix."""

    targets = validate_exact_targets(exact_targets)
    by_name = {str(item["name"]): item for item in scores}
    if targets:
        missing = [name for name in targets if name not in by_name]
        if missing:
            raise ProbeError(
                "target tensor is not an exact canonical parameter with a gradient for this "
                f"response: {missing[0]}"
            )
        return [by_name[name] for name in targets]
    return list(scores[:default_limit])


def _move_inputs(encoded: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {name: value.to(device) for name, value in encoded.items()}


def generate_target(
    *,
    model: Any,
    tokenizer: Any,
    rendered_prompt: str,
    raw: bool,
    device: str,
    max_input_tokens: int,
    max_new_tokens: int,
    torch_module: Any,
) -> tuple[dict[str, Any], Any, Any]:
    encoded = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=raw)
    if "input_ids" not in encoded:
        raise ProbeError("tokenizer did not return input_ids")
    inputs = _move_inputs(encoded, device)
    input_ids = inputs["input_ids"]
    input_tokens = int(input_ids.shape[-1])
    if input_tokens > max_input_tokens:
        raise ProbeError(f"prompt has {input_tokens} tokens; limit is {max_input_tokens}")
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if pad_token_id is None:
        raise ProbeError("tokenizer defines neither pad_token_id nor eos_token_id")
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": pad_token_id,
    }
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    with torch_module.inference_mode():
        output = model.generate(**inputs, **kwargs)
    generated_ids = output[:, input_tokens:]
    if int(generated_ids.shape[-1]) <= 0:
        raise ProbeError("model generated no target tokens to probe")
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    attention = inputs.get("attention_mask")
    if attention is None:
        attention = torch_module.ones_like(input_ids)
    generated_attention = torch_module.ones_like(generated_ids, device=device)
    full_ids = torch_module.cat((input_ids, generated_ids), dim=-1)
    full_attention = torch_module.cat((attention, generated_attention), dim=-1)
    return (
        {
            "response": response,
            "input_tokens": input_tokens,
            "generated_tokens": int(generated_ids.shape[-1]),
            "decoding": "deterministic_greedy",
        },
        full_ids,
        full_attention,
    )


def objective_loss(
    model: Any,
    full_ids: Any,
    full_attention: Any,
    *,
    prompt_tokens: int,
    torch_module: Any,
) -> Any:
    labels = full_ids.clone()
    labels[:, :prompt_tokens] = -100
    outputs = model(
        input_ids=full_ids,
        attention_mask=full_attention,
        labels=labels,
        use_cache=False,
    )
    loss = outputs.loss
    if loss is None or not bool(torch_module.isfinite(loss)):
        raise ProbeError("probe objective loss is missing or non-finite")
    return loss


def taylor_predicted_loss_delta(*, signed_attribution_total: float, fraction: float) -> float:
    """Predict the loss change for ``w <- (1-fraction) * w`` at first order.

    ``signed_attribution_total`` is ``sum(dL/dw * w)`` at the unmodified
    checkpoint.  The exact parameter displacement is ``-fraction * w``.
    """

    if not math.isfinite(signed_attribution_total):
        raise ProbeError("signed attribution total must be finite")
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ProbeError("attenuation fraction must be finite and within (0, 1]")
    return -fraction * signed_attribution_total


def compare_loss_deltas(*, predicted: float, observed: float) -> dict[str, Any]:
    """Return a bounded, explicit comparison of Taylor and observed deltas."""

    if not math.isfinite(predicted) or not math.isfinite(observed):
        raise ProbeError("predicted and observed loss deltas must be finite")
    residual = observed - predicted
    scale = max(abs(predicted), abs(observed), 1e-12)
    tolerance = 1e-12
    direction_agreement: bool | None
    if abs(predicted) <= tolerance or abs(observed) <= tolerance:
        direction_agreement = None
    else:
        direction_agreement = (predicted > 0.0) == (observed > 0.0)
    return {
        "taylor_prediction_error": residual,
        "taylor_absolute_prediction_error": abs(residual),
        "taylor_relative_error_to_max_magnitude": abs(residual) / scale,
        "taylor_observed_direction_agreement": direction_agreement,
    }


def local_attenuation_interventions(
    *,
    model: Any,
    parameters: Mapping[str, Any],
    selected_scores: Sequence[Mapping[str, Any]],
    full_ids: Any,
    full_attention: Any,
    prompt_tokens: int,
    baseline_loss: float,
    fraction: float,
    torch_module: Any,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for score in selected_scores:
        name = str(score["name"])
        parameter = parameters[name]
        backup = parameter.detach().clone()
        try:
            with torch_module.no_grad():
                parameter.mul_(1.0 - fraction)
            with torch_module.inference_mode():
                changed = float(
                    objective_loss(
                        model,
                        full_ids,
                        full_attention,
                        prompt_tokens=prompt_tokens,
                        torch_module=torch_module,
                    ).item()
                )
        finally:
            with torch_module.no_grad():
                parameter.copy_(backup)
            del backup
        checks.append(
            {
                "tensor": name,
                "evidence_class": "observed_local_whole_tensor_intervention",
                "fixed_generated_sequence": True,
                "attenuation_fraction": fraction,
                "baseline_loss": baseline_loss,
                "attenuated_loss": changed,
                "observed_loss_delta": changed - baseline_loss,
                "selection_proxy": {
                    "numel": int(score["numel"]),
                    "absolute_attribution_total": float(score["absolute_attribution_total"]),
                    "absolute_attribution_mean": float(score["absolute_attribution_mean"]),
                    "signed_attribution_total": float(score["signed_attribution_total"]),
                    "signed_attribution_mean": float(score["signed_attribution_mean"]),
                },
                "taylor_predicted_loss_delta": taylor_predicted_loss_delta(
                    signed_attribution_total=float(score["signed_attribution_total"]),
                    fraction=fraction,
                ),
                "interpretation": (
                    "A positive observed delta means attenuation increased loss for the same "
                    "fixed token sequence. This is local objective sensitivity, not a decoded "
                    "concept, an isolated tensor response, or a permanent semantic label."
                ),
            }
        )
        checks[-1].update(
            compare_loss_deltas(
                predicted=float(checks[-1]["taylor_predicted_loss_delta"]),
                observed=float(checks[-1]["observed_loss_delta"]),
            )
        )
    return checks


def runtime_memory_metrics(torch_module: Any, *, device: str) -> dict[str, Any]:
    """Collect optional runtime memory facts without making psutil mandatory."""

    metrics: dict[str, Any] = {
        "process_rss_gib": None,
        "cuda_max_memory_allocated_gib": None,
    }
    try:
        import psutil

        metrics["process_rss_gib"] = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    except (ImportError, OSError, RuntimeError):
        pass
    if device.startswith("cuda") and bool(torch_module.cuda.is_available()):
        metrics["cuda_max_memory_allocated_gib"] = float(
            torch_module.cuda.max_memory_allocated()
        ) / (1024**3)
    return metrics


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().absolute()
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ProbeError(f"probe output must be a regular file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def _default_output() -> Path:
    content = Path("/content")
    return (
        content / "dragonbrx-parameter-probe.lira.json"
        if content.is_dir()
        else Path.cwd() / "dragonbrx-parameter-probe.lira.json"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure prompt-conditioned parameter proxy scores in a verified local Devorar checkpoint. "
            "Coordinate scores are first-order proxies; selected whole tensors receive local "
            "interventions. This does not decode hidden thoughts or fixed parameter meanings."
        )
    )
    parser.add_argument("--output-dir", type=Path, help="V2 output root or assimilated-model directory")
    parser.add_argument("--json-output", type=Path, default=_default_output())
    parser.add_argument("--prompt", default="Quem é você e como foi criado?")
    parser.add_argument("--system", default=run_model.DRAGONBRX_SYSTEM_PROMPT)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-prompt-chars", type=bounded_int(1, 16_000), default=4_000)
    parser.add_argument("--max-input-tokens", type=bounded_int(8, 2_048), default=512)
    parser.add_argument("--max-new-tokens", type=bounded_int(1, 128), default=32)
    parser.add_argument("--top-tensors", type=bounded_int(1, 64), default=20)
    parser.add_argument("--top-weights-per-tensor", type=bounded_int(0, 16), default=3)
    parser.add_argument("--intervention-tensors", type=bounded_int(0, 8), default=4)
    parser.add_argument(
        "--target-tensor",
        action="append",
        default=[],
        metavar="EXACT_NAME",
        help=(
            "intervene on this exact canonical named_parameter; repeat up to 8 times. "
            "When supplied, overrides the top-ranked intervention selection"
        ),
    )
    parser.add_argument("--ablation-fraction", type=bounded_float(0.001, 0.50), default=0.05)
    parser.add_argument("--chunk-elements", type=bounded_int(16_384, 4_194_304), default=262_144)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_exact_targets(args.target_tensor)
    prompt = run_model.validate_prompt_inputs(
        [args.prompt],
        args.system,
        raw=args.raw,
        max_prompt_chars=args.max_prompt_chars,
    )[0]
    output_root, model_dir = run_model.resolve_checkpoint_layout(args.output_dir)
    manifest = run_model.load_manifest(output_root)
    artifacts = run_model.verify_artifact_inventory(output_root, model_dir, manifest)
    run_model.preflight_inner_manifests(model_dir, manifest)
    run_model.validate_model_config(model_dir)
    expected_state_hash = run_model.extract_expected_state_hash(manifest)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ProbeError("probe requires the Colab dependencies installed by start_colab.py") from error

    if args.cpu:
        device, runtime_dtype = "cpu", torch.float32
    else:
        device, runtime_dtype = run_model.select_runtime(torch)
    tokenizer = run_model.load_local_tokenizer(model_dir, AutoTokenizer)
    load_dtype = run_model.checkpoint_torch_dtype(manifest, torch)
    model = run_model.load_local_causal_lm(model_dir, AutoModelForCausalLM, dtype=load_dtype)
    verified_hash = run_model.verify_loaded_state_dict(model, expected_state_hash)
    model.to(device=device, dtype=runtime_dtype)
    model.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    from src import devorar

    pre_probe_runtime_hash = devorar.state_dict_sha256(model.state_dict())
    rendered = run_model.render_prompt(tokenizer, prompt, args.system, raw=args.raw)
    generation, full_ids, full_attention = generate_target(
        model=model,
        tokenizer=tokenizer,
        rendered_prompt=rendered,
        raw=args.raw,
        device=device,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        torch_module=torch,
    )

    model.zero_grad(set_to_none=True)
    loss = objective_loss(
        model,
        full_ids,
        full_attention,
        prompt_tokens=generation["input_tokens"],
        torch_module=torch,
    )
    baseline_loss = float(loss.detach().item())
    (loss * DEFAULT_GRADIENT_SCALE).backward()
    del loss

    groups = _named_parameter_groups(model)
    scores: list[dict[str, Any]] = []
    parameter_map: dict[str, Any] = {}
    for name, aliases, parameter in groups:
        parameter_map[name] = parameter
        score = score_parameter(
            name=name,
            aliases=aliases,
            parameter=parameter,
            gradient_scale=DEFAULT_GRADIENT_SCALE,
            chunk_elements=args.chunk_elements,
            top_weights=args.top_weights_per_tensor,
            torch_module=torch,
        )
        if score is not None:
            scores.append(score)
    if not scores:
        raise ProbeError("no parameter received a gradient for the selected response")
    _normalise_shares(scores)
    scores.sort(key=lambda item: item["absolute_attribution_total"], reverse=True)
    top_scores = scores[: args.top_tensors]
    layer_scores, role_scores = aggregate_scores(scores)

    model.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    intervention_selection = select_intervention_scores(
        scores,
        exact_targets=args.target_tensor,
        default_limit=args.intervention_tensors,
    )
    intervention_checks = local_attenuation_interventions(
        model=model,
        parameters=parameter_map,
        selected_scores=intervention_selection,
        full_ids=full_ids,
        full_attention=full_attention,
        prompt_tokens=generation["input_tokens"],
        baseline_loss=baseline_loss,
        fraction=args.ablation_fraction,
        torch_module=torch,
    )
    intervened_names = {str(item["tensor"]) for item in intervention_checks}
    for score in scores:
        score["locally_intervened"] = str(score["name"]) in intervened_names
    post_probe_runtime_hash = devorar.state_dict_sha256(model.state_dict())
    if post_probe_runtime_hash != pre_probe_runtime_hash:
        raise ProbeError("reversible tensor interventions changed the checkpoint state")

    report = {
        "format": PROBE_FORMAT,
        "format_version": PROBE_VERSION,
        "canonical_lira": False,
        "status": "completed_prompt_conditioned_probe",
        "probe_id": str(uuid.uuid4()),
        "created_at": _utc_now(),
        "checkpoint": {
            "output_root": str(output_root),
            "model_directory": str(model_dir),
            "build_id": manifest.get("build_id"),
            "verified_artifact_count": len(artifacts),
            "expected_state_sha256": expected_state_hash,
            "loaded_state_sha256": verified_hash,
            "runtime_state_sha256_before_probe": pre_probe_runtime_hash,
            "runtime_state_sha256_after_probe": post_probe_runtime_hash,
            "checkpoint_restored_exactly": True,
            "donor_loaded": False,
        },
        "runtime": {
            "device": device,
            "dtype": str(runtime_dtype),
            "gradient_scale": DEFAULT_GRADIENT_SCALE,
            "chunk_elements": args.chunk_elements,
            **runtime_memory_metrics(torch, device=device),
        },
        "observation": {
            "prompt": prompt,
            "system_prompt_source": "none_raw" if args.raw else "presentation_controller",
            **generation,
            "response_objective_loss": baseline_loss,
        },
        "method": {
            "primary_ranking": "gradient_times_weight_first_order_proxy_absolute_and_signed",
            "ranking_order": "absolute_attribution_total_size_dependent",
            "size_normalized_comparison_field": "absolute_attribution_mean",
            "scope": "one exact prompt and generated response",
            "single_backward_for_all_parameter_groups": True,
            "local_intervention": "reversible whole-tensor attenuation on fixed generated tokens",
            "intervention_attenuation_fraction": args.ablation_fraction,
            "intervention_selection": (
                "exact_user_selected_tensor_names"
                if args.target_tensor
                else "top_ranked_first_order_tensor_proxies"
            ),
            "taylor_prediction": "first_order_gradient_dot_exact_parameter_displacement",
        },
        "rankings": {
            "parameter_tensors_scored": len(scores),
            "top_tensors": top_scores,
            "layers": layer_scores,
            "roles": role_scores,
        },
        "local_tensor_interventions": intervention_checks,
        "scientific_limits": {
            "individual_parameter_has_a_fixed_sentence_or_concept": False,
            "attribution_is_prompt_dependent": True,
            "coordinate_proxy_is_an_intervention": False,
            "whole_tensor_interventions_are_local_objective_measurements": True,
            "tensor_intervention_proves_fixed_semantics": False,
            "intervention_regenerated_output": False,
            "taylor_prediction_is_exact": False,
            "hidden_chain_of_thought_accessed": False,
            "claim": (
                "The report ranks first-order proxies and measures selected local tensor "
                "interventions for one fixed response. It does not decode a parameter's permanent "
                "meaning, prove a stored fact, or expose private reasoning."
            ),
        },
    }
    del model
    run_model.release_runtime_memory(torch)
    return report


def print_report(report: Mapping[str, Any], destination: Path) -> None:
    observation = report["observation"]
    print("\nDragonBRX / Devorar — microscópio de parâmetros")
    print(f"Prompt: {observation['prompt']}")
    print(f"Resposta observada: {observation['response']}")
    print(f"Tensores com gradiente: {report['rankings']['parameter_tensors_scored']}")
    print("Top tensores pelo proxy local de primeira ordem:")
    for item in report["rankings"]["top_tensors"][:5]:
        print(
            f"  {item['name']}: share={item['absolute_attribution_share']:.4%}, "
            f"role={item['role']}"
        )
    print(f"Checkpoint restaurado exatamente: {report['checkpoint']['checkpoint_restored_exactly']}")
    print(f"Relatório: {destination}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
        _write_json_atomic(args.json_output, report)
        print_report(report, args.json_output.expanduser().absolute())
        return 0
    except (ProbeError, run_model.RunnerError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
