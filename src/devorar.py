"""Direct, auditable parameter assimilation for homologous PyTorch models.

"Devorar" is the project metaphor.  Technically this module performs a DARE
(drop-and-rescale) transformation of the parameter delta between two models
with the *same* state/configuration/tokenizer contract.  It never invokes the
donor model and does not inspect logits, prompts, activations, or hidden chain
of thought.

The module is intentionally side-effect free on import: it neither downloads
models nor imports Transformers.  Callers may pass ordinary ``torch.nn.Module``
instances or Transformers models that have already been loaded with
``trust_remote_code=False``.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

import torch
from torch import Tensor, nn


LIRA_MANIFEST_FORMAT = "lira.experimental.parametric-assimilation"
LIRA_MANIFEST_VERSION = 1
ALGORITHM = "dare-delta-direct"
METHOD_CLASS = "A/direct-parameter-assimilation"


class AssimilationError(RuntimeError):
    """Base exception for an assimilation or persistence failure."""


class CompatibilityError(AssimilationError):
    """Raised when base and donor are not strictly homologous."""


class DonorForwardProhibited(AssimilationError):
    """Raised if any code attempts to invoke the donor during a build."""


@dataclass(frozen=True)
class AssimilationRecipe:
    """Immutable recipe for a direct DARE parameter transformation.

    ``drop_rate`` is the probability of dropping each donor delta element.
    Surviving elements are divided by ``1 - drop_rate`` before ``alpha`` is
    applied, preserving the expected delta magnitude.
    """

    alpha: float = 0.35
    drop_rate: float = 0.90
    seed: int = 0xD12A60B
    require_config_match: bool = True
    require_tokenizer_match: bool = True
    strict_dtype: bool = True
    fail_on_nonfinite: bool = True
    non_floating_policy: str = "require_equal"
    algorithm: str = ALGORITHM

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be finite and within [0, 1]")
        if not math.isfinite(self.drop_rate) or not 0.0 <= self.drop_rate < 1.0:
            raise ValueError("drop_rate must be finite and within [0, 1)")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if self.non_floating_policy != "require_equal":
            raise ValueError("the only safe non_floating_policy is 'require_equal'")
        if self.algorithm != ALGORITHM:
            raise ValueError(f"unsupported algorithm: {self.algorithm!r}")


@dataclass(frozen=True)
class ModelMetadata:
    """Provenance that is embedded in every Lira experimental manifest."""

    model_id: str
    revision: str
    license: str
    source: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("model_id", "revision", "license"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in dataclasses.asdict(self).items() if value is not None}


@dataclass(frozen=True)
class TensorAudit:
    """Metrics and hashes for one physical tensor and all its tied aliases."""

    name: str
    aliases: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    numel: int
    status: str
    base_sha256: str
    donor_sha256: str
    output_sha256: str
    donor_delta_l2: float
    update_l2: float
    cosine_update_to_donor_delta: float
    projection_on_donor_delta: float
    keep_ratio: float
    changed_elements: int
    base_nan_count: int
    donor_nan_count: int
    output_nan_count: int
    base_inf_count: int
    donor_inf_count: int
    output_inf_count: int

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["aliases"] = list(self.aliases)
        value["shape"] = list(self.shape)
        return value


@dataclass(frozen=True)
class AssimilationAudit:
    """Complete build audit; all metrics are computed without donor forward."""

    build_id: str
    created_at: str
    algorithm: str
    logical_tensor_count: int
    physical_tensor_count: int
    tied_groups: tuple[tuple[str, ...], ...]
    base_state_sha256: str
    donor_state_sha256: str
    output_state_sha256: str
    config_sha256: str | None
    tokenizer_sha256: str | None
    base_config_profile_sha256: str | None
    donor_config_profile_sha256: str | None
    base_tokenizer_profile_sha256: str | None
    donor_tokenizer_profile_sha256: str | None
    donor_forward_calls_build: int
    total_donor_delta_l2: float
    total_update_l2: float
    cosine_update_to_donor_delta: float
    projection_on_donor_delta: float
    keep_ratio: float
    base_nan_count: int
    donor_nan_count: int
    output_nan_count: int
    base_inf_count: int
    donor_inf_count: int
    output_inf_count: int
    tensors: tuple[TensorAudit, ...] = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["tied_groups"] = [list(group) for group in self.tied_groups]
        value["tensors"] = [item.to_dict() for item in self.tensors]
        return value


@dataclass
class AssimilationResult:
    """A new state dictionary plus its immutable audit and Lira manifest."""

    state_dict: dict[str, Tensor]
    audit: AssimilationAudit
    manifest: dict[str, Any]


@dataclass
class ModelRollback:
    """One-shot rollback handle returned after applying a result to a model."""

    model: nn.Module = field(repr=False)
    _backup: dict[str, Tensor] = field(repr=False)
    active: bool = True

    def rollback(self) -> None:
        if not self.active:
            raise AssimilationError("rollback handle is no longer active")
        self.model.load_state_dict(self._backup, strict=True)
        self.active = False
        self._backup.clear()

    def commit(self) -> None:
        """Accept the applied state and securely release the backup reference."""

        if not self.active:
            raise AssimilationError("rollback handle is no longer active")
        self.active = False
        self._backup.clear()


_VOLATILE_CONFIG_FIELDS = frozenset(
    {
        "_name_or_path",
        "name_or_path",
        "transformers_version",
        "torch_dtype",
        "tf_legacy_loss",
        # Generation/tokenizer defaults do not alter parameter geometry.  The
        # vocabulary ID mapping is validated independently and exactly.
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "sep_token_id",
        "decoder_start_token_id",
        "forced_bos_token_id",
        "forced_eos_token_id",
        # Frontend-only metadata emitted by some Hugging Face repositories.
        "transformers.js_config",
        "transformers_js_config",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_plain(value: Any) -> Any:
    """Convert common ML configuration values into deterministic JSON data."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _json_plain(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_plain(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_plain(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if hasattr(value, "__dict__") and value.__class__.__module__.startswith("tokenizers"):
        return str(value)
    return str(value)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _raw_config(config: Any) -> dict[str, Any]:
    if config is None:
        raise CompatibilityError("config is required for strict compatibility validation")
    if isinstance(config, Mapping):
        raw = dict(config)
    elif hasattr(config, "to_dict") and callable(config.to_dict):
        raw = dict(config.to_dict())
    else:
        raise CompatibilityError("config must be a mapping or expose to_dict()")
    return raw


def _normalise_config(config: Any) -> dict[str, Any]:
    raw = _raw_config(config)
    for key in _VOLATILE_CONFIG_FIELDS:
        raw.pop(key, None)
    # SmolLM2 releases written by different Transformers versions may omit
    # this field when it has the architectural default.  Missing and explicit
    # ``False`` are structurally identical; ``True`` remains incompatible.
    raw.setdefault("mlp_bias", False)
    return _json_plain(raw)


def canonical_config_fingerprint(config: Any) -> str:
    """Hash all structural config fields, excluding only runtime provenance."""

    return _sha256_json(_normalise_config(config))


def _normalise_tokenizer(tokenizer: Any) -> dict[str, Any]:
    """Return the weight-critical tokenizer contract: class plus token IDs.

    Chat templates, padding side, and configured BOS/EOS defaults can differ
    between a base and instruction-tuned sibling without changing which
    embedding/output row belongs to each token.  Their full profiles are still
    hashed in the audit so the difference is never hidden.
    """

    if tokenizer is None:
        raise CompatibilityError("tokenizer is required for strict compatibility validation")
    if isinstance(tokenizer, Mapping):
        raw = dict(tokenizer)
        if "vocab" in raw:
            vocab = raw["vocab"]
            tokenizer_class = raw.get("class", "mapping")
        elif all(isinstance(value, int) and not isinstance(value, bool) for value in raw.values()):
            vocab = raw
            tokenizer_class = "mapping"
        else:
            raise CompatibilityError(
                "tokenizer mappings must contain a 'vocab' mapping or be a raw token-to-id mapping"
            )
        if not isinstance(vocab, Mapping):
            raise CompatibilityError("tokenizer 'vocab' must be a mapping")
        return _json_plain({"class": tokenizer_class, "vocab": dict(vocab)})
    if not hasattr(tokenizer, "get_vocab") or not callable(tokenizer.get_vocab):
        raise CompatibilityError("tokenizer must be a mapping or expose get_vocab()")
    try:
        vocab = tokenizer.get_vocab()
    except Exception as exc:  # pragma: no cover - defensive around third-party tokenizers
        raise CompatibilityError(f"could not inspect tokenizer vocabulary: {exc}") from exc
    if not isinstance(vocab, Mapping):
        raise CompatibilityError("tokenizer.get_vocab() must return a mapping")
    return _json_plain({
        "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
        "vocab": dict(vocab),
    })


def _tokenizer_profile(tokenizer: Any) -> dict[str, Any]:
    """Capture non-structural tokenizer settings for provenance hashes."""

    if isinstance(tokenizer, Mapping):
        return _json_plain(dict(tokenizer))
    contract: dict[str, Any] = dict(_normalise_tokenizer(tokenizer))
    for attr in (
        "special_tokens_map",
        "added_tokens_encoder",
        "added_tokens_decoder",
        "model_max_length",
        "padding_side",
        "truncation_side",
        "clean_up_tokenization_spaces",
        "chat_template",
    ):
        if hasattr(tokenizer, attr):
            contract[attr] = getattr(tokenizer, attr)
    return _json_plain(contract)


def canonical_tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash the tokenizer class and exact token-to-ID vocabulary contract."""

    return _sha256_json(_normalise_tokenizer(tokenizer))


def _tensor_bytes(tensor: Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if value.numel() == 0:
        return b""
    return value.view(torch.uint8).numpy().tobytes(order="C")


def tensor_sha256(tensor: Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json_bytes(list(tensor.shape)))
    digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def state_dict_sha256(state_dict: Mapping[str, Tensor]) -> str:
    """Create a stable content hash independent of device and dict ordering."""

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        if not isinstance(key, str) or not isinstance(tensor, Tensor):
            raise CompatibilityError("state dictionaries must map string keys to tensors")
        key_bytes = key.encode("utf-8")
        digest.update(len(key_bytes).to_bytes(8, "little"))
        digest.update(key_bytes)
        digest.update(tensor_sha256(tensor).encode("ascii"))
    return digest.hexdigest()


def _storage_identity(tensor: Tensor) -> tuple[Any, ...]:
    if tensor.layout != torch.strided:
        return ("unsupported", id(tensor))
    storage = tensor.untyped_storage()
    return (
        tensor.device.type,
        tensor.device.index,
        storage.data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def _infer_tied_groups(state_dict: Mapping[str, Tensor]) -> tuple[tuple[str, ...], ...]:
    aliases: dict[tuple[Any, ...], list[str]] = {}
    for key in sorted(state_dict):
        aliases.setdefault(_storage_identity(state_dict[key]), []).append(key)
    return tuple(tuple(keys) for keys in aliases.values() if len(keys) > 1)


def _normalise_tied_groups(
    base: Mapping[str, Tensor],
    donor: Mapping[str, Tensor],
    tied_groups: Iterable[Iterable[str]] | None,
) -> tuple[tuple[str, ...], ...]:
    inferred_base = {frozenset(group) for group in _infer_tied_groups(base)}
    inferred_donor = {frozenset(group) for group in _infer_tied_groups(donor)}
    if tied_groups is None:
        if inferred_base != inferred_donor:
            raise CompatibilityError(
                "tied-weight topology differs between base and donor; pass explicit "
                "tied_groups only when the architecture contract proves the aliases"
            )
        groups = inferred_base
    else:
        groups = set()
        occupied: set[str] = set()
        for raw_group in tied_groups:
            group = frozenset(str(key) for key in raw_group)
            if len(group) < 2:
                raise CompatibilityError("each tied group must contain at least two keys")
            missing = group - base.keys()
            if missing:
                raise CompatibilityError(f"tied group contains unknown keys: {sorted(missing)!r}")
            overlap = occupied.intersection(group)
            if overlap:
                raise CompatibilityError(f"tied groups overlap at keys: {sorted(overlap)!r}")
            occupied.update(group)
            groups.add(group)
        # Never silently omit physical aliases detected in either model.
        if not inferred_base.issubset(groups) or not inferred_donor.issubset(groups):
            raise CompatibilityError("explicit tied_groups omit an inferred shared tensor")

    normalised = tuple(sorted((tuple(sorted(group)) for group in groups), key=lambda group: group[0]))
    for group in normalised:
        base_reference = base[group[0]]
        donor_reference = donor[group[0]]
        for alias in group[1:]:
            if not torch.equal(base_reference, base[alias]):
                raise CompatibilityError(f"base tied aliases do not contain equal values: {group!r}")
            if not torch.equal(donor_reference, donor[alias]):
                raise CompatibilityError(f"donor tied aliases do not contain equal values: {group!r}")
    return normalised


def _validate_state_dicts(base: Mapping[str, Tensor], donor: Mapping[str, Tensor], recipe: AssimilationRecipe) -> None:
    base_keys = set(base)
    donor_keys = set(donor)
    if base_keys != donor_keys:
        missing = sorted(base_keys - donor_keys)
        unexpected = sorted(donor_keys - base_keys)
        raise CompatibilityError(
            f"state-dict keys differ; missing_in_donor={missing!r}, unexpected_in_donor={unexpected!r}"
        )
    if not base_keys:
        raise CompatibilityError("state dictionaries cannot be empty")
    for key in sorted(base_keys):
        if not isinstance(key, str):
            raise CompatibilityError("state-dict keys must be strings")
        base_tensor = base[key]
        donor_tensor = donor[key]
        if not isinstance(base_tensor, Tensor) or not isinstance(donor_tensor, Tensor):
            raise CompatibilityError(f"state entry {key!r} is not a tensor")
        if base_tensor.layout != torch.strided or donor_tensor.layout != torch.strided:
            raise CompatibilityError(f"state entry {key!r} must be a dense strided tensor")
        if base_tensor.is_quantized or donor_tensor.is_quantized:
            raise CompatibilityError(f"quantized tensor {key!r} is not supported by Class A")
        if base_tensor.shape != donor_tensor.shape:
            raise CompatibilityError(
                f"shape mismatch for {key!r}: base={tuple(base_tensor.shape)}, donor={tuple(donor_tensor.shape)}"
            )
        if recipe.strict_dtype and base_tensor.dtype != donor_tensor.dtype:
            raise CompatibilityError(
                f"dtype mismatch for {key!r}: base={base_tensor.dtype}, donor={donor_tensor.dtype}"
            )
        if base_tensor.is_floating_point() != donor_tensor.is_floating_point():
            raise CompatibilityError(f"floating-point category mismatch for {key!r}")
        if not base_tensor.is_floating_point() and not torch.equal(base_tensor, donor_tensor):
            raise CompatibilityError(
                f"non-floating tensor {key!r} differs; direct DARE is undefined and policy=require_equal"
            )
        if recipe.fail_on_nonfinite and base_tensor.is_floating_point():
            if not bool(torch.isfinite(base_tensor).all().item()):
                raise CompatibilityError(f"base tensor {key!r} contains NaN or infinity")
            if not bool(torch.isfinite(donor_tensor).all().item()):
                raise CompatibilityError(f"donor tensor {key!r} contains NaN or infinity")


def _validate_contracts(
    recipe: AssimilationRecipe,
    base_config: Any,
    donor_config: Any,
    base_tokenizer: Any,
    donor_tokenizer: Any,
) -> tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
]:
    config_hash: str | None = None
    tokenizer_hash: str | None = None
    base_config_profile_hash: str | None = None
    donor_config_profile_hash: str | None = None
    base_tokenizer_profile_hash: str | None = None
    donor_tokenizer_profile_hash: str | None = None
    if recipe.require_config_match or base_config is not None or donor_config is not None:
        if base_config is None or donor_config is None:
            raise CompatibilityError("both base and donor configs are required when either is provided")
        base_hash = canonical_config_fingerprint(base_config)
        donor_hash = canonical_config_fingerprint(donor_config)
        if base_hash != donor_hash:
            raise CompatibilityError(
                f"config contracts differ: base_sha256={base_hash}, donor_sha256={donor_hash}"
            )
        config_hash = base_hash
        base_config_profile_hash = _sha256_json(_raw_config(base_config))
        donor_config_profile_hash = _sha256_json(_raw_config(donor_config))
    if recipe.require_tokenizer_match or base_tokenizer is not None or donor_tokenizer is not None:
        if base_tokenizer is None or donor_tokenizer is None:
            raise CompatibilityError("both base and donor tokenizers are required when either is provided")
        base_hash = canonical_tokenizer_fingerprint(base_tokenizer)
        donor_hash = canonical_tokenizer_fingerprint(donor_tokenizer)
        if base_hash != donor_hash:
            raise CompatibilityError(
                f"tokenizer contracts differ: base_sha256={base_hash}, donor_sha256={donor_hash}"
            )
        tokenizer_hash = base_hash
        base_tokenizer_profile_hash = _sha256_json(_tokenizer_profile(base_tokenizer))
        donor_tokenizer_profile_hash = _sha256_json(_tokenizer_profile(donor_tokenizer))
    return (
        config_hash,
        tokenizer_hash,
        base_config_profile_hash,
        donor_config_profile_hash,
        base_tokenizer_profile_hash,
        donor_tokenizer_profile_hash,
    )


def _seed_for_tensor(seed: int, name: str) -> int:
    payload = f"{seed}:{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def _nonfinite_counts(value: Tensor) -> tuple[int, int]:
    if not value.is_floating_point():
        return 0, 0
    detached = value.detach()
    return int(torch.isnan(detached).sum().item()), int(torch.isinf(detached).sum().item())


def _safe_cosine_and_projection(update: Tensor, delta: Tensor) -> tuple[float, float]:
    update_flat = update.detach().float().reshape(-1)
    delta_flat = delta.detach().float().reshape(-1)
    if update_flat.numel() == 0:
        return 0.0, 0.0
    dot = float(torch.dot(update_flat, delta_flat).item())
    update_norm = float(torch.linalg.vector_norm(update_flat).item())
    delta_norm = float(torch.linalg.vector_norm(delta_flat).item())
    cosine = dot / (update_norm * delta_norm) if update_norm and delta_norm else 0.0
    projection = dot / (delta_norm * delta_norm) if delta_norm else 0.0
    return cosine, projection


def dare_assimilate_tensor(
    base: Tensor,
    donor: Tensor,
    recipe: AssimilationRecipe,
    *,
    tensor_name: str,
) -> tuple[Tensor, dict[str, Any]]:
    """Assimilate a single tensor using deterministic DARE.

    The returned metrics dictionary is deliberately primitive/JSON-friendly so
    this function can also be used in focused experiments.
    """

    _validate_state_dicts({tensor_name: base}, {tensor_name: donor}, recipe)
    base_cpu = base.detach().cpu()
    donor_cpu = donor.detach().cpu()
    base_nan, base_inf = _nonfinite_counts(base_cpu)
    donor_nan, donor_inf = _nonfinite_counts(donor_cpu)

    if not base_cpu.is_floating_point():
        output = base_cpu.clone()
        output_nan, output_inf = 0, 0
        return output, {
            "status": "preserved_nonfloating",
            "donor_delta_l2": 0.0,
            "update_l2": 0.0,
            "cosine_update_to_donor_delta": 0.0,
            "projection_on_donor_delta": 0.0,
            "keep_ratio": 1.0,
            "changed_elements": 0,
            "base_nan_count": base_nan,
            "donor_nan_count": donor_nan,
            "output_nan_count": output_nan,
            "base_inf_count": base_inf,
            "donor_inf_count": donor_inf,
            "output_inf_count": output_inf,
        }

    calculation_dtype = torch.float64 if base_cpu.dtype == torch.float64 else torch.float32
    base_calc = base_cpu.to(calculation_dtype)
    donor_calc = donor_cpu.to(calculation_dtype)
    delta = donor_calc - base_calc
    keep_probability = 1.0 - recipe.drop_rate

    if recipe.drop_rate == 0.0:
        mask = torch.ones(base_calc.shape, dtype=torch.bool)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_seed_for_tensor(recipe.seed, tensor_name))
        mask = torch.rand(base_calc.shape, generator=generator, dtype=torch.float32) < keep_probability
    rescaled_delta = delta * mask.to(calculation_dtype) / keep_probability
    output_calc = base_calc + recipe.alpha * rescaled_delta
    output = output_calc.to(base_cpu.dtype)
    effective_update = output.to(calculation_dtype) - base_calc

    delta_l2 = float(torch.linalg.vector_norm(delta.float()).item()) if delta.numel() else 0.0
    update_l2 = float(torch.linalg.vector_norm(effective_update.float()).item()) if delta.numel() else 0.0
    cosine, projection = _safe_cosine_and_projection(effective_update, delta)
    output_nan, output_inf = _nonfinite_counts(output)
    if recipe.fail_on_nonfinite and (output_nan or output_inf):
        raise AssimilationError(
            f"assimilation output for {tensor_name!r} became non-finite after "
            f"dtype conversion: nan={output_nan}, inf={output_inf}"
        )
    changed = int(torch.ne(output, base_cpu).sum().item())
    keep_ratio = float(mask.float().mean().item()) if mask.numel() else 1.0
    return output, {
        "status": "assimilated",
        "donor_delta_l2": delta_l2,
        "update_l2": update_l2,
        "cosine_update_to_donor_delta": cosine,
        "projection_on_donor_delta": projection,
        "keep_ratio": keep_ratio,
        "changed_elements": changed,
        "base_nan_count": base_nan,
        "donor_nan_count": donor_nan,
        "output_nan_count": output_nan,
        "base_inf_count": base_inf,
        "donor_inf_count": donor_inf,
        "output_inf_count": output_inf,
    }


def _coerce_metadata(value: ModelMetadata | Mapping[str, Any] | None, role: str) -> ModelMetadata:
    if value is None:
        return ModelMetadata(
            model_id=f"{role}:local-in-memory",
            revision="unversioned",
            license="unknown",
            source="local",
        )
    if isinstance(value, ModelMetadata):
        return value
    if isinstance(value, Mapping):
        try:
            return ModelMetadata(**dict(value))
        except (TypeError, ValueError) as exc:
            raise CompatibilityError(f"invalid {role} metadata: {exc}") from exc
    raise CompatibilityError(f"{role} metadata must be ModelMetadata or a mapping")


def _build_manifest(
    *,
    recipe: AssimilationRecipe,
    audit: AssimilationAudit,
    base_metadata: ModelMetadata,
    donor_metadata: ModelMetadata,
) -> dict[str, Any]:
    unresolved_license = any(
        item.license.strip().lower() in {"unknown", "unspecified", "n/a"}
        for item in (base_metadata, donor_metadata)
    )
    return {
        "format": LIRA_MANIFEST_FORMAT,
        "schema_version": LIRA_MANIFEST_VERSION,
        "status": "experimental",
        "build_id": audit.build_id,
        "created_at": audit.created_at,
        "method": {
            "class": METHOD_CLASS,
            "algorithm": recipe.algorithm,
            "formula": "output = base + alpha * DARE(donor - base)",
            "homologous_models_required": True,
            "donor_forward_calls_build": audit.donor_forward_calls_build,
            "does_not_use": [
                "donor_forward_pass",
                "donor_logits",
                "teacher_outputs",
                "prompts",
                "activations",
                "hidden_chain_of_thought",
            ],
        },
        "recipe": dataclasses.asdict(recipe),
        "provenance": {
            "base": {
                **base_metadata.to_dict(),
                "state_sha256": audit.base_state_sha256,
                "config_sha256": audit.config_sha256,
                "tokenizer_sha256": audit.tokenizer_sha256,
            },
            "donor": {
                **donor_metadata.to_dict(),
                "state_sha256": audit.donor_state_sha256,
                "config_sha256": audit.config_sha256,
                "tokenizer_sha256": audit.tokenizer_sha256,
            },
            "output": {
                "state_sha256": audit.output_state_sha256,
            },
        },
        "compatibility": {
            "state_keys_and_shapes": "exact",
            "dtype": "exact" if recipe.strict_dtype else "floating-category",
            "config_sha256": audit.config_sha256,
            "tokenizer_sha256": audit.tokenizer_sha256,
            "tied_groups": [list(group) for group in audit.tied_groups],
            "non_structural_profiles": {
                "config": {
                    "base_sha256": audit.base_config_profile_sha256,
                    "donor_sha256": audit.donor_config_profile_sha256,
                    "different": audit.base_config_profile_sha256
                    != audit.donor_config_profile_sha256,
                    "excluded_from_weight_contract": sorted(_VOLATILE_CONFIG_FIELDS),
                    "canonical_defaults": {"mlp_bias": False},
                },
                "tokenizer": {
                    "base_sha256": audit.base_tokenizer_profile_sha256,
                    "donor_sha256": audit.donor_tokenizer_profile_sha256,
                    "different": audit.base_tokenizer_profile_sha256
                    != audit.donor_tokenizer_profile_sha256,
                    "weight_contract": "tokenizer class plus exact token-to-id vocabulary",
                },
            },
        },
        "license_review": {
            "base_license": base_metadata.license,
            "donor_license": donor_metadata.license,
            "operator_must_verify_redistribution_and_derivative_terms": True,
            "unresolved_license": unresolved_license,
        },
        "audit_sha256": _sha256_json(audit.to_dict()),
        "artifacts": {},
        "scientific_scope": (
            "Direct parameter-space transformation. It does not decode a model's thoughts, "
            "recover private training examples, or guarantee retention of donor capabilities."
        ),
    }


def assimilate_state_dict(
    base_state: Mapping[str, Tensor],
    donor_state: Mapping[str, Tensor],
    recipe: AssimilationRecipe | None = None,
    *,
    tied_groups: Iterable[Iterable[str]] | None = None,
    base_config: Any = None,
    donor_config: Any = None,
    base_tokenizer: Any = None,
    donor_tokenizer: Any = None,
    base_metadata: ModelMetadata | Mapping[str, Any] | None = None,
    donor_metadata: ModelMetadata | Mapping[str, Any] | None = None,
    donor_forward_calls_build: int = 0,
) -> AssimilationResult:
    """Build a new state from base and donor tensors without model execution."""

    active_recipe = recipe or AssimilationRecipe()
    if donor_forward_calls_build != 0:
        raise DonorForwardProhibited("Class A requires donor_forward_calls_build == 0")
    _validate_state_dicts(base_state, donor_state, active_recipe)
    (
        config_hash,
        tokenizer_hash,
        base_config_profile_hash,
        donor_config_profile_hash,
        base_tokenizer_profile_hash,
        donor_tokenizer_profile_hash,
    ) = _validate_contracts(
        active_recipe,
        base_config,
        donor_config,
        base_tokenizer,
        donor_tokenizer,
    )
    groups = _normalise_tied_groups(base_state, donor_state, tied_groups)
    alias_to_canonical: dict[str, str] = {}
    canonical_to_aliases: dict[str, tuple[str, ...]] = {}
    for group in groups:
        canonical_to_aliases[group[0]] = group
        for alias in group:
            alias_to_canonical[alias] = group[0]

    output_state: dict[str, Tensor] = {}
    tensor_audits: list[TensorAudit] = []
    processed: set[str] = set()
    weighted_kept = 0.0
    weighted_keep_count = 0
    global_delta_sq = 0.0
    global_update_sq = 0.0
    global_dot = 0.0

    for key in sorted(base_state):
        canonical = alias_to_canonical.get(key, key)
        if canonical in processed:
            continue
        processed.add(canonical)
        aliases = canonical_to_aliases.get(canonical, (canonical,))
        output, metrics = dare_assimilate_tensor(
            base_state[canonical], donor_state[canonical], active_recipe, tensor_name=canonical
        )
        for alias in aliases:
            output_state[alias] = output

        delta_l2 = float(metrics["donor_delta_l2"])
        update_l2 = float(metrics["update_l2"])
        cosine = float(metrics["cosine_update_to_donor_delta"])
        dot = cosine * delta_l2 * update_l2 if delta_l2 and update_l2 else 0.0
        global_delta_sq += delta_l2 * delta_l2
        global_update_sq += update_l2 * update_l2
        global_dot += dot
        numel = base_state[canonical].numel()
        if base_state[canonical].is_floating_point():
            weighted_kept += float(metrics["keep_ratio"]) * numel
            weighted_keep_count += numel
        tensor_audits.append(
            TensorAudit(
                name=canonical,
                aliases=aliases,
                shape=tuple(base_state[canonical].shape),
                dtype=str(base_state[canonical].dtype),
                numel=numel,
                status=str(metrics["status"]),
                base_sha256=tensor_sha256(base_state[canonical]),
                donor_sha256=tensor_sha256(donor_state[canonical]),
                output_sha256=tensor_sha256(output),
                donor_delta_l2=delta_l2,
                update_l2=update_l2,
                cosine_update_to_donor_delta=cosine,
                projection_on_donor_delta=float(metrics["projection_on_donor_delta"]),
                keep_ratio=float(metrics["keep_ratio"]),
                changed_elements=int(metrics["changed_elements"]),
                base_nan_count=int(metrics["base_nan_count"]),
                donor_nan_count=int(metrics["donor_nan_count"]),
                output_nan_count=int(metrics["output_nan_count"]),
                base_inf_count=int(metrics["base_inf_count"]),
                donor_inf_count=int(metrics["donor_inf_count"]),
                output_inf_count=int(metrics["output_inf_count"]),
            )
        )

    # Preserve input key insertion order for conventional state-dict consumers.
    output_state = {key: output_state[key] for key in base_state}
    total_delta = math.sqrt(global_delta_sq)
    total_update = math.sqrt(global_update_sq)
    global_cosine = global_dot / (total_delta * total_update) if total_delta and total_update else 0.0
    global_projection = global_dot / global_delta_sq if global_delta_sq else 0.0
    created_at = _utc_now()
    audit = AssimilationAudit(
        build_id=str(uuid.uuid4()),
        created_at=created_at,
        algorithm=active_recipe.algorithm,
        logical_tensor_count=len(base_state),
        physical_tensor_count=len(tensor_audits),
        tied_groups=groups,
        base_state_sha256=state_dict_sha256(base_state),
        donor_state_sha256=state_dict_sha256(donor_state),
        output_state_sha256=state_dict_sha256(output_state),
        config_sha256=config_hash,
        tokenizer_sha256=tokenizer_hash,
        base_config_profile_sha256=base_config_profile_hash,
        donor_config_profile_sha256=donor_config_profile_hash,
        base_tokenizer_profile_sha256=base_tokenizer_profile_hash,
        donor_tokenizer_profile_sha256=donor_tokenizer_profile_hash,
        donor_forward_calls_build=0,
        total_donor_delta_l2=total_delta,
        total_update_l2=total_update,
        cosine_update_to_donor_delta=global_cosine,
        projection_on_donor_delta=global_projection,
        keep_ratio=weighted_kept / weighted_keep_count if weighted_keep_count else 1.0,
        base_nan_count=sum(item.base_nan_count for item in tensor_audits),
        donor_nan_count=sum(item.donor_nan_count for item in tensor_audits),
        output_nan_count=sum(item.output_nan_count for item in tensor_audits),
        base_inf_count=sum(item.base_inf_count for item in tensor_audits),
        donor_inf_count=sum(item.donor_inf_count for item in tensor_audits),
        output_inf_count=sum(item.output_inf_count for item in tensor_audits),
        tensors=tuple(tensor_audits),
    )
    base_meta = _coerce_metadata(base_metadata, "base")
    donor_meta = _coerce_metadata(donor_metadata, "donor")
    manifest = _build_manifest(
        recipe=active_recipe,
        audit=audit,
        base_metadata=base_meta,
        donor_metadata=donor_meta,
    )
    return AssimilationResult(state_dict=output_state, audit=audit, manifest=manifest)


@contextlib.contextmanager
def _forbid_donor_forward(donor: nn.Module) -> Iterator[MutableMapping[str, int]]:
    """Monkey-patch a donor instance so accidental execution fails closed."""

    calls: MutableMapping[str, int] = {"count": 0}
    had_instance_forward = "forward" in donor.__dict__
    original_instance_forward = donor.__dict__.get("forward")

    def prohibited(*args: Any, **kwargs: Any) -> Any:
        calls["count"] += 1
        raise DonorForwardProhibited("the donor forward pass is prohibited during Class A assimilation")

    donor.forward = prohibited  # type: ignore[method-assign]
    try:
        yield calls
    finally:
        if had_instance_forward:
            donor.forward = original_instance_forward  # type: ignore[method-assign,assignment]
        else:
            # Restore normal descriptor lookup instead of retaining a bound
            # method that would create a donor -> bound-method -> donor cycle.
            delattr(donor, "forward")


def _module_tied_groups(module: nn.Module) -> tuple[tuple[str, ...], ...]:
    # ``remove_duplicate=False`` is available on supported modern PyTorch.
    entries: list[tuple[str, Tensor]] = list(module.named_parameters(remove_duplicate=False))
    entries.extend(module.named_buffers(remove_duplicate=False))
    identities: dict[int, list[str]] = {}
    for name, value in entries:
        identities.setdefault(id(value), []).append(name)
    return tuple(tuple(sorted(names)) for names in identities.values() if len(names) > 1)


def assimilate_models(
    base_model: nn.Module,
    donor_model: nn.Module,
    recipe: AssimilationRecipe | None = None,
    *,
    base_config: Any = None,
    donor_config: Any = None,
    base_tokenizer: Any = None,
    donor_tokenizer: Any = None,
    base_metadata: ModelMetadata | Mapping[str, Any] | None = None,
    donor_metadata: ModelMetadata | Mapping[str, Any] | None = None,
) -> AssimilationResult:
    """Assimilate already-loaded models while actively forbidding donor forward."""

    if not isinstance(base_model, nn.Module) or not isinstance(donor_model, nn.Module):
        raise TypeError("base_model and donor_model must be torch.nn.Module instances")
    active_recipe = recipe or AssimilationRecipe()
    resolved_base_config = base_config if base_config is not None else getattr(base_model, "config", None)
    resolved_donor_config = donor_config if donor_config is not None else getattr(donor_model, "config", None)
    base_groups = {frozenset(group) for group in _module_tied_groups(base_model)}
    donor_groups = {frozenset(group) for group in _module_tied_groups(donor_model)}
    if base_groups != donor_groups:
        raise CompatibilityError("model tied-weight topology differs between base and donor")
    explicit_groups = tuple(tuple(sorted(group)) for group in sorted(base_groups, key=lambda item: sorted(item)[0]))

    with _forbid_donor_forward(donor_model) as calls:
        # ``state_dict`` only reads registered parameters/buffers.  The guard
        # turns any surprising hook that attempts a forward into a hard error.
        donor_state = donor_model.state_dict()
        result = assimilate_state_dict(
            base_model.state_dict(),
            donor_state,
            active_recipe,
            tied_groups=explicit_groups,
            base_config=resolved_base_config,
            donor_config=resolved_donor_config,
            base_tokenizer=base_tokenizer,
            donor_tokenizer=donor_tokenizer,
            base_metadata=base_metadata,
            donor_metadata=donor_metadata,
            donor_forward_calls_build=calls["count"],
        )
    if calls["count"] != 0:  # defensive; a forward would already have raised
        raise DonorForwardProhibited("donor was executed during assimilation")
    return result


def apply_result_to_model(model: nn.Module, result: AssimilationResult) -> ModelRollback:
    """Apply a result transactionally and return a one-shot rollback handle."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    current = model.state_dict()
    validation_recipe = AssimilationRecipe(
        alpha=0.0,
        drop_rate=0.0,
        require_config_match=False,
        require_tokenizer_match=False,
    )
    _validate_state_dicts(current, result.state_dict, validation_recipe)
    backup = {key: value.detach().cpu().clone() for key, value in current.items()}
    try:
        model.load_state_dict(result.state_dict, strict=True)
    except Exception as exc:
        model.load_state_dict(backup, strict=True)
        raise AssimilationError(f"could not apply result; original state restored: {exc}") from exc
    return ModelRollback(model=model, _backup=backup)


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_config(config: Any, output_dir: Path) -> None:
    if config is None:
        return
    path = output_dir / "config.json"
    if isinstance(config, Mapping):
        _write_json(path, _json_plain(dict(config)))
    elif hasattr(config, "to_json_file") and callable(config.to_json_file):
        config.to_json_file(str(path), use_diff=False)
    elif hasattr(config, "to_dict") and callable(config.to_dict):
        _write_json(path, _json_plain(config.to_dict()))
    else:
        raise AssimilationError("config must be a mapping or expose to_dict()/to_json_file()")


def save_standalone(
    result: AssimilationResult,
    output_dir: str | os.PathLike[str],
    *,
    config: Any = None,
    tokenizer: Any = None,
    overwrite: bool = False,
) -> Path:
    """Atomically save standalone safetensors, config/tokenizer, audit and Lira manifest."""

    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise AssimilationError("saving requires the 'safetensors' package") from exc

    destination = Path(output_dir).expanduser().resolve()
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output directory already exists: {destination}")

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
    backup: Path | None = None
    try:
        # Clone every logical key: safetensors intentionally rejects shared
        # storage, while the manifest retains the exact tied-group topology.
        serializable = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in result.state_dict.items()
        }
        weights_path = temporary / "model.safetensors"
        save_file(
            serializable,
            str(weights_path),
            metadata={
                "format": "pt",
                "method": METHOD_CLASS,
                "output_state_sha256": result.audit.output_state_sha256,
            },
        )
        _save_config(config, temporary)
        if tokenizer is not None:
            if not hasattr(tokenizer, "save_pretrained") or not callable(tokenizer.save_pretrained):
                raise AssimilationError("tokenizer must expose save_pretrained(output_dir)")
            tokenizer.save_pretrained(str(temporary))

        _write_json(temporary / "audit.json", result.audit.to_dict())
        manifest = copy.deepcopy(result.manifest)
        artifacts: dict[str, Any] = {}
        for artifact in sorted(path for path in temporary.iterdir() if path.is_file()):
            artifacts[artifact.name] = {
                "sha256": _file_sha256(artifact),
                "bytes": artifact.stat().st_size,
            }
        manifest["artifacts"] = artifacts
        _write_json(temporary / "lira_manifest.json", manifest)

        if destination.exists():
            backup = parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
            os.replace(destination, backup)
        os.replace(temporary, destination)
        if backup is not None:
            shutil.rmtree(backup)
        result.manifest = manifest
        return destination
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
