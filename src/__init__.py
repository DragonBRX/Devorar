"""Public API for Devorar, with ML dependencies loaded only when requested."""

from __future__ import annotations

from typing import Any


_ENGINE_EXPORTS = frozenset(
    {
        "AssimilationAudit",
        "AssimilationError",
        "AssimilationRecipe",
        "AssimilationResult",
        "CompatibilityError",
        "DonorForwardProhibited",
        "ModelMetadata",
        "ModelRollback",
        "TensorAudit",
        "apply_result_to_model",
        "assimilate_models",
        "assimilate_state_dict",
        "canonical_config_fingerprint",
        "canonical_tokenizer_fingerprint",
        "dare_assimilate_tensor",
        "save_standalone",
        "state_dict_sha256",
    }
)


def __getattr__(name: str) -> Any:
    if name not in _ENGINE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import devorar as engine

    value = getattr(engine, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _ENGINE_EXPORTS)


__all__ = sorted(_ENGINE_EXPORTS)
