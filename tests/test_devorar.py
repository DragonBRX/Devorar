from __future__ import annotations

import copy
import gc
import json
import weakref
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from src.devorar import (
    AssimilationError,
    AssimilationRecipe,
    CompatibilityError,
    ModelMetadata,
    apply_result_to_model,
    assimilate_models,
    assimilate_state_dict,
    dare_assimilate_tensor,
    save_standalone,
)


CONFIG = {"model_type": "tiny", "hidden_size": 4, "vocab_size": 7}
TOKENIZER = {
    "vocab": {"<pad>": 0, "a": 1, "b": 2},
    "special_tokens_map": {"pad_token": "<pad>"},
}
BASE_META = ModelMetadata("tests/base", "base-commit", "MIT", "memory")
DONOR_META = ModelMetadata("tests/donor", "donor-commit", "Apache-2.0", "memory")


def strict_recipe(**changes: object) -> AssimilationRecipe:
    values = {
        "alpha": 0.5,
        "drop_rate": 0.25,
        "seed": 12345,
        "require_config_match": True,
        "require_tokenizer_match": True,
    }
    values.update(changes)
    return AssimilationRecipe(**values)


class TinyModel(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.config = dict(CONFIG)
        self.forward_calls = 0
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters()):
                values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
                parameter.copy_(values + offset + index)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.linear(value)


class TiedModel(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(7, 4)
        self.lm_head = nn.Linear(4, 7, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.config = dict(CONFIG)
        with torch.no_grad():
            values = torch.arange(28, dtype=torch.float32).reshape(7, 4)
            self.embedding.weight.copy_(values + offset)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.embedding(ids))


def assimilate_tiny(base: nn.Module, donor: nn.Module, recipe: AssimilationRecipe | None = None):
    return assimilate_models(
        base,
        donor,
        recipe or strict_recipe(),
        base_tokenizer=TOKENIZER,
        donor_tokenizer=copy.deepcopy(TOKENIZER),
        base_metadata=BASE_META,
        donor_metadata=DONOR_META,
    )


def test_dare_is_deterministic_tensor_by_tensor_and_result_is_distinct() -> None:
    base = TinyModel(0.0)
    donor = TinyModel(10.0)
    first = assimilate_tiny(base, donor)
    second = assimilate_tiny(base, donor)

    assert first.state_dict.keys() == second.state_dict.keys()
    assert all(torch.equal(first.state_dict[key], second.state_dict[key]) for key in first.state_dict)
    assert any(not torch.equal(first.state_dict[key], base.state_dict()[key]) for key in first.state_dict)
    assert first.audit.output_state_sha256 == second.audit.output_state_sha256
    assert first.audit.keep_ratio == pytest.approx(second.audit.keep_ratio)
    assert 0.0 <= first.audit.keep_ratio <= 1.0
    assert first.audit.total_update_l2 > 0


def test_donor_forward_is_guarded_and_never_called() -> None:
    base = TinyModel(0.0)
    donor = TinyModel(2.0)
    original_bound_forward = donor.forward

    result = assimilate_tiny(base, donor)

    assert donor.forward_calls == 0
    assert result.audit.donor_forward_calls_build == 0
    assert donor.forward.__func__ is original_bound_forward.__func__
    assert "donor_forward_pass" in result.manifest["method"]["does_not_use"]

    donor_reference = weakref.ref(donor)
    del donor
    del original_bound_forward
    gc.collect()
    assert donor_reference() is None
    # Keeping the assimilation result alive must not keep the donor alive.
    assert result.audit.donor_forward_calls_build == 0


def test_tied_weights_are_assimilated_once_and_alias_the_same_output() -> None:
    base = TiedModel(0.0)
    donor = TiedModel(5.0)

    result = assimilate_tiny(base, donor, strict_recipe(drop_rate=0.0))

    group = ("embedding.weight", "lm_head.weight")
    assert group in result.audit.tied_groups
    assert result.audit.logical_tensor_count == 2
    assert result.audit.physical_tensor_count == 1
    assert result.state_dict[group[0]] is result.state_dict[group[1]]
    assert len(result.audit.tensors) == 1
    assert result.audit.tensors[0].aliases == group


@pytest.mark.parametrize(
    ("base", "donor", "message"),
    [
        ({"a": torch.zeros(2)}, {"b": torch.zeros(2)}, "keys differ"),
        ({"a": torch.zeros(2)}, {"a": torch.zeros(3)}, "shape mismatch"),
        ({"a": torch.zeros(2)}, {"a": torch.zeros(2, dtype=torch.float64)}, "dtype mismatch"),
    ],
)
def test_incompatible_state_is_rejected(base, donor, message: str) -> None:
    recipe = AssimilationRecipe(require_config_match=False, require_tokenizer_match=False)
    with pytest.raises(CompatibilityError, match=message):
        assimilate_state_dict(base, donor, recipe)


def test_config_and_tokenizer_mismatches_are_rejected() -> None:
    base = TinyModel(0.0)
    donor = TinyModel(1.0)
    donor.config["hidden_size"] = 8
    with pytest.raises(CompatibilityError, match="config contracts differ"):
        assimilate_tiny(base, donor)

    donor.config = dict(CONFIG)
    bad_tokenizer = copy.deepcopy(TOKENIZER)
    bad_tokenizer["vocab"]["x"] = 9
    with pytest.raises(CompatibilityError, match="tokenizer contracts differ"):
        assimilate_models(
            base,
            donor,
            strict_recipe(),
            base_tokenizer=TOKENIZER,
            donor_tokenizer=bad_tokenizer,
            base_metadata=BASE_META,
            donor_metadata=DONOR_META,
        )


def test_weight_contract_accepts_and_records_nonstructural_sibling_differences() -> None:
    base = TinyModel(0.0)
    donor = TinyModel(1.0)
    base.config.update({"bos_token_id": 0, "eos_token_id": 0})
    donor.config.update(
        {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 2,
            "mlp_bias": False,
            "transformers.js_config": {"dtype": "q4"},
        }
    )
    base_tokenizer = copy.deepcopy(TOKENIZER)
    donor_tokenizer = copy.deepcopy(TOKENIZER)
    base_tokenizer.update({"padding_side": "right", "special_tokens_map": {"bos": "<pad>"}})
    donor_tokenizer.update(
        {
            "padding_side": "left",
            "special_tokens_map": {"bos": "a", "eos": "b"},
            "chat_template": "{{ messages }}",
        }
    )

    result = assimilate_models(
        base,
        donor,
        strict_recipe(drop_rate=0.0),
        base_tokenizer=base_tokenizer,
        donor_tokenizer=donor_tokenizer,
        base_metadata=BASE_META,
        donor_metadata=DONOR_META,
    )

    profiles = result.manifest["compatibility"]["non_structural_profiles"]
    assert profiles["config"]["different"] is True
    assert profiles["tokenizer"]["different"] is True
    assert result.audit.config_sha256
    assert result.audit.tokenizer_sha256


def test_nonfinite_metrics_exist_and_default_policy_rejects_nan() -> None:
    base = torch.tensor([0.0, float("nan")])
    donor = torch.tensor([1.0, 2.0])
    strict = AssimilationRecipe(require_config_match=False, require_tokenizer_match=False)
    with pytest.raises(CompatibilityError, match="NaN or infinity"):
        dare_assimilate_tensor(base, donor, strict, tensor_name="x")

    inspect_recipe = AssimilationRecipe(
        alpha=0.5,
        drop_rate=0.0,
        require_config_match=False,
        require_tokenizer_match=False,
        fail_on_nonfinite=False,
    )
    _, metrics = dare_assimilate_tensor(base, donor, inspect_recipe, tensor_name="x")
    assert metrics["base_nan_count"] == 1
    assert metrics["output_nan_count"] == 1
    assert "donor_inf_count" in metrics


def test_default_policy_rejects_overflow_created_by_dare_rescaling() -> None:
    base = torch.full((10_000,), -60_000.0, dtype=torch.float16)
    donor = torch.full((10_000,), 60_000.0, dtype=torch.float16)
    recipe = AssimilationRecipe(
        alpha=1.0,
        drop_rate=0.99,
        seed=7,
        require_config_match=False,
        require_tokenizer_match=False,
    )
    with pytest.raises(AssimilationError, match="became non-finite"):
        dare_assimilate_tensor(base, donor, recipe, tensor_name="overflow")


def test_lira_manifest_has_hashes_revisions_licenses_and_scientific_boundary() -> None:
    result = assimilate_tiny(TinyModel(0.0), TinyModel(3.0))
    manifest = result.manifest

    assert manifest["format"] == "lira.experimental.parametric-assimilation"
    assert manifest["status"] == "experimental"
    assert manifest["provenance"]["base"]["revision"] == "base-commit"
    assert manifest["provenance"]["donor"]["revision"] == "donor-commit"
    assert manifest["provenance"]["base"]["license"] == "MIT"
    assert manifest["provenance"]["donor"]["license"] == "Apache-2.0"
    assert len(manifest["provenance"]["base"]["state_sha256"]) == 64
    assert len(manifest["provenance"]["output"]["state_sha256"]) == 64
    assert "does not decode" in manifest["scientific_scope"]


def test_apply_is_transactional_and_rollback_restores_exact_state() -> None:
    base = TinyModel(0.0)
    donor = TinyModel(4.0)
    before = {key: value.clone() for key, value in base.state_dict().items()}
    result = assimilate_tiny(base, donor, strict_recipe(drop_rate=0.0))

    rollback = apply_result_to_model(base, result)
    assert any(not torch.equal(base.state_dict()[key], before[key]) for key in before)
    rollback.rollback()
    assert all(torch.equal(base.state_dict()[key], before[key]) for key in before)
    assert rollback.active is False


def test_save_standalone_writes_safetensors_and_auditable_artifacts(tmp_path: Path) -> None:
    result = assimilate_tiny(TiedModel(0.0), TiedModel(2.0), strict_recipe(drop_rate=0.0))
    output = save_standalone(result, tmp_path / "artifact", config=CONFIG)

    weights = load_file(str(output / "model.safetensors"))
    assert set(weights) == set(result.state_dict)
    assert torch.equal(weights["embedding.weight"], result.state_dict["embedding.weight"])
    manifest = json.loads((output / "lira_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["model.safetensors"]["sha256"]
    assert manifest["artifacts"]["audit.json"]["bytes"] > 0
    assert (output / "config.json").is_file()
