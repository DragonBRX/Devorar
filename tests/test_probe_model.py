from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import probe_model


class ProbePureFunctionTests(unittest.TestCase):
    def test_classifies_structural_roles_and_layers(self):
        self.assertEqual(
            probe_model.parameter_role("model.layers.4.self_attn.q_proj.weight"),
            "attention_query",
        )
        self.assertEqual(
            probe_model.parameter_role("model.layers.2.mlp.down_proj.weight"),
            "mlp_contraction",
        )
        self.assertEqual(probe_model.parameter_role("model.embed_tokens.weight"), "token_embedding")
        self.assertEqual(probe_model.parameter_layer("model.layers.17.mlp.up_proj.weight"), 17)
        self.assertIsNone(probe_model.parameter_layer("lm_head.weight"))

    def test_unravels_flat_indices_without_torch(self):
        self.assertEqual(probe_model.unravel_index(0, [2, 3]), [0, 0])
        self.assertEqual(probe_model.unravel_index(5, [2, 3]), [1, 2])
        self.assertEqual(probe_model.unravel_index(0, []), [])
        with self.assertRaises(probe_model.ProbeError):
            probe_model.unravel_index(6, [2, 3])

    def test_aggregates_prompt_conditioned_scores(self):
        scores = [
            {
                "layer": 0,
                "role": "attention_query",
                "numel": 2,
                "absolute_attribution_total": 4.0,
                "signed_attribution_total": 1.0,
            },
            {
                "layer": 0,
                "role": "mlp_contraction",
                "numel": 6,
                "absolute_attribution_total": 2.0,
                "signed_attribution_total": -1.0,
            },
            {
                "layer": None,
                "role": "output_vocabulary_head",
                "numel": 2,
                "absolute_attribution_total": 3.0,
                "signed_attribution_total": 0.5,
            },
        ]
        layers, roles = probe_model.aggregate_scores(scores)
        layer_zero = next(item for item in layers if item["group"] == "layer_0")
        self.assertEqual(layer_zero["tensor_count"], 2)
        self.assertEqual(layer_zero["parameter_values"], 8)
        self.assertEqual(layer_zero["absolute_attribution_total"], 6.0)
        self.assertEqual(layer_zero["absolute_attribution_mean"], 0.75)
        self.assertEqual(layer_zero["signed_attribution_mean"], 0.0)
        self.assertEqual(roles[0]["group"], "attention_query")

    def test_taylor_prediction_matches_exact_attenuation_direction(self):
        # For w'=(1-f)w, Delta-w=-f*w and first-order Delta-L=-f*sum(grad*w).
        predicted = probe_model.taylor_predicted_loss_delta(
            signed_attribution_total=2.5,
            fraction=0.2,
        )
        self.assertEqual(predicted, -0.5)
        comparison = probe_model.compare_loss_deltas(predicted=predicted, observed=-0.45)
        self.assertAlmostEqual(comparison["taylor_prediction_error"], 0.05)
        self.assertTrue(comparison["taylor_observed_direction_agreement"])
        self.assertGreaterEqual(comparison["taylor_relative_error_to_max_magnitude"], 0.0)

    def test_taylor_comparison_does_not_invent_direction_near_zero(self):
        comparison = probe_model.compare_loss_deltas(predicted=0.0, observed=0.25)
        self.assertIsNone(comparison["taylor_observed_direction_agreement"])
        with self.assertRaises(probe_model.ProbeError):
            probe_model.taylor_predicted_loss_delta(
                signed_attribution_total=float("nan"),
                fraction=0.05,
            )

    def test_selects_exact_tensor_targets_in_requested_order(self):
        scores = [{"name": "layer.a"}, {"name": "layer.b"}, {"name": "layer.c"}]
        selected = probe_model.select_intervention_scores(
            scores,
            exact_targets=["layer.c", "layer.a"],
            default_limit=1,
        )
        self.assertEqual([item["name"] for item in selected], ["layer.c", "layer.a"])
        fallback = probe_model.select_intervention_scores(
            scores,
            exact_targets=[],
            default_limit=2,
        )
        self.assertEqual([item["name"] for item in fallback], ["layer.a", "layer.b"])

    def test_rejects_missing_or_duplicate_exact_tensor_targets(self):
        scores = [{"name": "layer.a"}]
        with self.assertRaises(probe_model.ProbeError):
            probe_model.select_intervention_scores(
                scores,
                exact_targets=["layer.missing"],
                default_limit=1,
            )
        with self.assertRaises(probe_model.ProbeError):
            probe_model.select_intervention_scores(
                scores,
                exact_targets=["layer.a", "layer.a"],
                default_limit=1,
            )
        with self.assertRaises(probe_model.ProbeError):
            probe_model.validate_exact_targets([f"layer.{index}" for index in range(9)])

    def test_atomic_report_is_noncanonical_and_utf8(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "probe.lira.json"
            report = {
                "format": probe_model.PROBE_FORMAT,
                "format_version": probe_model.PROBE_VERSION,
                "canonical_lira": False,
                "text": "parâmetro",
            }
            probe_model._write_json_atomic(destination, report)
            loaded = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(loaded["text"], "parâmetro")
            self.assertFalse(loaded["canonical_lira"])


class ProbeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("probe_model.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_probe_has_no_donor_loader_or_hidden_reasoning_claim(self):
        self.assertNotIn("DONOR_MODEL", self.source)
        self.assertIn('"hidden_chain_of_thought_accessed": False', self.source)
        self.assertIn('"attribution_is_prompt_dependent": True', self.source)
        self.assertIn('"coordinate_proxy_is_an_intervention": False', self.source)
        self.assertIn('"tensor_intervention_proves_fixed_semantics": False', self.source)
        self.assertIn("checkpoint_restored_exactly", self.source)

    def test_coordinate_proxy_and_tensor_intervention_are_separate(self):
        self.assertIn("unverified_first_order_coordinate_proxy", self.source)
        self.assertIn("observed_local_whole_tensor_intervention", self.source)
        self.assertIn('"taylor_predicted_loss_delta"', self.source)
        self.assertIn('"observed_loss_delta"', self.source)
        self.assertNotIn('"causal_checks"', self.source)

    def test_default_probe_is_bounded(self):
        args = probe_model.build_parser().parse_args([])
        self.assertLessEqual(args.max_input_tokens, 512)
        self.assertLessEqual(args.max_new_tokens, 32)
        self.assertLessEqual(args.intervention_tensors, 4)
        self.assertEqual(args.target_tensor, [])
        self.assertLessEqual(args.top_weights_per_tensor, 3)
        self.assertGreater(args.ablation_fraction, 0.0)
        self.assertLessEqual(args.ablation_fraction, 0.05)


if __name__ == "__main__":
    unittest.main()
