import unittest
from types import SimpleNamespace
import tempfile
from pathlib import Path

import tensorflow as tf

from inputs.adcsp.schema import ExposureContext, UserProfile
from inputs.adcsp.synthetic_trajectory import SyntheticTrajectoryConfig, SyntheticTrajectoryGenerator
from models.adcsp.generator import AdCSPPPathGenerator
from models.adcsp.flow_prior import AdCSPPFlowPrior
from models.adcsp.state import UserStateBuilder
from models.adcsp.model import _append_state_history


def context(index: int) -> ExposureContext:
    return ExposureContext(
        impression_id=f"imp-{index}", event_timestamp=100 + index * 60,
        slot_id=f"slot-{index}", features={
            "slot_form": "feed", "indu_name": "commerce", "objective_type": "purchase",
            "promotion_target": "purchase", "slot_quality_score": 0.9,
        },
    )


class AlwaysEngageModel:
    def differentiable_probabilities(self, state, request, click_window_seconds, conversion_window_seconds):
        return tf.constant(1.0), tf.constant(1.0)

    def transition_state(self, state, request, is_click, is_conversion, timestamp):
        next_state = state.clone()
        _append_state_history(next_state, request, is_click, is_conversion)
        next_state.last_update_timestamp = timestamp
        return next_state


class AdCSPPGeneratorTest(unittest.TestCase):
    def test_rollout_updates_history_and_enforces_conversion_after_click(self):
        profile = UserProfile("u1", {"auid_ctr_90d": 0.2})
        state = UserStateBuilder.build(profile, aggregate_features={}, latent_dim=4)
        steps = AdCSPPPathGenerator(AlwaysEngageModel(), seed=1).rollout(
            state, [context(0), context(1)], sample=True
        )
        self.assertEqual([(step.is_click, step.is_conversion) for step in steps], [(1, 1), (1, 1)])
        self.assertEqual(steps[0].state_before.sequence_features.get("auid_hwdsp_clk_slotid_seq_time_365d", []), [])
        self.assertEqual(steps[0].state_after.sequence_features["auid_hwdsp_clk_slotid_seq_time_365d"], ["slot-0"])
        self.assertEqual(steps[1].state_before.sequence_features["auid_hwdsp_clk_slotid_seq_time_365d"], ["slot-0"])
        self.assertEqual(steps[1].state_after.sequence_features["auid_hwdsp_clk_slotid_seq_time_365d"], ["slot-0", "slot-1"])
        self.assertEqual(steps[1].state_after.sequence_features["auid_hwdsp_clk_slotid_seq_time_365d"], ["slot-0", "slot-1"])
        self.assertEqual(steps[1].state_after.sequence_features["auid_hwdsp_conv_slotid_seq_time_365d"], ["slot-0", "slot-1"])
        self.assertEqual(steps[1].state_after.dynamic_features["conversion_count"], 2)

    def test_offline_fixture_has_stable_static_features_and_logical_labels(self):
        fixture = SyntheticTrajectoryGenerator(SyntheticTrajectoryConfig(num_users=3, exposures_per_user=5, seed=3))
        records = fixture.generate_records()
        self.assertFalse(any(key.startswith("hist_") for key in records[0]))
        self.assertNotIn("gender", records[0])
        self.assertNotIn("device_type", records[0])
        by_user = {}
        for row in records:
            by_user.setdefault(row["auid"], []).append(row)
            self.assertFalse(row["label_keyaction"] and not row["label"])
        self.assertEqual(len(by_user), 3)
        for rows in by_user.values():
            self.assertEqual(len({row["auid"] for row in rows}), 1)
            self.assertEqual([row["logid"] for row in rows], sorted(row["logid"] for row in rows))
            for index, row in enumerate(rows):
                if index:
                    previous = rows[index - 1]
                    if previous["label"]:
                        self.assertIn(previous["slot_id"], row["auid_hwdsp_clk_slotid_seq_time_365d"])
                    else:
                        self.assertTrue(row["rt_auid_hwdsp_imp_unclk_3rd_indid_seq_time_12h"])
        dataset = fixture.as_dataset(batch_size=4)
        feature_batch, label_batch = next(iter(dataset))
        self.assertEqual(int(feature_batch["auid"].shape[0]), 4)
        self.assertEqual(tuple(label_batch["label"].shape), (4,))

    def test_full_generation_requires_and_uses_learned_flow_prior(self):
        profile = UserProfile("u1", {"auid_ctr_90d": 0.2})
        state = UserStateBuilder.build(
            profile,
            aggregate_features={"auid_hwdsp_clk_slotid_seq_time_365d": ["slot-history"]},
            latent_dim=4,
        )
        examples = [
            SimpleNamespace(state=state, context=context(0)),
            SimpleNamespace(state=state, context=context(1)),
        ]
        prior = AdCSPPFlowPrior.fit(examples, seed=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flow_prior.json"
            prior.save(path)
            loaded = AdCSPPFlowPrior.load(path)
            sampled_profile, sampled_history = loaded.sample_profile_with_history(user_id="generated")
            self.assertNotIn("auid_hwdsp_clk_slotid_seq_time_365d", sampled_profile.features)
            self.assertIn("auid_hwdsp_clk_slotid_seq_time_365d", sampled_history)
            generated = AdCSPPPathGenerator(AlwaysEngageModel(), seed=2).generate(
                state=state, prior=loaded, num_steps=3, start_timestamp=100, sample=False
            )
        self.assertEqual(len(generated), 3)
        self.assertEqual([item.is_click for item in generated], [1, 1, 1])
        self.assertEqual([item.is_conversion for item in generated], [1, 1, 1])
        self.assertEqual([item.timestamp for item in generated], sorted(item.timestamp for item in generated))
        self.assertEqual({item.context.slot_id for item in generated}, {"slot-0", "slot-1"})


if __name__ == "__main__":
    unittest.main()

