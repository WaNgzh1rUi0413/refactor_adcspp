import unittest

import tensorflow as tf

from inputs.adcsp.schema import ExposureContext, UserProfile
from losses.adcsp_losses import AdCSPPJointLoss
from models.adcsp.joint_generator import JointTrafficGenerator
from models.adcsp.model import AdCSPPModel
from models.adcsp.scheduler import JointEventScheduler
from models.adcsp.heads import CandidateMasker, CandidateSelectionTrainer
from models.adcsp.state import UserStateBuilder
from models.adcsp.state_updater import DynamicStateUpdater


class AlwaysJointModel:
    def select_candidate(self, state, candidates, policy_embedding=None, valid_mask=None):
        return 0, tf.zeros((len(candidates),)), tf.one_hot(0, len(candidates))

    def exposure_opportunity_rate(self, state, context, policy_embedding=None):
        return 1.0 / 30.0

    def differentiable_probabilities(self, state, context, click_window_seconds, conversion_window_seconds, click_outcome=None):
        return tf.constant(1.0), tf.constant(1.0)

    def transition_event(self, state, context, event_type, timestamp=None):
        next_state = state.clone()
        updater = DynamicStateUpdater()
        if event_type == "EXPOSURE":
            updater.apply_exposure(next_state, context)
        elif event_type == "CLICK":
            updater.apply_click(next_state, context)
        elif event_type == "CONVERSION":
            updater.apply_conversion(next_state, context)
        else:
            updater.apply_no_click(next_state, context)
        next_state.last_update_timestamp = int(timestamp or context.event_timestamp)
        return next_state


class AdCSPPJointGenerationTest(unittest.TestCase):
    def test_scheduler_is_timestamp_ordered_and_stable(self):
        scheduler = JointEventScheduler()
        scheduler.schedule("CLICK", 20)
        scheduler.schedule("EXPOSURE", 10)
        scheduler.schedule("CONVERSION", 20)
        self.assertEqual([scheduler.pop().event_type for _ in range(3)], ["EXPOSURE", "CLICK", "CONVERSION"])

    def test_candidate_masking_keeps_eligible_candidate(self):
        candidates = [{"slot_id": "slot_a", "ad_id": "a"}, {"slot_id": "slot_b", "ad_id": "b"}]
        mask = CandidateMasker().mask(candidates, required_slot_id="slot_a")
        self.assertEqual(mask.tolist(), [True, False])
        with self.assertRaises(ValueError):
            CandidateMasker().mask(candidates, blocked_ad_ids=["a", "b"])

    def test_joint_generator_emits_legal_concurrent_path(self):
        profile = UserProfile("u1", {"auid_ctr_90d": 0.2})
        state = UserStateBuilder.build(profile, latent_dim=4)
        candidates = [ExposureContext("template-a", 0, "slot_a", {"ad_id": "a", "promotion_target": "purchase"})]
        events = JointTrafficGenerator(AlwaysJointModel(), seed=3).generate(state, candidates, start_timestamp=100, horizon_seconds=90, sample=False, max_events=100)
        self.assertTrue(events)
        types = [event["event_type"] for event in events]
        self.assertEqual(types[0], "EXPOSURE")
        self.assertIn("CLICK", types)
        self.assertIn("CONVERSION", types)
        self.assertEqual(sorted(event["timestamp"] for event in events), [event["timestamp"] for event in events])

    def test_joint_loss_is_finite(self):
        result = AdCSPPJointLoss()(tf.constant([0.1, 0.2]), tf.constant([10.0, 20.0]), tf.constant([1.0, -1.0]), tf.constant(0), tf.constant([True, True]))
        self.assertTrue(bool(tf.math.is_finite(result["loss"])))

    def test_model_exposes_opportunity_rate_and_masked_selection(self):
        state = UserStateBuilder.build(UserProfile("u1", {"auid_ctr_90d": 0.2}), latent_dim=16)
        candidates = [ExposureContext("a", 100, "slot_a", {"ad_id": "a", "slot_quality_score": 0.8}), ExposureContext("b", 100, "slot_b", {"ad_id": "b", "slot_quality_score": 0.9})]
        model = AdCSPPModel(seed=5)
        rate = model.exposure_opportunity_rate(state, candidates[0])
        index, logits, probabilities = model.select_candidate(state, candidates, valid_mask=tf.constant([True, False]))
        self.assertGreater(rate, 0.0)
        self.assertEqual(index, 0)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertAlmostEqual(float(tf.reduce_sum(probabilities).numpy()), 1.0, places=5)

    def test_candidate_selection_trainer_accepts_observed_mark(self):
        state = UserStateBuilder.build(UserProfile("u1", {"auid_ctr_90d": 0.2}), latent_dim=16)
        candidates = [ExposureContext("a", 100, "slot_a", {"ad_id": "a"}), ExposureContext("b", 100, "slot_b", {"ad_id": "b"})]
        result = CandidateSelectionTrainer().fit(AdCSPPModel(seed=8), [{"state": state, "candidates": candidates, "selected_index": 1, "valid_mask": tf.constant([True, True])}])
        self.assertEqual(result["selection_examples"], 1)
        self.assertTrue(tf.math.is_finite(result["selection_loss"]))


if __name__ == "__main__":
    unittest.main()

