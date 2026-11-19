import unittest

from inputs.adcsp.schema import ExposureContext, UserProfile
from models.adcsp.fsm import FiniteStateMachine
from models.adcsp.state import UserStateBuilder
from models.adcsp.state_updater import DynamicStateUpdater
from models.adcsp.state_transition import DynamicStateTransition
from models.adcsp.event_generator import HierarchicalEventGenerator
import tensorflow as tf


def make_context(timestamp=100):
    return ExposureContext(
        impression_id=f"imp-{timestamp}",
        event_timestamp=timestamp,
        slot_id="slot_a",
        features={"indu_name": "commerce", "promotion_target": "purchase"},
    )


class AdCSPPArchitectureTest(unittest.TestCase):
    def test_fsm_enforces_exposure_click_conversion_chain(self):
        fsm = FiniteStateMachine(click_window_seconds=30, conversion_window_seconds=3600)
        context = make_context()
        fsm.register_exposure(context)
        fsm.record_click(context)
        fsm.record_conversion(context)
        self.assertEqual([event.event_type for event in fsm.ledger.events], ["EXPOSURE", "CLICK", "CONVERSION"])

    def test_fsm_rejects_conversion_without_click(self):
        fsm = FiniteStateMachine()
        context = make_context()
        fsm.register_exposure(context)
        with self.assertRaises(ValueError):
            fsm.record_conversion(context)

    def test_fsm_closes_pending_conversion_window(self):
        fsm = FiniteStateMachine(click_window_seconds=30, conversion_window_seconds=60)
        context = make_context()
        fsm.register_exposure(context)
        fsm.record_click(context)
        expired = ExposureContext("imp-100", 160, "slot_a", context.features)
        fsm.record_conversion_expired(expired)
        self.assertEqual(fsm.ledger.events[-1].event_type, "CONVERSION_EXPIRED")

    def test_dynamic_updater_is_single_observable_state_mutation_point(self):
        profile = UserProfile("u1", {})
        state = UserStateBuilder.build(profile, latent_dim=4)
        updater = DynamicStateUpdater(max_sequence_length=20)
        context = make_context()
        updater.apply_outcome(state, context, is_click=1, is_conversion=1)
        self.assertEqual(state.dynamic_features["exposure_count"], 1)
        self.assertEqual(state.dynamic_features["click_count"], 1)
        self.assertEqual(state.dynamic_features["conversion_count"], 1)
        self.assertEqual(state.sequence_features["auid_hwdsp_clk_slotid_seq_time_365d"], ["slot_a"])
        self.assertEqual(state.sequence_features["auid_hwdsp_conv_slotid_seq_time_365d"], ["slot_a"])

    def test_dynamic_state_transition_is_explicit_and_event_conditioned(self):
        layer = DynamicStateTransition(latent_dim=4, hidden_dim=8)
        zeros = tf.zeros((4,), dtype=tf.float32)
        static = tf.zeros((6,), dtype=tf.float32)
        context = tf.zeros((5,), dtype=tf.float32)
        history = tf.zeros((5,), dtype=tf.float32)
        no_click = layer(zeros, static, context, history, tf.constant(0.0), tf.constant(0.0))
        click = layer(zeros, static, context, history, tf.constant(1.0), tf.constant(0.0))
        self.assertEqual(tuple(no_click.shape), (4,))
        self.assertEqual(tuple(click.shape), (4,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(click)))

    def test_hierarchical_generator_samples_valid_event_times(self):
        class FixedProbabilityModel:
            def differentiable_probabilities(self, *args, **kwargs):
                return tf.constant(0.8), tf.constant(0.7)

        profile = UserProfile("u1", {})
        state = UserStateBuilder.build(profile, latent_dim=4)
        context = make_context(timestamp=100)
        outcome = HierarchicalEventGenerator(
            FixedProbabilityModel(), click_window_seconds=30, conversion_window_seconds=3600, seed=7
        ).generate(state, context, sample=False)
        self.assertEqual(outcome.is_click, 1)
        self.assertEqual(outcome.is_conversion, 1)
        self.assertGreaterEqual(outcome.click_timestamp, context.event_timestamp)
        self.assertLessEqual(outcome.click_timestamp, context.event_timestamp + 30)
        self.assertGreaterEqual(outcome.conversion_timestamp, outcome.click_timestamp)
        self.assertLessEqual(outcome.conversion_timestamp, outcome.click_timestamp + 3600)


if __name__ == "__main__":
    unittest.main()
