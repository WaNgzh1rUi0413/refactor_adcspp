import unittest

from inputs.adcsp.schema import ClickResult, ConversionResult, EventType, ExposureContext, SimulationConfig, UserProfile
from models.adcsp.simulator import ConditionalSimulator
from models.adcsp.state import UserStateBuilder


class DeterministicModel:
    def sample_click(self, state, context, window_seconds, rng):
        return ClickResult(True, 1.0, 1)

    def sample_conversion(self, state, context, window_seconds, rng):
        return ConversionResult(True, 1.0, 2, "purchase", "purchase", 1.0)


class AdCSPPSimulatorTest(unittest.TestCase):
    def test_concurrent_exposure_chains_preserve_parent_links(self):
        profile = UserProfile("u1", "female", "25_34", "east", "phone", "ios", "active", ("commerce",), 10.0, "medium")
        context = lambda i, t: ExposureContext(f"i{i}", t, "slot_a", "feed", "top", f"ad{i}", f"cr{i}", "camp1", "adv1", "commerce", "purchase", "image", "square", 1.0, 2.0, 0.8, "product", "purchase", "s1", "organic", "shopping", "home", "wifi")
        state = UserStateBuilder.build(profile, latent_dim=16)
        trajectory = ConditionalSimulator(DeterministicModel(), SimulationConfig(
            seed=1,
            start_timestamp=100,
            horizon_seconds=100,
            feature_config={
                "persist_sequence_features": True,
                "sequence_features": {"hist_clk_slots": {"event_type": "CLICK", "field": "slot_id", "max_len": 2}},
            },
        )).simulate(state, [context(1, 100), context(2, 101)])
        events = trajectory.events
        self.assertEqual(sum(event.event_type is EventType.EXPOSURE for event in events), 2)
        self.assertEqual(sum(event.event_type is EventType.CLICK for event in events), 2)
        self.assertEqual(sum(event.event_type is EventType.CONVERSION for event in events), 2)
        for event in events:
            if event.event_type is EventType.CLICK:
                self.assertEqual(event.parent_event_id, f"EXP_{event.impression_id}")
            if event.event_type is EventType.CONVERSION:
                self.assertEqual(event.parent_event_id, f"CLK_{event.impression_id}")
        self.assertEqual(trajectory.final_state.dynamic_features["hist_clk_slots"], ["slot_a", "slot_a"])
        self.assertEqual(trajectory.final_state.sequence_features["hist_clk_slots"], ["slot_a", "slot_a"])
        click_events = [event for event in events if event.event_type is EventType.CLICK]
        self.assertEqual(click_events[0].metadata["sequence_features"]["hist_clk_slots"], ["slot_a"])

