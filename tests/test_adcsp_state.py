import unittest

from inputs.adcsp.schema import UserProfile
from models.adcsp.state import UserStateBuilder


class AdCSPPStateTest(unittest.TestCase):
    def test_online_history_lists_are_used_as_model_state(self):
        profile = UserProfile("u1", "female", "25_34", "east", "phone", "ios", "active", ("commerce",), 10.0, "medium")
        state = UserStateBuilder.build(
            profile,
            aggregate_features={"hist_clk_slot": ["slot_old"], "user_click_cnt_1h": 1},
            latent_dim=8,
        )
        self.assertEqual(state.sequence_features["hist_clk_slot"], ["slot_old"])
        self.assertEqual(state.dynamic_features["user_click_cnt_1h"], 1)
        self.assertEqual(state.latent_state.shape[0], 8)

