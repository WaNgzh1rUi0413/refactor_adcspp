import unittest

import numpy as np

from inputs.adcsp.schema import Event, EventType, UserProfile
from models.adcsp.state import StateUpdater, UserStateBuilder


def profile():
    return UserProfile("u1", "female", "25_34", "east", "phone", "ios", "active", ("commerce",), 10.0, "medium")


class AdCSPPStateTest(unittest.TestCase):
    def test_state_update_recomputes_dynamic_features(self):
        state = UserStateBuilder.build(profile(), latent_dim=8)
        updater = StateUpdater()
        updater.apply_event(state, Event("u1", "e1", 100, EventType.EXPOSURE, impression_id="i1", slot_id="slot_a", ad_id="ad1"))
        updater.apply_event(state, Event("u1", "e2", 101, EventType.CLICK, impression_id="i1", click_id="c1", parent_event_id="e1", slot_id="slot_a", ad_id="ad1"))
        updater.apply_event(state, Event("u1", "e3", 110, EventType.CONVERSION, impression_id="i1", click_id="c1", parent_event_id="e2", slot_id="slot_a", ad_id="ad1", conversion_type="purchase"))
        self.assertEqual(state.user_exposure_cnt, 1)
        self.assertEqual(state.user_click_cnt, 1)
        self.assertEqual(state.user_conversion_cnt["purchase"], 1)
        self.assertEqual(state.dynamic_features["user_exposure_cnt_1h"], 1)
        self.assertEqual(state.dynamic_features["user_click_cnt_1h"], 1)
        self.assertTrue(np.all(np.isfinite(state.latent_state)))
        self.assertEqual(state.state_version, 3)

