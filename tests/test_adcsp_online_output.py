import unittest

from inputs.adcsp.online_output import trajectory_to_online_records
from inputs.adcsp.schema import Event, EventType, ExposureContext, UserProfile


class AdCSPPOnlineOutputTest(unittest.TestCase):
    def test_flat_row_contains_history_and_labels_without_event_type(self):
        profile = UserProfile("auid_1", "female", "25_34", "east", "phone", "ios", "active", (), 30.0, "medium")
        context = ExposureContext("imp_1", 100, "s6711dasd", "feed", "top", "ad_1", "cr_1", "camp_1", "adv_1", "commerce", "click", "image", "square", 1.0, 2.0, 0.8, "product", "purchase", "sess_1", "organic", "shopping", "home", "wifi")
        events = [
            Event("auid_1", "exp", 100, EventType.EXPOSURE, impression_id="imp_1", slot_id="s6711dasd", ad_id="ad_1"),
            Event("auid_1", "clk", 101, EventType.CLICK, impression_id="imp_1", click_id="clk", slot_id="s6711dasd", ad_id="ad_1"),
            Event("auid_1", "cnv", 102, EventType.CONVERSION, impression_id="imp_1", click_id="clk", slot_id="s6711dasd", ad_id="ad_1"),
        ]
        rows = trajectory_to_online_records(
            profile,
            [context],
            events,
            initial_sequence_features={"hist_clk_slots": ["old_slot"]},
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["auid"], "auid_1")
        self.assertEqual(rows[0]["hist_clk_slot"], ["old_slot"])
        self.assertEqual(rows[0]["is_click"], 1)
        self.assertEqual(rows[0]["is_conversion"], 1)
        self.assertEqual(rows[0]["time_stamp"], 100)
        self.assertNotIn("event_type", rows[0])

    def test_empty_history_columns_are_present(self):
        profile = UserProfile("u1", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown", (), 0.0, "unknown")
        context = ExposureContext("imp", 10, "slot", "unknown", "unknown", "ad", "cr", "camp", "adv", "unknown", "unknown", "unknown", "unknown", 0.0, 0.0, 0.0, "unknown", "unknown", "sess", "unknown", "unknown", "unknown", "unknown")
        row = trajectory_to_online_records(profile, [context], [Event("u1", "e", 10, EventType.EXPOSURE, impression_id="imp", slot_id="slot", ad_id="ad")])[0]
        self.assertEqual(row["hist_clk_slot"], [])
        self.assertEqual(row["hist_conv_slot"], [])

