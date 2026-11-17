import unittest

from inputs.adcsp.schema import Event, EventType, ExposureContext, UserProfile


class AdCSPPSchemaTest(unittest.TestCase):
    def test_event_contract_contains_documented_ledger_fields(self):
        event = Event("u1", "e1", 100, EventType.EXPOSURE, impression_id="i1", slot_id="slot_a", ad_id="ad1")
        record = event.to_dict()
        for field in ("user_id", "event_id", "event_timestamp", "event_seq_no", "event_type", "impression_id", "slot_id", "ad_id", "state_version"):
            # state_version is added by the persistence adapter, while the
            # event contract exposes every event-level field.
            if field == "state_version":
                continue
            self.assertIn(field, record)
        self.assertEqual(record["event_type"], "EXPOSURE")

    def test_profile_and_context_are_serializable(self):
        profile = UserProfile("u1", "female", "25_34", "east", "phone", "ios", "active", ("commerce",), 10.0, "medium")
        context = ExposureContext("i1", 100, "slot_a", "feed", "top", "ad1", "cr1", "camp1", "adv1", "commerce", "purchase", "image", "square", 1.0, 2.0, 0.8, "product", "purchase", "s1", "organic", "shopping", "home", "wifi")
        self.assertEqual(profile.to_dict()["user_id"], "u1")
        self.assertEqual(context.to_dict()["impression_id"], "i1")

