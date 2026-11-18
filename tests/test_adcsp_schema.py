import unittest

from inputs.adcsp.schema import ExposureContext, UserProfile


class AdCSPPSchemaTest(unittest.TestCase):
    def test_profile_and_context_are_serializable(self):
        profile = UserProfile("u1", {"auid_ctr_90d": 0.2})
        context = ExposureContext("i1", 100, "slot_a", {"slot_form": "feed", "indu_name": "commerce"})
        self.assertEqual(profile.to_dict()["user_id"], "u1")
        self.assertEqual(context.to_dict()["slot_id"], "slot_a")

