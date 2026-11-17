import unittest

from inputs.adcsp.schema import ExposureContext, UserProfile


class AdCSPPSchemaTest(unittest.TestCase):
    def test_profile_and_context_are_serializable(self):
        profile = UserProfile("u1", "female", "25_34", "east", "phone", "ios", "active", ("commerce",), 10.0, "medium")
        context = ExposureContext("i1", 100, "slot_a", "feed", "top", "ad1", "cr1", "camp1", "adv1", "commerce", "purchase", "image", "square", 1.0, 2.0, 0.8, "product", "purchase", "s1", "organic", "shopping", "home", "wifi")
        self.assertEqual(profile.to_dict()["user_id"], "u1")
        self.assertEqual(context.to_dict()["slot_id"], "slot_a")

