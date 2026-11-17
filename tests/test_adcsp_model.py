import unittest

import numpy as np

from inputs.adcsp.schema import ExposureContext, UserProfile
from models.adcsp.model import AdCSPPModel
from models.adcsp.state import UserStateBuilder


class AdCSPPModelTest(unittest.TestCase):
    def test_encoder_and_hazards_return_finite_values(self):
        profile = UserProfile("u1", "female", "25_34", "east", "phone", "ios", "active", ("commerce",), 10.0, "medium")
        context = ExposureContext("i1", 100, "slot_a", "feed", "top", "ad1", "cr1", "camp1", "adv1", "commerce", "purchase", "image", "square", 1.0, 2.0, 0.8, "product", "purchase", "s1", "organic", "shopping", "home", "wifi")
        state = UserStateBuilder.build(profile, latent_dim=16)
        model = AdCSPPModel(seed=3)
        embedding = model.encode_history(state, context)
        click_probability = model.click_probability(state, context, 30)
        conversion_probability = model.conversion_probability(state, context, 3600)
        self.assertEqual(tuple(embedding.shape), (32,))
        self.assertTrue(np.isfinite(embedding.detach().numpy()).all())
        self.assertGreaterEqual(click_probability, 0.0)
        self.assertLessEqual(click_probability, 1.0)
        self.assertGreaterEqual(conversion_probability, 0.0)
        self.assertLessEqual(conversion_probability, 1.0)

