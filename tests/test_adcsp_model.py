import unittest

import numpy as np
import tensorflow as tf

from inputs.adcsp.schema import ExposureContext, UserProfile
from models.adcsp.model import AdCSPPModel
from models.adcsp.state import UserStateBuilder


class AdCSPPModelTest(unittest.TestCase):
    def test_encoder_and_hazards_return_finite_values(self):
        profile = UserProfile("u1", {"auid_ctr_90d": 0.2})
        context = ExposureContext("i1", 100, "slot_a", {"slot_form": "feed", "indu_name": "commerce", "slot_quality_score": 0.8})
        state = UserStateBuilder.build(profile, latent_dim=16)
        model = AdCSPPModel(seed=3)
        embedding = model.encode_history(state, context)
        click_probability = model.click_probability(state, context, 30)
        conversion_probability = model.conversion_probability(state, context, 3600)
        self.assertEqual(tuple(embedding.shape), (32,))
        self.assertTrue(np.isfinite(embedding.numpy()).all())
        self.assertGreaterEqual(click_probability, 0.0)
        self.assertLessEqual(click_probability, 1.0)
        self.assertGreaterEqual(conversion_probability, 0.0)
        self.assertLessEqual(conversion_probability, 1.0)

    def test_differentiable_behavior_outputs_are_probability_bounded(self):
        profile = UserProfile("u1", {"auid_ctr_90d": 0.2})
        context = ExposureContext("i1", 100, "slot_a", {"slot_form": "feed", "indu_name": "commerce", "slot_quality_score": 0.8})
        state = UserStateBuilder.build(profile, aggregate_features={"auid_hwdsp_clk_slotid_seq_time_365d": ["slot_b"]}, latent_dim=16)
        model = AdCSPPModel(seed=3)
        click_probability, conversion_probability = model.differentiable_probabilities(state, context, 30, 3600)
        self.assertTrue(np.isfinite(click_probability.numpy()).all())
        self.assertTrue(np.isfinite(conversion_probability.numpy()).all())
        self.assertGreaterEqual(float(click_probability.numpy()), 0.0)
        self.assertLessEqual(float(click_probability.numpy()), 1.0)
        self.assertGreaterEqual(float(conversion_probability.numpy()), 0.0)
        self.assertLessEqual(float(conversion_probability.numpy()), 1.0)

