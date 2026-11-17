import unittest

import tensorflow as tf

from losses.adcsp_losses import AdCSPPSurvivalLoss


class AdCSPPLossTest(unittest.TestCase):
    def test_survival_loss_returns_finite_components(self):
        result = AdCSPPSurvivalLoss()(tf.constant([0.5, 0.8]), tf.constant([0.0, 1.0]), tf.constant([0.2, 0.7]), tf.constant([0.0, 1.0]))
        self.assertTrue(bool(tf.math.is_finite(result["loss"])))
        self.assertIn("click_loss", result)
        self.assertIn("conversion_loss", result)

