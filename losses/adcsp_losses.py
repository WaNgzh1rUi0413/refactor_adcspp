"""TensorFlow losses for the AdCSPP behavior model."""

from __future__ import annotations

import tensorflow as tf


class AdCSPPSurvivalLoss:
    """Masked Bernoulli loss for click and post-click conversion."""

    def __init__(self, conversion_weight: float = 2.0):
        self.conversion_weight = float(conversion_weight)

    def __call__(self, click_probability, click_label, conversion_probability, conversion_label, mask=None):
        click_probability = tf.clip_by_value(tf.cast(click_probability, tf.float32), 1e-6, 1.0 - 1e-6)
        conversion_probability = tf.clip_by_value(tf.cast(conversion_probability, tf.float32), 1e-6, 1.0 - 1e-6)
        click_label = tf.cast(click_label, tf.float32)
        conversion_label = tf.cast(conversion_label, tf.float32)
        if mask is None:
            mask = tf.ones_like(click_label)
        mask = tf.cast(mask, tf.float32)
        click_element_loss = tf.keras.backend.binary_crossentropy(click_label, click_probability)
        conversion_element_loss = tf.keras.backend.binary_crossentropy(conversion_label, conversion_probability)
        denominator = tf.maximum(tf.reduce_sum(mask), 1.0)
        click_loss = tf.reduce_sum(click_element_loss * mask) / denominator
        conversion_loss = tf.reduce_sum(conversion_element_loss * mask) / denominator
        return {
            "loss": click_loss + self.conversion_weight * conversion_loss,
            "click_loss": click_loss,
            "conversion_loss": conversion_loss,
        }

