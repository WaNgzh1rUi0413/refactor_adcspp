"""TensorFlow losses for the AdCSPP behavior model."""

from __future__ import annotations

import tensorflow as tf


class AdCSPPSurvivalLoss:
    """Masked Bernoulli loss for click and post-click conversion."""

    def __init__(self, conversion_weight: float = 2.0):
        self.conversion_weight = float(conversion_weight)

    def __call__(self, click_probability, click_label, conversion_probability, conversion_label, mask=None, conversion_mask=None):
        click_probability = tf.clip_by_value(tf.cast(click_probability, tf.float32), 1e-6, 1.0 - 1e-6)
        conversion_probability = tf.clip_by_value(tf.cast(conversion_probability, tf.float32), 1e-6, 1.0 - 1e-6)
        click_label = tf.cast(click_label, tf.float32)
        conversion_label = tf.cast(conversion_label, tf.float32)
        click_mask = tf.ones_like(click_label) if mask is None else tf.cast(mask, tf.float32)
        conversion_mask = click_mask if conversion_mask is None else tf.cast(conversion_mask, tf.float32)
        click_element_loss = tf.keras.backend.binary_crossentropy(click_label, click_probability)
        conversion_element_loss = tf.keras.backend.binary_crossentropy(conversion_label, conversion_probability)
        click_denominator = tf.maximum(tf.reduce_sum(click_mask), 1.0)
        conversion_denominator = tf.maximum(tf.reduce_sum(conversion_mask), 1.0)
        click_loss = tf.reduce_sum(click_element_loss * click_mask) / click_denominator
        conversion_loss = tf.reduce_sum(conversion_element_loss * conversion_mask) / conversion_denominator
        return {
            "loss": click_loss + self.conversion_weight * conversion_loss,
            "click_loss": click_loss,
            "conversion_loss": conversion_loss,
        }


class ExposureOpportunityLoss:
    """Exponential point-process NLL for observed inter-exposure gaps."""

    def __init__(self, minimum_rate: float = 1e-6):
        self.minimum_rate = float(minimum_rate)

    def __call__(self, rate, gap_seconds, mask=None):
        rate = tf.maximum(tf.cast(rate, tf.float32), self.minimum_rate)
        gap_seconds = tf.maximum(tf.cast(gap_seconds, tf.float32), 0.0)
        element_loss = rate * gap_seconds - tf.math.log(rate)
        if mask is None:
            return tf.reduce_mean(element_loss)
        mask = tf.cast(mask, tf.float32)
        return tf.reduce_sum(element_loss * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


class AdSelectionLoss:
    """Masked conditional cross-entropy for the observed selected candidate."""

    def __call__(self, logits, selected_index, valid_mask=None):
        logits = tf.cast(logits, tf.float32)
        selected_index = tf.cast(selected_index, tf.int32)
        if valid_mask is not None:
            valid_mask = tf.cast(valid_mask, tf.bool)
            logits = tf.where(valid_mask, logits, tf.constant(-1e9, dtype=logits.dtype))
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(selected_index, logits))


class AdCSPPJointLoss:
    """Joint objective for opportunity timing and candidate selection."""

    def __init__(self, opportunity_weight: float = 1.0, selection_weight: float = 1.0):
        self.opportunity_weight = float(opportunity_weight)
        self.selection_weight = float(selection_weight)
        self.opportunity_loss = ExposureOpportunityLoss()
        self.selection_loss = AdSelectionLoss()

    def __call__(self, opportunity_rate, gap_seconds, selection_logits, selected_index, valid_mask=None, gap_mask=None):
        opportunity = self.opportunity_loss(opportunity_rate, gap_seconds, gap_mask)
        selection = self.selection_loss(selection_logits, selected_index, valid_mask)
        total = self.opportunity_weight * opportunity + self.selection_weight * selection
        return {"loss": total, "opportunity_loss": opportunity, "selection_loss": selection}

