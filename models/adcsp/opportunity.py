"""Exposure opportunity generation over a learned prior or intensity head."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


class ExposureOpportunityHead(tf.keras.layers.Layer):
    """Predict the continuous-time intensity of the next exposure opportunity.

    The output is a positive rate (events per second).  The head is kept
    separate from candidate selection because an opportunity and the ad mark
    shown at that opportunity are different decisions.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, name: str = "exposure_opportunity_head"):
        super().__init__(name=name)
        self.network = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(int(hidden_dim), activation="tanh"),
                tf.keras.layers.Dense(1),
            ],
            name="opportunity_rate_network",
        )

    def call(self, features, training: bool = False):
        logits = self.network(tf.expand_dims(tf.cast(features, tf.float32), axis=0), training=training)[0, 0]
        return tf.nn.softplus(logits) + tf.constant(1e-6, dtype=tf.float32)


def sample_exposure_wait(rate: float, rng: np.random.Generator, max_wait_seconds: int | None = None) -> int:
    """Sample an inter-opportunity wait from a homogeneous Poisson intensity."""
    rate = float(rate)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("exposure opportunity rate must be finite and positive")
    wait = max(1, int(round(float(rng.exponential(1.0 / rate)))))
    if max_wait_seconds is not None:
        wait = min(wait, max(1, int(max_wait_seconds)))
    return wait


class ExposureOpportunityGenerator:
    """Generate the next exposure opportunity from empirical training flow.

    V1 uses the learned empirical prior instead of an independently trained
    opportunity intensity head because exposure opportunities are externally
    observed in the platform input.  The class makes that boundary explicit
    and provides the replacement point for a learned ``lambda_opp`` head.
    """

    def __init__(self, prior):
        if prior is None:
            raise ValueError("ExposureOpportunityGenerator requires a learned prior")
        self.prior = prior

    def next(self, profile, timestamp: int, previous_context=None):
        next_timestamp = int(timestamp) + self.prior.sample_gap()
        context = self.prior.sample_context(profile, next_timestamp, previous_context=previous_context)
        return next_timestamp, context

