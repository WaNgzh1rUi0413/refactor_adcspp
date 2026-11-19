"""Learned dynamic latent-state transition for AdCSPP.

The transition is deliberately separate from the observable feature updater:
``DynamicStateTransition`` changes the latent state, while
``DynamicStateUpdater`` appends event-derived sequences and counters.  Keeping
the two paths separate mirrors the design document and avoids double counting
events in online rollouts.
"""

from __future__ import annotations

import tensorflow as tf


class DynamicStateTransition(tf.keras.layers.Layer):
    """Event-conditioned latent state transition.

    Inputs are the previous latent state, static user embedding, current
    context embedding, historical sequence embedding, and the observed event
    marks.  The layer returns the latent state used by the next request.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 32, name: str = "dynamic_state_transition"):
        super().__init__(name=name)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.hidden_dim, activation="tanh"),
                tf.keras.layers.Dense(self.latent_dim),
            ],
            name="transition_network",
        )

    def call(
        self,
        latent_state,
        static_embedding,
        context_embedding,
        history_embedding,
        is_click,
        is_conversion,
        training: bool = False,
    ):
        event = tf.stack([
            tf.cast(is_click, tf.float32),
            tf.cast(is_conversion, tf.float32),
        ])
        vector = tf.concat(
            [
                tf.cast(latent_state, tf.float32),
                tf.cast(static_embedding, tf.float32),
                tf.cast(context_embedding, tf.float32),
                tf.cast(history_embedding, tf.float32),
                event,
            ],
            axis=-1,
        )
        return self.network(tf.expand_dims(vector, axis=0), training=training)[0]

