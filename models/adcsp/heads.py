"""TensorFlow click and conversion hazard heads."""

from __future__ import annotations

import tensorflow as tf


class _HazardHead(tf.keras.layers.Layer):
    def __init__(self, input_dim: int, hidden_dim: int = 32, bias: float = -3.0):
        super().__init__()
        self.network = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="tanh"),
            tf.keras.layers.Dense(1, bias_initializer=tf.keras.initializers.Constant(bias)),
        ])

    def rate(self, state_embedding, context_embedding):
        state = tf.cast(state_embedding, tf.float32)
        context = tf.cast(context_embedding, tf.float32)
        vector = tf.concat([state, context], axis=-1)
        return tf.squeeze(tf.nn.softplus(self.network(tf.expand_dims(vector, axis=0))), axis=-1)

    def probability(self, state_embedding, context_embedding, window_seconds: int):
        rate = self.rate(state_embedding, context_embedding)
        return tf.reshape(1.0 - tf.exp(-rate * float(window_seconds) / 60.0), [])


class ClickHazardHead(_HazardHead):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__(input_dim, hidden_dim, bias=-1.0)


class ConversionHazardHead(_HazardHead):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__(input_dim, hidden_dim, bias=-6.0)

