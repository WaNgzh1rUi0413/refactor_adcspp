"""TensorFlow AdCSPP conditional click/conversion behavior model."""

from __future__ import annotations

from typing import Optional

import tensorflow as tf

from inputs.adcsp.schema import ExposureContext, UserState
from .encoders import ContextEncoder, ContinuousTimeStateBackbone, SequenceFeatureEncoder, StaticUserEncoder
from .heads import ClickHazardHead, ConversionHazardHead


class AdCSPPModel(tf.keras.Model):
    def __init__(self, latent_dim: int = 16, static_dim: int = 16, context_dim: int = 16, hidden_dim: int = 32, seed: int = 42, feature_config=None):
        super().__init__(name="adcsp_model")
        tf.keras.utils.set_random_seed(seed)
        feature_config = feature_config or {}
        hash_buckets = int(feature_config.get("hash_buckets", 4096))
        self.latent_dim = int(latent_dim)
        self.static_encoder = StaticUserEncoder(output_dim=static_dim, hash_buckets=hash_buckets)
        self.context_encoder = ContextEncoder(output_dim=context_dim, hash_buckets=hash_buckets)
        sequence_names = list(feature_config.get("sequence_features", {}).keys()) or None
        self.sequence_encoder = SequenceFeatureEncoder(output_dim=context_dim, sequence_names=sequence_names)
        self.backbone = ContinuousTimeStateBackbone(latent_dim, static_dim, context_dim, hidden_dim)
        self.click_head = ClickHazardHead(hidden_dim + context_dim)
        self.conversion_head = ConversionHazardHead(hidden_dim + context_dim)

    def encode_history(self, state: UserState, context: Optional[ExposureContext] = None) -> tf.Tensor:
        static_embedding = self.static_encoder.call(state.static_features)
        if context is None:
            context_embedding = tf.zeros((self.context_encoder.projection.layers[0].units,), dtype=tf.float32)
        else:
            context_embedding = self.context_encoder.call(context)
        history_embedding = self.sequence_encoder.call(state.sequence_features)
        return self.backbone.call(state.latent_state, static_embedding, context_embedding + history_embedding, 0.0)

    def click_probability(self, state: UserState, context: ExposureContext, window_seconds: int) -> float:
        state_embedding = self.encode_history(state, context)
        context_embedding = self.context_encoder.call(context)
        return float(tf.reshape(self.click_head.probability(state_embedding, context_embedding, window_seconds), []).numpy())

    def conversion_probability(self, state: UserState, context: ExposureContext, window_seconds: int) -> float:
        state_embedding = self.encode_history(state, context)
        context_embedding = self.context_encoder.call(context)
        return float(tf.reshape(self.conversion_head.probability(state_embedding, context_embedding, window_seconds), []).numpy())

    def differentiable_probabilities(self, state: UserState, context: ExposureContext, click_window_seconds: int, conversion_window_seconds: int):
        static_embedding = self.static_encoder.call(state.static_features)
        context_embedding = self.context_encoder.call(context)
        history_embedding = self.sequence_encoder.call(state.sequence_features)
        state_embedding = self.backbone.call(state.latent_state, static_embedding, context_embedding + history_embedding, 0.0)
        click_probability = self.click_head.probability(state_embedding, context_embedding, click_window_seconds)
        conversion_probability = self.conversion_head.probability(state_embedding, context_embedding, conversion_window_seconds)
        return tf.reshape(click_probability, []), tf.reshape(conversion_probability, [])

