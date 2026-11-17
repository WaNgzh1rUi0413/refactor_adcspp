"""TensorFlow encoders for online AdCSPP behavior modeling."""

from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import tensorflow as tf

from inputs.adcsp.schema import ExposureContext, UserProfile


def _hash_index(field: str, value: object, num_buckets: int) -> int:
    digest = hashlib.sha1(f"{field}::{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % int(num_buckets)


class StaticUserEncoder(tf.keras.layers.Layer):
    def __init__(self, output_dim: int = 16, embedding_dim: int = 8, hash_buckets: int = 4096):
        super().__init__()
        self.hash_buckets = int(hash_buckets)
        self.fields = ["gender", "age_group", "region", "device_type", "os_type", "user_segment", "user_value_segment"]
        for field in self.fields:
            setattr(self, f"embedding_{field}", tf.keras.layers.Embedding(self.hash_buckets, embedding_dim))
        self.interest_embedding = tf.keras.layers.Embedding(self.hash_buckets, embedding_dim)
        self.projection = tf.keras.Sequential([
            tf.keras.layers.Dense(output_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("tanh"),
        ])

    def call(self, profile: UserProfile | Mapping[str, object]) -> tf.Tensor:
        values = profile.to_dict() if isinstance(profile, UserProfile) else profile
        encoded = []
        for field in self.fields:
            index = tf.constant([_hash_index(field, values.get(field, "unknown"), self.hash_buckets)], dtype=tf.int32)
            encoded.append(tf.squeeze(getattr(self, f"embedding_{field}")(index), axis=0))
        interests = values.get("user_long_term_interest", ()) or ()
        if interests:
            indices = tf.constant([_hash_index("user_long_term_interest", item, self.hash_buckets) for item in interests], dtype=tf.int32)
            interest = tf.reduce_mean(self.interest_embedding(indices), axis=0)
        else:
            interest = tf.zeros((self.interest_embedding.output_dim,), dtype=tf.float32)
        age = tf.reshape(tf.cast(values.get("user_registration_age", 0.0), tf.float32) / 3650.0, [1])
        vector = tf.concat(encoded + [interest, age], axis=0)
        return tf.squeeze(self.projection(tf.expand_dims(vector, axis=0)), axis=0)


class ContextEncoder(tf.keras.layers.Layer):
    def __init__(self, output_dim: int = 16, embedding_dim: int = 6, hash_buckets: int = 4096):
        super().__init__()
        self.hash_buckets = int(hash_buckets)
        self.fields = ["slot_id", "slot_type", "slot_position", "ad_category", "creative_type", "creative_size", "objective_type", "landing_page_type", "promotion_target", "traffic_source", "app_category", "page_context", "network_type"]
        for field in self.fields:
            setattr(self, f"embedding_{field}", tf.keras.layers.Embedding(self.hash_buckets, embedding_dim))
        self.projection = tf.keras.Sequential([
            tf.keras.layers.Dense(output_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("tanh"),
        ])

    def call(self, context: ExposureContext | Mapping[str, object]) -> tf.Tensor:
        values = context.to_dict() if isinstance(context, ExposureContext) else context
        encoded = []
        for field in self.fields:
            index = tf.constant([_hash_index(field, values.get(field, "unknown"), self.hash_buckets)], dtype=tf.int32)
            encoded.append(tf.squeeze(getattr(self, f"embedding_{field}")(index), axis=0))
        numeric = tf.stack([
            tf.cast(values.get("bid_price", 0.0), tf.float32),
            tf.cast(values.get("ecpm", 0.0), tf.float32),
            tf.cast(values.get("ad_quality_score", 0.0), tf.float32),
            tf.cast(values.get("is_holiday", False), tf.float32),
            tf.cast(values.get("event_timestamp", 0), tf.float32) % 86400.0 / 86400.0,
        ])
        vector = tf.concat(encoded + [numeric], axis=0)
        return tf.squeeze(self.projection(tf.expand_dims(vector, axis=0)), axis=0)


class SequenceFeatureEncoder(tf.keras.layers.Layer):
    def __init__(self, output_dim: int = 16, embedding_dim: int = 8, num_embeddings: int = 256, sequence_names=None):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.sequence_names = list(sequence_names or ["hist_exp_slots", "hist_exp_ads", "hist_clk_slots", "hist_clk_ads", "hist_clk_categories", "hist_conv_slots", "hist_conv_targets"])
        for name in self.sequence_names:
            setattr(self, f"embedding_{name}", tf.keras.layers.Embedding(self.num_embeddings, embedding_dim))
        self.projection = tf.keras.Sequential([
            tf.keras.layers.Dense(output_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("tanh"),
        ])

    def call(self, sequence_features: Mapping[str, Sequence[object]] | None) -> tf.Tensor:
        values = sequence_features or {}
        encoded = []
        for name in self.sequence_names:
            items = list(values.get(name, []))
            if items:
                indices = tf.constant([_hash_index("sequence", item, self.num_embeddings) for item in items], dtype=tf.int32)
                encoded.append(tf.reduce_mean(getattr(self, f"embedding_{name}")(indices), axis=0))
            else:
                encoded.append(tf.zeros((getattr(self, f"embedding_{name}").output_dim,), dtype=tf.float32))
        vector = tf.concat(encoded, axis=0)
        return tf.squeeze(self.projection(tf.expand_dims(vector, axis=0)), axis=0)


class ContinuousTimeStateBackbone(tf.keras.layers.Layer):
    def __init__(self, latent_dim: int = 16, static_dim: int = 16, context_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.decay_rate = self.add_weight(name="decay_rate", shape=(), initializer=tf.constant_initializer(1.0 / 3600.0), trainable=True)
        self.fusion = tf.keras.Sequential([tf.keras.layers.Dense(hidden_dim), tf.keras.layers.Activation("tanh")])

    def call(self, latent_state, static_embedding, context_embedding, delta_seconds: float = 0.0) -> tf.Tensor:
        latent = tf.cast(latent_state, tf.float32)
        static = tf.cast(static_embedding, tf.float32)
        context = tf.cast(context_embedding, tf.float32)
        decay = tf.exp(-tf.maximum(self.decay_rate, 1e-6) * float(max(delta_seconds, 0.0)))
        vector = tf.concat([latent * decay, static, context], axis=-1)
        return tf.squeeze(self.fusion(tf.expand_dims(vector, axis=0)), axis=0)

