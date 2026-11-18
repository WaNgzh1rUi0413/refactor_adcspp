"""TensorFlow encoders for online AdCSPP behavior modeling."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Mapping, Sequence

import tensorflow as tf

from inputs.adcsp.schema import ExposureContext, UserProfile


def _hash_index(field: str, value: object, num_buckets: int) -> int:
    digest = hashlib.sha1(f"{field}::{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % int(num_buckets)


def _safe_event_timestamp(value: object) -> int:
    """Bound a logid-like value before it enters a TensorFlow numeric op."""
    text = str(value).strip()
    digits = text[1:] if text.startswith("-") else text
    if len(digits) >= 14 and digits[:14].isdigit():
        try:
            parsed = datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
            return int(parsed.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            pass
    try:
        numeric = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return 0
    return numeric if -(2**63) <= numeric <= 2**63 - 1 else 0


class StaticUserEncoder(tf.keras.layers.Layer):
    def __init__(self, output_dim: int = 16, embedding_dim: int = 8, hash_buckets: int = 4096,
                 categorical_fields=None, numeric_fields=None):
        super().__init__()
        self.hash_buckets = int(hash_buckets)
        self.fields = list(categorical_fields if categorical_fields is not None else [
            "normalized_tag2_id", "unifiled_tag_level1_all_ssp", "unifiled_tag_level2_all_ssp",
            "unifiled_tag_level3", "app_class1", "app_class2", "app_class3",
        ])
        self.numeric_fields = list(numeric_fields if numeric_fields is not None else [
            "auid_site_click_cnt_90d", "auid_normalizedt2_click_cnt_180d", "auid_ctr_90d",
            "auid_show_cnt_180d", "auid_app_show_cnt_180d", "auid_app_click_cnt_90d",
            "auid_app_ctr_90d", "auid_appc3_ctr_180d", "auid_slot_show_cnt_90d",
            "auid_slotid_imp_cnt_30d", "auid_slot_ctr_90d", "auid_slot_click_cnt_90d",
            "auid_site_show_cnt_90d", "auid_site_ctr_90d", "auid_normalizedt2_ctr_90d",
        ])
        for field in self.fields:
            setattr(self, f"embedding_{field}", tf.keras.layers.Embedding(self.hash_buckets, embedding_dim))
        self.interest_embedding = tf.keras.layers.Embedding(self.hash_buckets, embedding_dim)
        self.projection = tf.keras.Sequential([
            tf.keras.layers.Dense(output_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("tanh"),
        ])

    def call(self, profile: UserProfile | Mapping[str, object]) -> tf.Tensor:
        values = profile.to_dict() if isinstance(profile, UserProfile) else dict(profile)
        values = {**(values.get("features", {}) or {}), **values}
        encoded = []
        for field in self.fields:
            index = tf.constant([_hash_index(field, values.get(field, "unknown"), self.hash_buckets)], dtype=tf.int32)
            encoded.append(tf.squeeze(getattr(self, f"embedding_{field}")(index), axis=0))
        interests = values.get("unifiled_tag_level3", ()) or ()
        interests = interests if isinstance(interests, (list, tuple)) else [interests]
        if interests and interests != [None]:
            indices = tf.constant([_hash_index("unifiled_tag_level3", item, self.hash_buckets) for item in interests], dtype=tf.int32)
            interest = tf.reduce_mean(self.interest_embedding(indices), axis=0)
        else:
            interest = tf.zeros((self.interest_embedding.output_dim,), dtype=tf.float32)
        numeric = []
        for field in self.numeric_fields:
            value = values.get(field, 0.0)
            if value is None:
                value = 0.0
            if isinstance(value, (list, tuple)):
                value = value[0] if value else 0.0
            numeric.append(tf.reshape(tf.cast(value, tf.float32), [1]))
        vector = tf.concat(encoded + [interest] + numeric, axis=0)
        return tf.squeeze(self.projection(tf.expand_dims(vector, axis=0)), axis=0)


class ContextEncoder(tf.keras.layers.Layer):
    def __init__(self, output_dim: int = 16, embedding_dim: int = 6, hash_buckets: int = 4096,
                 categorical_fields=None, numeric_fields=None):
        super().__init__()
        self.hash_buckets = int(hash_buckets)
        self.fields = list(categorical_fields if categorical_fields is not None else [
            "slot_id", "slot_form", "slot_suited_creative_type", "slot_star_ranking_cn",
            "developer_up_id", "task_name", "indu_name", "objective_type",
            "promotion_target", "creative_image_size", "creativetype", "app_class1",
            "app_class2", "app_class3", "app_type", "media_type_new",
        ])
        self.numeric_fields = list(numeric_fields if numeric_fields is not None else [
            "slot_quality_score", "slot_avg_click_bid_ecpm_3d", "slot_ctr_3rdpty_dsp_7d",
            "slot_ctr_3rdpty_dsp_30d", "ecpm_floor_price", "is_video", "is_short_video",
            "site_click_cnt_30d", "site_paid_cnt_30d", "site_paid_rate_smooth_3d",
            "site_activate_cnt_14d", "site_browse_cnt_14d", "site_click_cnt_14d",
            "site_register_count_14d", "site_retain_cnt_14d", "site_show_cnt_14d",
            "site_register_hour_top_30d", "site_paid_hour_top3_30d",
        ])
        for field in self.fields:
            setattr(self, f"embedding_{field}", tf.keras.layers.Embedding(self.hash_buckets, embedding_dim))
        self.projection = tf.keras.Sequential([
            tf.keras.layers.Dense(output_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("tanh"),
        ])

    def call(self, context: ExposureContext | Mapping[str, object]) -> tf.Tensor:
        values = context.to_dict() if isinstance(context, ExposureContext) else dict(context)
        values = {**(values.get("features", {}) or {}), **values}
        encoded = []
        for field in self.fields:
            index = tf.constant([_hash_index(field, values.get(field, "unknown"), self.hash_buckets)], dtype=tf.int32)
            encoded.append(tf.squeeze(getattr(self, f"embedding_{field}")(index), axis=0))
        numeric_values = []
        for field in self.numeric_fields:
            value = values.get(field, 0.0)
            if value is None:
                value = 0.0
            if isinstance(value, (list, tuple)):
                value = value[0] if value else 0.0
            numeric_values.append(tf.cast(value, tf.float32))
        numeric_values.append(tf.cast(
            _safe_event_timestamp(values.get("event_timestamp", values.get("logid", 0))),
            tf.float32,
        ) % 86400.0 / 86400.0)
        numeric = tf.stack(numeric_values)
        vector = tf.concat(encoded + [numeric], axis=0)
        return tf.squeeze(self.projection(tf.expand_dims(vector, axis=0)), axis=0)


class SequenceFeatureEncoder(tf.keras.layers.Layer):
    def __init__(self, output_dim: int = 16, embedding_dim: int = 8, num_embeddings: int = 256, sequence_names=None):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.sequence_names = list(sequence_names or [
            "rt_auid_hwdsp_imp_unclk_3rd_indid_seq_time_12h",
            "auid_hwdsp_clk_slotid_seq_time_365d",
            "auid_hwdsp_clk_ts_seq_time_365d",
            "auid_hwdsp_conv_slotid_seq_time_365d",
            "auid_hwdsp_conv_ts_seq_time_365d",
            "auid_hwdsp_conv_evttyp_seq_time_365d",
            "auid_hwdsp_conv_ocpx_trgt_seq_time_365d",
        ])
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

