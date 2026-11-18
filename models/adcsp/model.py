"""TensorFlow AdCSPP conditional click/conversion behavior model."""

from __future__ import annotations

from typing import Optional

import copy
import numpy as np

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
        self.static_encoder = StaticUserEncoder(
            output_dim=static_dim,
            hash_buckets=hash_buckets,
            categorical_fields=feature_config.get("static_categorical_fields"),
            numeric_fields=feature_config.get("static_numeric_fields"),
        )
        self.context_encoder = ContextEncoder(
            output_dim=context_dim,
            hash_buckets=hash_buckets,
            categorical_fields=feature_config.get("context_categorical_fields"),
            numeric_fields=feature_config.get("context_numeric_fields"),
        )
        sequence_names = list(feature_config.get("sequence_features", {}).keys())
        sequence_names = sequence_names or list(feature_config.get("sequence_fields", [])) or None
        self.sequence_encoder = SequenceFeatureEncoder(output_dim=context_dim, sequence_names=sequence_names)
        self.backbone = ContinuousTimeStateBackbone(latent_dim, static_dim, context_dim, hidden_dim)
        self.click_head = ClickHazardHead(hidden_dim + context_dim)
        self.conversion_head = ConversionHazardHead(hidden_dim + context_dim)
        # The transition is deliberately separate from the behavior heads.
        # The heads learn p(click | state, request) and p(conversion | state,
        # request, click); this network learns how the observed event changes
        # the latent user state for the next request.
        self.transition_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="tanh"),
            tf.keras.layers.Dense(self.latent_dim),
        ], name="state_transition")

    def encode_history(self, state: UserState, context: Optional[ExposureContext] = None) -> tf.Tensor:
        static_embedding = self.static_encoder.call(state.static_features)
        if context is None:
            context_embedding = tf.zeros((self.context_encoder.projection.layers[0].units,), dtype=tf.float32)
        else:
            context_embedding = self.context_encoder.call(context)
        history_embedding = self.sequence_encoder.call(state.sequence_features)
        return self.backbone.call(state.latent_state, static_embedding, context_embedding + history_embedding, 0.0)

    def _encoded_inputs(self, state: UserState, context: ExposureContext):
        static_embedding = self.static_encoder.call(state.static_features)
        context_embedding = self.context_encoder.call(context)
        history_embedding = self.sequence_encoder.call(state.sequence_features)
        return static_embedding, context_embedding, history_embedding

    def click_probability(self, state: UserState, context: ExposureContext, window_seconds: int) -> float:
        state_embedding = self.encode_history(state, context)
        context_embedding = self.context_encoder.call(context)
        return float(tf.reshape(self.click_head.probability(state_embedding, context_embedding, window_seconds), []).numpy())

    def conversion_probability(self, state: UserState, context: ExposureContext, window_seconds: int) -> float:
        state_embedding = self.encode_history(state, context)
        context_embedding = self.context_encoder.call(context)
        return float(tf.reshape(self.conversion_head.probability(state_embedding, context_embedding, window_seconds), []).numpy())

    def differentiable_probabilities(
        self,
        state: UserState,
        context: ExposureContext,
        click_window_seconds: int,
        conversion_window_seconds: int,
        latent_state=None,
    ):
        static_embedding, context_embedding, history_embedding = self._encoded_inputs(state, context)
        latent = state.latent_state if latent_state is None else latent_state
        state_embedding = self.backbone.call(latent, static_embedding, context_embedding + history_embedding, 0.0)
        click_probability = self.click_head.probability(state_embedding, context_embedding, click_window_seconds)
        conversion_probability = self.conversion_head.probability(state_embedding, context_embedding, conversion_window_seconds)
        return tf.reshape(click_probability, []), tf.reshape(conversion_probability, [])

    def differentiable_transition(self, latent_state, state: UserState, context: ExposureContext, is_click, is_conversion):
        """Differentiable latent transition used by sequential training."""
        static_embedding, context_embedding, history_embedding = self._encoded_inputs(state, context)
        event = tf.stack([tf.cast(is_click, tf.float32), tf.cast(is_conversion, tf.float32)])
        transition_input = tf.concat([
            tf.cast(latent_state, tf.float32), static_embedding, context_embedding,
            history_embedding, event,
        ], axis=-1)
        return self.transition_net(tf.expand_dims(transition_input, axis=0), training=True)[0]

    def transition_state(
        self,
        state: UserState,
        context: ExposureContext,
        is_click: int,
        is_conversion: int,
        timestamp: Optional[int] = None,
    ) -> UserState:
        """Apply one observed/sampled event and return the next user state.

        This method is used by both offline trajectory simulation and online
        path generation.  It updates the learned latent state as well as the
        observable sequence features, so the next generated request sees the
        same state transition semantics as the next training example.
        """
        latent = tf.convert_to_tensor(state.latent_state, dtype=tf.float32)
        next_latent = self.differentiable_transition(latent, state, context, is_click, is_conversion)

        next_state = copy.deepcopy(state)
        next_state.latent_state = np.asarray(next_latent.numpy(), dtype=np.float32)
        next_state.last_update_timestamp = int(timestamp if timestamp is not None else context.event_timestamp)
        _append_state_history(next_state, context, int(is_click), int(is_conversion))
        return next_state


def _append_state_history(state: UserState, context: ExposureContext, is_click: int, is_conversion: int, max_len: int = 20) -> None:
    """Update production sequence features and aggregate counters."""

    def append(name: str, value: object) -> None:
        values = list(state.sequence_features.get(name, []))
        values.append(value)
        values = values[-max_len:]
        state.sequence_features[name] = list(values)

    if is_click:
        append("auid_hwdsp_clk_slotid_seq_time_365d", context.slot_id)
        append("auid_hwdsp_clk_ts_seq_time_365d", int(context.event_timestamp))
    else:
        # Core negative-feedback sequence used by the first online schema.
        category = context.features.get("indu_name", "unknown")
        append("rt_auid_hwdsp_imp_unclk_3rd_indid_seq_time_12h", category)
    if is_conversion:
        target = context.features.get("promotion_target", "unknown")
        append("auid_hwdsp_conv_slotid_seq_time_365d", context.slot_id)
        append("auid_hwdsp_conv_ts_seq_time_365d", int(context.event_timestamp))
        append("auid_hwdsp_conv_evttyp_seq_time_365d", context.features.get("crtvid_conversion_target", target))
        append("auid_hwdsp_conv_ocpx_trgt_seq_time_365d", target)

    dynamic = state.dynamic_features
    dynamic["exposure_count"] = int(dynamic.get("exposure_count", 0)) + 1
    dynamic["click_count"] = int(dynamic.get("click_count", 0)) + int(bool(is_click))
    dynamic["conversion_count"] = int(dynamic.get("conversion_count", 0)) + int(bool(is_conversion))
    dynamic["last_exposure_slot"] = context.slot_id
    if is_click:
        dynamic["last_click_slot"] = context.slot_id
        dynamic["last_click_timestamp"] = int(context.event_timestamp)
    if is_conversion:
        dynamic["last_conversion_timestamp"] = int(context.event_timestamp)


