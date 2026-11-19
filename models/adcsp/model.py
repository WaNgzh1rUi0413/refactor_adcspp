"""TensorFlow AdCSPP conditional click/conversion behavior model."""

from __future__ import annotations

from typing import Optional

import copy
import numpy as np

import tensorflow as tf

from inputs.adcsp.schema import ExposureContext, UserState
from .encoders import ContextEncoder, ContinuousTimeStateBackbone, SequenceFeatureEncoder, StaticUserEncoder
from .heads import ClickHazardHead, ConversionHazardHead, AdSelectionHead, CandidateMasker
from .state_updater import DynamicStateUpdater
from .state_transition import DynamicStateTransition
from .opportunity import ExposureOpportunityHead


class AdCSPPModel(tf.keras.Model):
    def __init__(self, latent_dim: int = 16, static_dim: int = 16, context_dim: int = 16, hidden_dim: int = 32, seed: int = 42, feature_config=None, policy_dim: int = 8):
        super().__init__(name="adcsp_model")
        tf.keras.utils.set_random_seed(seed)
        feature_config = feature_config or {}
        hash_buckets = int(feature_config.get("hash_buckets", 4096))
        self.latent_dim = int(latent_dim)
        self.policy_dim = int(policy_dim)
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
        self.state_transition = DynamicStateTransition(self.latent_dim, hidden_dim)
        # Backward-compatible alias for checkpoints and integrations that used
        # the old internal attribute name.
        self.transition_net = self.state_transition.network
        self.state_updater = DynamicStateUpdater()
        self.opportunity_head = ExposureOpportunityHead(hidden_dim + context_dim + self.policy_dim, hidden_dim)
        self.ad_selection_head = AdSelectionHead(hidden_dim + context_dim + self.policy_dim, hidden_dim)
        self.candidate_masker = CandidateMasker()

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

    def conversion_probability(self, state: UserState, context: ExposureContext, window_seconds: int, click_outcome: int = 0) -> float:
        state_embedding = self.encode_history(state, context)
        context_embedding = self.context_encoder.call(context)
        return float(tf.reshape(self.conversion_head.probability(state_embedding, context_embedding, window_seconds, event_embedding=[click_outcome]), []).numpy())

    def differentiable_probabilities(
        self,
        state: UserState,
        context: ExposureContext,
        click_window_seconds: int,
        conversion_window_seconds: int,
        latent_state=None,
        click_outcome=None,
    ):
        static_embedding, context_embedding, history_embedding = self._encoded_inputs(state, context)
        latent = state.latent_state if latent_state is None else latent_state
        delta_seconds = 0.0
        if state.last_update_timestamp > 0:
            delta_seconds = max(0, int(context.event_timestamp) - int(state.last_update_timestamp))
        state_embedding = self.backbone.call(latent, static_embedding, context_embedding + history_embedding, delta_seconds)
        click_probability = self.click_head.probability(state_embedding, context_embedding, click_window_seconds)
        click_condition = 0.0 if click_outcome is None else tf.cast(click_outcome, tf.float32)
        conversion_probability = self.conversion_head.probability(
            state_embedding, context_embedding, conversion_window_seconds, event_embedding=[click_condition]
        )
        return tf.reshape(click_probability, []), tf.reshape(conversion_probability, [])

    def differentiable_transition(self, latent_state, state: UserState, context: ExposureContext, is_click, is_conversion):
        """Differentiable latent transition used by sequential training."""
        static_embedding, context_embedding, history_embedding = self._encoded_inputs(state, context)
        return self.state_transition(
            latent_state,
            static_embedding,
            context_embedding,
            history_embedding,
            tf.cast(is_click, tf.float32),
            tf.cast(is_conversion, tf.float32),
            training=True,
        )

    def exposure_opportunity_rate(self, state: UserState, context: ExposureContext, policy_embedding=None) -> float:
        """Return exposure-opportunity intensity in events per second."""
        return float(self.differentiable_opportunity_rate(state, context, policy_embedding).numpy())

    def differentiable_opportunity_rate(self, state: UserState, context: ExposureContext, policy_embedding=None):
        """Differentiable exposure intensity used by the opportunity loss."""
        state_embedding = self.encode_history(state, context)
        context_embedding = self.context_encoder.call(context)
        policy = self._policy_embedding(policy_embedding)
        features = tf.concat([state_embedding, context_embedding, policy], axis=-1)
        return self.opportunity_head(features, training=True)

    def select_candidate(self, state: UserState, candidates, policy_embedding=None, valid_mask=None):
        """Score a candidate pool under the current user state and policy."""
        candidates = list(candidates)
        if not candidates:
            raise ValueError("candidate pool must not be empty")
        logits, probabilities = self.differentiable_candidate_scores(
            state, candidates, policy_embedding=policy_embedding, valid_mask=valid_mask
        )
        selected_index = int(tf.argmax(probabilities, axis=0).numpy())
        return selected_index, logits, probabilities

    def differentiable_candidate_scores(self, state: UserState, candidates, policy_embedding=None, valid_mask=None):
        candidates = list(candidates)
        if not candidates:
            raise ValueError("candidate pool must not be empty")
        state_embedding = self.encode_history(state, candidates[0])
        context_embeddings = tf.stack([self.context_encoder.call(candidate) for candidate in candidates])
        state_embeddings = tf.repeat(tf.expand_dims(state_embedding, axis=0), repeats=len(candidates), axis=0)
        policy = self._policy_embedding(policy_embedding)
        policy_embeddings = tf.repeat(tf.expand_dims(policy, axis=0), repeats=len(candidates), axis=0)
        features = tf.concat([state_embeddings, context_embeddings, policy_embeddings], axis=-1)
        if valid_mask is None:
            valid_mask = tf.ones((len(candidates),), dtype=tf.bool)
        return self.ad_selection_head(features, valid_mask=valid_mask, training=True)

    def _policy_embedding(self, policy_embedding=None):
        if policy_embedding is None:
            return tf.zeros((self.policy_dim,), dtype=tf.float32)
        policy = tf.reshape(tf.cast(policy_embedding, tf.float32), [-1])
        if int(policy.shape[0]) < self.policy_dim:
            policy = tf.pad(policy, [[0, self.policy_dim - int(policy.shape[0])]])
        return policy[: self.policy_dim]

    def transition_event(
        self,
        state: UserState,
        context: ExposureContext,
        event_type: str,
        timestamp: Optional[int] = None,
    ) -> UserState:
        """Apply one event at a time for the concurrent scheduler."""
        event_type = str(event_type).upper()
        if event_type not in {"EXPOSURE", "CLICK", "CONVERSION", "CONVERSION_EXPIRED", "NO_CLICK_EXPIRED"}:
            raise ValueError(f"Unsupported event type: {event_type}")
        is_click = int(event_type == "CLICK")
        is_conversion = int(event_type == "CONVERSION")
        static_embedding, context_embedding, history_embedding = self._encoded_inputs(state, context)
        latent = tf.convert_to_tensor(state.latent_state, dtype=tf.float32)
        next_latent = self.differentiable_transition(latent, state, context, is_click, is_conversion)
        next_state = copy.deepcopy(state)
        next_state.latent_state = np.asarray(next_latent.numpy(), dtype=np.float32)
        event_timestamp = int(timestamp if timestamp is not None else context.event_timestamp)
        next_state.last_update_timestamp = event_timestamp
        if event_type == "EXPOSURE":
            self.state_updater.apply_exposure(next_state, context)
        elif event_type == "CLICK":
            self.state_updater.apply_click(next_state, context)
        elif event_type == "CONVERSION":
            self.state_updater.apply_conversion(next_state, context)
        elif event_type == "NO_CLICK_EXPIRED":
            self.state_updater.apply_no_click(next_state, context)
        return next_state

    def transition_state(
        self,
        state: UserState,
        context: ExposureContext,
        is_click: int,
        is_conversion: int,
        timestamp: Optional[int] = None,
        click_timestamp: Optional[int] = None,
        conversion_timestamp: Optional[int] = None,
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
        final_timestamp = conversion_timestamp or click_timestamp or timestamp or context.event_timestamp
        next_state.last_update_timestamp = int(final_timestamp)
        self.state_updater.apply_outcome(
            next_state,
            context,
            int(is_click),
            int(is_conversion),
            click_timestamp=click_timestamp,
            conversion_timestamp=conversion_timestamp,
        )
        return next_state


def _append_state_history(state: UserState, context: ExposureContext, is_click: int, is_conversion: int, max_len: int = 20) -> None:
    """Backward-compatible wrapper around the explicit state updater."""
    DynamicStateUpdater(max_sequence_length=max_len).apply_outcome(state, context, is_click, is_conversion)


