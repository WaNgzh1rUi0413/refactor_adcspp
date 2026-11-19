"""TensorFlow click and conversion hazard heads."""

from __future__ import annotations

import tensorflow as tf
import numpy as np
from typing import Iterable, Mapping, Sequence

from losses.adcsp_losses import AdSelectionLoss


class _HazardHead(tf.keras.layers.Layer):
    def __init__(self, input_dim: int, hidden_dim: int = 32, bias: float = -3.0):
        super().__init__()
        self.network = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="tanh"),
            tf.keras.layers.Dense(1, bias_initializer=tf.keras.initializers.Constant(bias)),
        ])

    def rate(self, state_embedding, context_embedding, event_embedding=None):
        state = tf.cast(state_embedding, tf.float32)
        context = tf.cast(context_embedding, tf.float32)
        vector = tf.concat([state, context], axis=-1)
        if event_embedding is not None:
            event = tf.reshape(tf.cast(event_embedding, tf.float32), [-1])
            vector = tf.concat([vector, event], axis=-1)
        return tf.squeeze(tf.nn.softplus(self.network(tf.expand_dims(vector, axis=0))), axis=-1)

    def probability(self, state_embedding, context_embedding, window_seconds: int, event_embedding=None):
        rate = self.rate(state_embedding, context_embedding, event_embedding=event_embedding)
        return tf.reshape(1.0 - tf.exp(-rate * float(window_seconds) / 60.0), [])


class ClickHazardHead(_HazardHead):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__(input_dim, hidden_dim, bias=-1.0)


class ConversionHazardHead(_HazardHead):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__(input_dim, hidden_dim, bias=-6.0)


def _candidate_values(candidate) -> dict:
    if hasattr(candidate, "to_dict"):
        payload = candidate.to_dict()
    elif isinstance(candidate, Mapping):
        payload = dict(candidate)
    else:
        payload = dict(vars(candidate))
    return {**(payload.get("features", {}) or {}), **payload}


class CandidateMasker:
    """Apply deterministic slot, block-list, and frequency constraints."""

    def mask(
        self,
        candidates: Sequence,
        *,
        required_slot_id: str | None = None,
        blocked_ad_ids: Iterable[str] | None = None,
        frequency_counts: Mapping[str, int] | None = None,
        max_frequency: int | None = None,
    ) -> np.ndarray:
        blocked = {str(value) for value in (blocked_ad_ids or ())}
        counts = {str(key): int(value) for key, value in (frequency_counts or {}).items()}
        valid = []
        for candidate in candidates:
            values = _candidate_values(candidate)
            slot_id = str(values.get("slot_id", "unknown"))
            ad_id = str(values.get("ad_id", values.get("adid", values.get("creative_id", "unknown"))))
            allowed = True
            if required_slot_id is not None and slot_id != str(required_slot_id):
                allowed = False
            if ad_id in blocked:
                allowed = False
            if max_frequency is not None and counts.get(ad_id, 0) >= int(max_frequency):
                allowed = False
            valid.append(allowed)
        if not valid or not any(valid):
            raise ValueError("candidate mask removed every candidate")
        return np.asarray(valid, dtype=bool)


class AdSelectionHead(tf.keras.layers.Layer):
    """Score candidate embeddings and return masked logits/probabilities."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, name: str = "ad_selection_head"):
        super().__init__(name=name)
        self.network = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(int(hidden_dim), activation="tanh"),
                tf.keras.layers.Dense(1),
            ],
            name="candidate_score_network",
        )

    def call(self, candidate_features, valid_mask=None, training: bool = False):
        features = tf.cast(candidate_features, tf.float32)
        logits = tf.squeeze(self.network(features, training=training), axis=-1)
        if valid_mask is not None:
            valid_mask = tf.cast(valid_mask, tf.bool)
            logits = tf.where(valid_mask, logits, tf.constant(-1e9, dtype=logits.dtype))
        probabilities = tf.nn.softmax(logits)
        return logits, probabilities


class CandidateSelectionTrainer:
    """Train candidate marks when upstream provides candidate sets and labels."""

    def fit(self, model, examples: Sequence[Mapping], epochs: int = 1, learning_rate: float = 1e-3) -> dict:
        if not examples:
            raise ValueError("candidate selection training requires examples")
        first = examples[0]
        model.differentiable_candidate_scores(
            first["state"], first["candidates"],
            policy_embedding=first.get("policy_embedding"),
            valid_mask=first.get("valid_mask"),
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))
        loss_fn = AdSelectionLoss()
        last_loss = None
        for _ in range(max(1, int(epochs))):
            with tf.GradientTape() as tape:
                losses = []
                for example in examples:
                    logits, _ = model.differentiable_candidate_scores(
                        example["state"], example["candidates"],
                        policy_embedding=example.get("policy_embedding"),
                        valid_mask=example.get("valid_mask"),
                    )
                    losses.append(loss_fn(logits, example["selected_index"], example.get("valid_mask")))
                loss = tf.reduce_mean(tf.stack(losses))
            variables = model.ad_selection_head.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients((gradient, variable) for gradient, variable in zip(gradients, variables) if gradient is not None)
            last_loss = float(loss.numpy())
        return {"selection_loss": last_loss, "selection_examples": len(examples)}

