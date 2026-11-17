"""PyTorch AdCSPP V1 response model used by the conditional simulator."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import nn

from inputs.adcsp.schema import ClickResult, ConversionResult, Event, EventType, ExposureContext, UserState
from .encoders import ContextEncoder, ContinuousTimeStateBackbone, SequenceFeatureEncoder, StaticUserEncoder
from .heads import ClickHazardHead, ConversionHazardHead


class AdCSPPModel(nn.Module):
    def __init__(self, latent_dim: int = 16, static_dim: int = 16, context_dim: int = 16, hidden_dim: int = 32, seed: int = 42, feature_config=None):
        super().__init__()
        torch.manual_seed(seed)
        vocab_config = (feature_config or {}).get("categorical_vocab", {})
        self.static_encoder = StaticUserEncoder(output_dim=static_dim, vocabs=vocab_config)
        self.context_encoder = ContextEncoder(output_dim=context_dim, vocabs=vocab_config)
        sequence_names = list((feature_config or {}).get("sequence_features", {}).keys()) or None
        self.sequence_encoder = SequenceFeatureEncoder(output_dim=context_dim, sequence_names=sequence_names)
        self.backbone = ContinuousTimeStateBackbone(latent_dim, static_dim, context_dim, hidden_dim)
        self.click_head = ClickHazardHead(hidden_dim + context_dim)
        self.conversion_head = ConversionHazardHead(hidden_dim + context_dim)

    def encode_history(self, state: UserState, context: Optional[ExposureContext] = None) -> torch.Tensor:
        if not state.static_embedding.any():
            state.static_embedding = self.static_encoder(state.static_features).detach().numpy()
        context_embedding = torch.zeros(self.context_encoder.projection[0].out_features)
        if context is not None:
            context_embedding = self.context_encoder(context)
        history_embedding = self.sequence_encoder(state.sequence_features)
        context_embedding = context_embedding + history_embedding
        return self.backbone(state.latent_state, state.static_embedding, context_embedding, 0.0)

    def click_probability(self, state: UserState, context: ExposureContext, window_seconds: int) -> float:
        state_embedding = self.encode_history(state, context)
        context_embedding = self.context_encoder(context)
        return self.click_head.probability(state_embedding, context_embedding, window_seconds)

    def conversion_probability(self, state: UserState, context: ExposureContext, window_seconds: int) -> float:
        state_embedding = self.encode_history(state, context)
        context_embedding = self.context_encoder(context)
        return self.conversion_head.probability(state_embedding, context_embedding, window_seconds)

    @staticmethod
    def _sample_delay(probability: float, window_seconds: int, rng: np.random.Generator) -> int:
        if probability <= 0.0:
            return window_seconds
        rate = -math.log(max(1.0 - probability, 1e-6)) / max(float(window_seconds), 1.0)
        return max(1, min(window_seconds, int(rng.exponential(1.0 / max(rate, 1e-6))) + 1))

    def sample_click(self, state: UserState, context: ExposureContext, window_seconds: int, rng: np.random.Generator) -> ClickResult:
        probability = self.click_probability(state, context, window_seconds)
        clicked = bool(rng.random() < probability)
        return ClickResult(clicked, probability, self._sample_delay(probability, window_seconds, rng) if clicked else None)

    def sample_conversion(self, state: UserState, context: ExposureContext, window_seconds: int, rng: np.random.Generator) -> ConversionResult:
        probability = self.conversion_probability(state, context, window_seconds)
        converted = bool(rng.random() < probability)
        return ConversionResult(
            converted=converted,
            probability=probability,
            delay_seconds=self._sample_delay(probability, window_seconds, rng) if converted else None,
            conversion_type="purchase",
            conversion_target="purchase",
            conversion_value=1.0 if converted else 0.0,
        )
