"""Static, context and continuous-time state encoders."""

from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from inputs.adcsp.schema import ExposureContext, UserProfile


DEFAULT_VOCABS = {
    "gender": ["unknown", "female", "male"],
    "age_group": ["unknown", "18_24", "25_34", "35_44", "45_plus"],
    "region": ["unknown", "north", "east", "south", "west"],
    "device_type": ["unknown", "phone", "tablet", "desktop"],
    "os_type": ["unknown", "android", "ios", "windows"],
    "user_segment": ["unknown", "new", "active", "high_intent"],
    "user_value_segment": ["unknown", "low", "medium", "high"],
    "slot_id": ["unknown", "slot_a", "slot_b", "slot_c"],
    "slot_type": ["unknown", "feed", "search", "detail"],
    "slot_position": ["unknown", "top", "middle", "bottom"],
    "ad_category": ["unknown", "commerce", "content", "finance", "travel"],
    "creative_type": ["unknown", "image", "video", "native"],
    "creative_size": ["unknown", "square", "portrait", "landscape"],
    "objective_type": ["unknown", "click", "activate", "purchase"],
    "landing_page_type": ["unknown", "product", "content", "form"],
    "promotion_target": ["unknown", "purchase", "activate", "signup"],
    "traffic_source": ["unknown", "organic", "search", "push"],
    "app_category": ["unknown", "shopping"],
    "page_context": ["unknown", "home", "search", "detail"],
    "network_type": ["unknown", "wifi", "4g", "5g"],
}


def merge_vocabs(vocabs=None):
    merged = {key: list(values) for key, values in DEFAULT_VOCABS.items()}
    for key, values in (vocabs or {}).items():
        merged[key] = list(values)
    return merged


def _index(vocab: Sequence[str], value: str) -> int:
    try:
        return vocab.index(value)
    except ValueError:
        return 0


class StaticUserEncoder(nn.Module):
    def __init__(self, output_dim: int = 16, embedding_dim: int = 8, vocabs=None):
        super().__init__()
        self.vocabs = merge_vocabs(vocabs)
        fields = ["gender", "age_group", "region", "device_type", "os_type", "user_segment", "user_value_segment"]
        self.fields = fields
        self.embeddings = nn.ModuleDict({field: nn.Embedding(len(self.vocabs[field]), embedding_dim) for field in fields})
        self.interest_embedding = nn.Embedding(len(self.vocabs["ad_category"]), embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(len(fields) * embedding_dim + embedding_dim + 1, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh(),
        )

    def forward(self, profile: UserProfile | Mapping[str, object]) -> torch.Tensor:
        values = profile.to_dict() if isinstance(profile, UserProfile) else profile
        encoded = [self.embeddings[field](torch.tensor(_index(self.vocabs[field], str(values.get(field, "unknown"))))) for field in self.fields]
        interests = values.get("user_long_term_interest", ()) or ()
        interest_ids = [_index(self.vocabs["ad_category"], str(item)) for item in interests]
        if interest_ids:
            interest = self.interest_embedding(torch.tensor(interest_ids)).mean(dim=0)
        else:
            interest = torch.zeros(self.interest_embedding.embedding_dim)
        age = torch.tensor([float(values.get("user_registration_age", 0.0)) / 3_650.0])
        return self.projection(torch.cat(encoded + [interest, age], dim=0)).float()


class ContextEncoder(nn.Module):
    def __init__(self, output_dim: int = 16, embedding_dim: int = 6, vocabs=None):
        super().__init__()
        self.vocabs = merge_vocabs(vocabs)
        fields = ["slot_id", "slot_type", "slot_position", "ad_category", "creative_type", "creative_size", "objective_type", "landing_page_type", "promotion_target", "traffic_source", "app_category", "page_context", "network_type"]
        self.fields = fields
        self.embeddings = nn.ModuleDict({field: nn.Embedding(len(self.vocabs[field]), embedding_dim) for field in fields})
        self.projection = nn.Sequential(nn.Linear(len(fields) * embedding_dim + 5, output_dim), nn.LayerNorm(output_dim), nn.Tanh())

    def forward(self, context: ExposureContext | Mapping[str, object]) -> torch.Tensor:
        values = context.to_dict() if isinstance(context, ExposureContext) else context
        encoded = [self.embeddings[field](torch.tensor(_index(self.vocabs[field], str(values.get(field, "unknown"))))) for field in self.fields]
        numeric = torch.tensor([
            float(values.get("bid_price", 0.0)),
            float(values.get("ecpm", 0.0)),
            float(values.get("ad_quality_score", 0.0)),
            float(values.get("is_holiday", False)),
            float(values.get("event_timestamp", 0) % 86_400) / 86_400.0,
        ])
        return self.projection(torch.cat(encoded + [numeric], dim=0)).float()


class SequenceFeatureEncoder(nn.Module):
    """Encode model-ready historical list features such as hist_clk_slots."""

    def __init__(self, output_dim: int = 16, embedding_dim: int = 8, num_embeddings: int = 256, sequence_names=None):
        super().__init__()
        self.output_dim = output_dim
        self.num_embeddings = num_embeddings
        self.sequence_names = list(sequence_names or ["hist_exp_slots", "hist_exp_ads", "hist_clk_slots", "hist_clk_ads", "hist_clk_categories", "hist_conv_slots", "hist_conv_targets"])
        self.embeddings = nn.ModuleDict({name: nn.Embedding(num_embeddings, embedding_dim) for name in self.sequence_names})
        self.projection = nn.Sequential(nn.Linear(len(self.sequence_names) * embedding_dim, output_dim), nn.LayerNorm(output_dim), nn.Tanh())

    def _index(self, value: object) -> int:
        digest = hashlib.sha1(str(value).encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.num_embeddings

    def forward(self, sequence_features: Mapping[str, Sequence[object]] | None) -> torch.Tensor:
        values = sequence_features or {}
        encoded = []
        for name in self.sequence_names:
            items = list(values.get(name, []))
            if items:
                indices = torch.tensor([self._index(item) for item in items], dtype=torch.long)
                encoded.append(self.embeddings[name](indices).mean(dim=0))
            else:
                encoded.append(torch.zeros(self.embeddings[name].embedding_dim))
        return self.projection(torch.cat(encoded, dim=0)).float()


class ContinuousTimeStateBackbone(nn.Module):
    """Compact continuous-time state update used for the local V1 simulator."""

    def __init__(self, latent_dim: int = 16, static_dim: int = 16, context_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.decay_rate = nn.Parameter(torch.tensor(1.0 / 3_600.0))
        self.fusion = nn.Sequential(nn.Linear(latent_dim + static_dim + context_dim, hidden_dim), nn.Tanh())
        self.event_projection = nn.Linear(4, latent_dim)

    def forward(self, latent_state, static_embedding, context_embedding, delta_seconds: float = 0.0):
        latent = torch.as_tensor(latent_state, dtype=torch.float32)
        static = torch.as_tensor(static_embedding, dtype=torch.float32)
        context = torch.as_tensor(context_embedding, dtype=torch.float32)
        decay = torch.exp(-torch.clamp(self.decay_rate, min=1e-6) * float(max(delta_seconds, 0.0)))
        return self.fusion(torch.cat([latent * decay, static, context], dim=-1))

    def event_jump(self, latent_state, event_type: int) -> np.ndarray:
        one_hot = torch.zeros(4)
        one_hot[int(event_type)] = 1.0
        jump = self.event_projection(one_hot).detach().numpy()
        return np.asarray(latent_state, dtype=np.float32) + jump.astype(np.float32)
