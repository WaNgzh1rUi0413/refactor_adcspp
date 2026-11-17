"""Click and conversion hazard heads."""

from __future__ import annotations

import math

import torch
from torch import nn


class _HazardHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, bias: float = -3.0):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        nn.init.constant_(self.network[-1].bias, bias)

    def rate(self, state_embedding, context_embedding):
        state = torch.as_tensor(state_embedding, dtype=torch.float32)
        context = torch.as_tensor(context_embedding, dtype=torch.float32)
        return torch.nn.functional.softplus(self.network(torch.cat([state, context], dim=-1))).squeeze(-1)

    def probability(self, state_embedding, context_embedding, window_seconds: int) -> float:
        rate = float(self.rate(state_embedding, context_embedding).detach().cpu())
        return float(1.0 - math.exp(-rate * float(window_seconds) / 60.0))


class ClickHazardHead(_HazardHead):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__(input_dim, hidden_dim, bias=-1.0)


class ConversionHazardHead(_HazardHead):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        # Keep the synthetic post-click conversion rate below 1.0 by default;
        # the value remains learnable when real logs are introduced.
        super().__init__(input_dim, hidden_dim, bias=-6.0)
