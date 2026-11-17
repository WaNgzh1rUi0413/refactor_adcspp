"""Point-process losses for future model training."""

from __future__ import annotations

import torch
from torch import nn


class AdCSPPSurvivalLoss(nn.Module):
    """Masked Bernoulli loss for V1 click and conversion supervision.

    The simulator uses synthetic sampling today; this loss defines the
    training contract for later real event-log integration.
    """

    def __init__(self, conversion_weight: float = 2.0):
        super().__init__()
        self.conversion_weight = conversion_weight

    def forward(self, click_probability, click_label, conversion_probability, conversion_label, mask=None):
        click_probability = torch.clamp(click_probability, 1e-6, 1 - 1e-6)
        conversion_probability = torch.clamp(conversion_probability, 1e-6, 1 - 1e-6)
        click_loss = nn.functional.binary_cross_entropy(click_probability, click_label.float(), reduction="none")
        conversion_loss = nn.functional.binary_cross_entropy(conversion_probability, conversion_label.float(), reduction="none")
        if mask is None:
            mask = torch.ones_like(click_loss)
        mask = mask.float()
        click_loss = (click_loss * mask).sum() / mask.sum().clamp_min(1.0)
        conversion_loss = (conversion_loss * mask).sum() / mask.sum().clamp_min(1.0)
        total = click_loss + self.conversion_weight * conversion_loss
        return {"loss": total, "click_loss": click_loss, "conversion_loss": conversion_loss}
