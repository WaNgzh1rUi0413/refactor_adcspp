"""Hierarchical exposure outcome generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from inputs.adcsp.schema import ExposureContext, UserState


@dataclass(frozen=True)
class EventOutcome:
    click_probability: float
    conversion_probability: float
    is_click: int
    is_conversion: int
    click_timestamp: Optional[int] = None
    conversion_timestamp: Optional[int] = None


class HierarchicalEventGenerator:
    """Generate click first, then conditionally generate conversion."""

    def __init__(self, model, click_window_seconds: int = 30, conversion_window_seconds: int = 3600, seed: int = 42):
        self.model = model
        self.click_window_seconds = int(click_window_seconds)
        self.conversion_window_seconds = int(conversion_window_seconds)
        self.rng = np.random.default_rng(seed)

    def generate(self, state: UserState, context: ExposureContext, sample: bool = True, rng=None) -> EventOutcome:
        random = rng or self.rng
        try:
            click_probability, _ = self.model.differentiable_probabilities(
                state, context, self.click_window_seconds, self.conversion_window_seconds, click_outcome=0
            )
        except TypeError:
            click_probability, _ = self.model.differentiable_probabilities(
                state, context, self.click_window_seconds, self.conversion_window_seconds
            )
        p_click = float(np.clip(click_probability.numpy(), 0.0, 1.0))
        is_click = int(random.random() < p_click) if sample else int(p_click >= 0.5)
        click_timestamp = None
        if is_click:
            click_offset = self._sample_event_offset(p_click, self.click_window_seconds, random, sample)
            click_timestamp = int(context.event_timestamp) + click_offset
        try:
            _, conversion_probability = self.model.differentiable_probabilities(
                state, context, self.click_window_seconds, self.conversion_window_seconds, click_outcome=is_click
            )
        except TypeError:
            _, conversion_probability = self.model.differentiable_probabilities(
                state, context, self.click_window_seconds, self.conversion_window_seconds
            )
        p_conversion = float(np.clip(conversion_probability.numpy(), 0.0, 1.0))
        is_conversion = int(is_click and (random.random() < p_conversion if sample else p_conversion >= 0.5))
        conversion_timestamp = None
        if is_conversion:
            conversion_offset = self._sample_event_offset(p_conversion, self.conversion_window_seconds, random, sample)
            conversion_timestamp = int(click_timestamp) + conversion_offset
        return EventOutcome(p_click, p_conversion, is_click, is_conversion, click_timestamp, conversion_timestamp)

    @staticmethod
    def _sample_event_offset(probability: float, window_seconds: int, random, sample: bool) -> int:
        """Sample an event time from a constant hazard calibrated to p(window).

        The production model currently predicts window-level probabilities.  A
        truncated exponential turns that probability into a valid event time
        without introducing a second hand-written timing distribution.
        """
        if not sample:
            return 0
        probability = float(np.clip(probability, 1e-8, 1.0 - 1e-8))
        window = max(1, int(window_seconds))
        rate = -np.log1p(-probability) / float(window)
        uniform = float(random.random())
        offset = -np.log(max(1e-12, 1.0 - uniform * probability)) / rate
        return int(np.clip(round(offset), 0, window))

