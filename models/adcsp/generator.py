"""Continuous user-path generation for the AdCSPP behavior model.

The generator does not choose ads.  A request/context sequence is supplied by
the caller (the online platform supplies real requests; the offline fixture
creates them).  AdCSPP samples the user's click and, conditional on a click,
conversion outcome, then feeds the event back into the state before the next
request.  This is the missing link between per-row probability prediction and
high-fidelity path generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from inputs.adcsp.schema import ExposureContext, UserProfile, UserState
from models.adcsp.state import UserStateBuilder
from .model import AdCSPPModel


@dataclass(frozen=True)
class GeneratedStep:
    user_id: str
    timestamp: int
    context: ExposureContext
    click_probability: float
    conversion_probability: float
    is_click: int
    is_conversion: int
    state_before: UserState
    state_after: UserState

    def to_dict(self) -> dict:
        """Return an in-memory platform-neutral result; no file serialization."""
        return {
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "context": self.context.to_dict(),
            "click_probability": self.click_probability,
            "conversion_probability": self.conversion_probability,
            "is_click": self.is_click,
            "is_conversion": self.is_conversion,
            "state_before": self.state_before,
            "state_after": self.state_after,
        }


class AdCSPPPathGenerator:
    """Roll out complete paths from a trained AdCSPP model."""

    def __init__(
        self,
        model: AdCSPPModel,
        click_window_seconds: int = 30,
        conversion_window_seconds: int = 3600,
        max_sequence_length: int = 20,
        seed: int = 42,
    ):
        self.model = model
        self.click_window_seconds = int(click_window_seconds)
        self.conversion_window_seconds = int(conversion_window_seconds)
        self.max_sequence_length = int(max_sequence_length)
        self.rng = np.random.default_rng(seed)

    def rollout(
        self,
        state: UserState,
        contexts: Iterable[ExposureContext],
        sample: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> List[GeneratedStep]:
        """Generate one continuous path from a supplied request sequence.

        ``state`` is copied before the first step.  Every returned step keeps
        both snapshots, which makes the state refresh and event constraints
        directly testable without any intermediate files.
        """
        current = state.clone()
        random = rng or self.rng
        result: List[GeneratedStep] = []
        for context in contexts:
            before = current.clone()
            click_probability, conversion_probability = self.model.differentiable_probabilities(
                current, context, self.click_window_seconds, self.conversion_window_seconds
            )
            p_click = float(click_probability.numpy())
            p_conversion = float(conversion_probability.numpy())
            p_click = float(np.clip(p_click, 0.0, 1.0))
            p_conversion = float(np.clip(p_conversion, 0.0, 1.0))
            is_click = int(random.random() < p_click) if sample else int(p_click >= 0.5)
            # A conversion is causally downstream of a click.  This hard
            # constraint prevents impossible conversion-only trajectories.
            is_conversion = int(is_click and (random.random() < p_conversion if sample else p_conversion >= 0.5))
            current = self.model.transition_state(current, context, is_click, is_conversion, context.event_timestamp)
            _truncate_histories(current, self.max_sequence_length)
            result.append(GeneratedStep(
                user_id=current.user_id,
                timestamp=int(context.event_timestamp),
                context=context,
                click_probability=p_click,
                conversion_probability=p_conversion,
                is_click=is_click,
                is_conversion=is_conversion,
                state_before=before,
                state_after=current.clone(),
            ))
        return result

    def generate(
        self,
        state: Optional[UserState] = None,
        prior=None,
        num_steps: Optional[int] = None,
        start_timestamp: Optional[int] = None,
        user_id: Optional[str] = None,
        sample: bool = True,
    ) -> List[GeneratedStep]:
        """Generate contexts and outcomes from a learned traffic prior.

        A prior is mandatory when contexts are not supplied.  This prevents a
        silent fallback to hand-written feature rules and makes the dependency
        on real training data explicit.
        """
        if prior is None:
            raise ValueError("AdCSPPPathGenerator.generate requires a fitted AdCSPPFlowPrior")
        if state is None:
            profile = prior.sample_profile(user_id=user_id)
            # The prior carries the observed user's long-term values and
            # core sequence state, so generation starts from a real history
            # rather than an empty synthetic profile.
            state = UserStateBuilder.build(
                profile,
                aggregate_features=profile.features,
                latent_dim=self.model.latent_dim,
            )
        else:
            profile = UserProfile(user_id=state.user_id, features=dict(state.static_features.get("features", state.static_features)))
        timestamp = int(start_timestamp if start_timestamp is not None else state.last_update_timestamp)
        if timestamp <= 0:
            raise ValueError("generate requires a positive start_timestamp or state.last_update_timestamp")
        if num_steps is None:
            num_steps = prior.sample_path_length()
        contexts = []
        previous = None
        for _ in range(int(num_steps)):
            timestamp += prior.sample_gap()
            request = prior.sample_context(profile, timestamp, previous_context=previous)
            contexts.append(request)
            previous = request
        return self.rollout(state, contexts, sample=sample)


def _truncate_histories(state: UserState, max_len: int) -> None:
    for name, values in list(state.sequence_features.items()):
        if isinstance(values, list):
            state.sequence_features[name] = values[-max_len:]

