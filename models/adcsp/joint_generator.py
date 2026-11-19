"""Autonomous exposure and candidate generation over the shared state model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from inputs.adcsp.schema import ExposureContext, UserState
from .event_generator import HierarchicalEventGenerator
from .fsm import FiniteStateMachine
from .heads import CandidateMasker
from .opportunity import sample_exposure_wait
from .scheduler import JointEventScheduler


def _candidate_context(candidate, timestamp: int, impression_id: str) -> ExposureContext:
    if isinstance(candidate, ExposureContext):
        return replace(candidate, impression_id=impression_id, event_timestamp=int(timestamp))
    if isinstance(candidate, Mapping):
        values = dict(candidate)
        features = dict(values.get("features", {}) or {})
        features.update({key: value for key, value in values.items() if key not in {"features", "impression_id", "event_timestamp", "slot_id"}})
        return ExposureContext(
            impression_id=impression_id,
            event_timestamp=int(timestamp),
            slot_id=str(values.get("slot_id", features.get("slot_id", "unknown"))),
            features=features,
        )
    raise TypeError("candidate must be ExposureContext or mapping")


@dataclass(frozen=True)
class JointGeneratedEvent:
    user_id: str
    event_type: str
    timestamp: int
    impression_id: str
    click_id: str | None = None
    slot_id: str | None = None
    ad_id: str | None = None
    click_probability: float | None = None
    conversion_probability: float | None = None
    selected_candidate_index: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class JointTrafficGenerator:
    """Generate a complete path with autonomous opportunities and candidates."""

    def __init__(self, model, click_window_seconds: int = 30, conversion_window_seconds: int = 3600, seed: int = 42, max_sequence_length: int = 20):
        self.model = model
        self.click_window_seconds = int(click_window_seconds)
        self.conversion_window_seconds = int(conversion_window_seconds)
        self.max_sequence_length = int(max_sequence_length)
        self.rng = np.random.default_rng(seed)
        self.event_generator = HierarchicalEventGenerator(model, self.click_window_seconds, self.conversion_window_seconds, seed=seed)
        self.masker = CandidateMasker()

    def generate(self, state: UserState, candidates: Sequence, start_timestamp: int, horizon_seconds: int, *, policy_embedding=None, valid_mask=None, mask_constraints=None, sample: bool = True, max_events: int = 10000) -> list[dict]:
        candidates = list(candidates)
        if not candidates:
            raise ValueError("joint generation requires a non-empty candidate pool")
        if int(horizon_seconds) <= 0:
            raise ValueError("horizon_seconds must be positive")
        current = state.clone()
        horizon = int(start_timestamp) + int(horizon_seconds)
        scheduler = JointEventScheduler()
        scheduler.schedule("EXPOSURE_OPPORTUNITY", int(start_timestamp))
        fsm = FiniteStateMachine(self.click_window_seconds, self.conversion_window_seconds)
        contexts: dict[str, ExposureContext] = {}
        generated: list[dict] = []
        opportunity_index = 0
        processed = 0
        while scheduler.has_pending() and processed < int(max_events):
            event = scheduler.pop()
            processed += 1
            if event.timestamp > horizon:
                break
            if event.event_type == "EXPOSURE_OPPORTUNITY":
                mask = valid_mask
                if mask is None:
                    mask = self.masker.mask(candidates, **(mask_constraints or {})) if mask_constraints else np.ones((len(candidates),), dtype=bool)
                selected_index, _, _ = self.model.select_candidate(current, candidates, policy_embedding=policy_embedding, valid_mask=mask)
                impression_id = f"joint:{current.user_id}:{opportunity_index}"
                opportunity_index += 1
                context = _candidate_context(candidates[selected_index], event.timestamp, impression_id)
                contexts[impression_id] = context
                fsm.register_exposure(context)
                current = self.model.transition_event(current, context, "EXPOSURE", event.timestamp)
                outcome = self.event_generator.generate(current, context, sample=sample, rng=self.rng)
                ad_id = str(context.features.get("ad_id", context.features.get("adid", "unknown")))
                generated.append(JointGeneratedEvent(current.user_id, "EXPOSURE", event.timestamp, impression_id, slot_id=context.slot_id, ad_id=ad_id, click_probability=outcome.click_probability, conversion_probability=outcome.conversion_probability, selected_candidate_index=selected_index).to_dict())
                if outcome.is_click:
                    scheduler.schedule("CLICK", int(outcome.click_timestamp), impression_id=impression_id)
                    scheduler.schedule("CONVERSION" if outcome.is_conversion else "CONVERSION_EXPIRED", int(outcome.conversion_timestamp) if outcome.is_conversion else int(outcome.click_timestamp) + self.conversion_window_seconds, impression_id=impression_id)
                else:
                    scheduler.schedule("NO_CLICK_EXPIRED", event.timestamp + self.click_window_seconds, impression_id=impression_id)
                rate = self.model.exposure_opportunity_rate(current, context, policy_embedding=policy_embedding)
                next_timestamp = event.timestamp + sample_exposure_wait(rate, self.rng, max_wait_seconds=max(1, horizon - event.timestamp))
                if next_timestamp <= horizon:
                    scheduler.schedule("EXPOSURE_OPPORTUNITY", next_timestamp)
                continue
            context = contexts.get(str(event.impression_id))
            if context is None:
                raise ValueError(f"event {event.event_type} references unknown impression {event.impression_id}")
            event_context = replace(context, event_timestamp=event.timestamp)
            ad_id = str(context.features.get("ad_id", context.features.get("adid", "unknown")))
            if event.event_type == "CLICK":
                click = fsm.record_click(event_context)
                current = self.model.transition_event(current, event_context, "CLICK", event.timestamp)
                generated.append(JointGeneratedEvent(current.user_id, "CLICK", event.timestamp, context.impression_id, click_id=click.click_id, slot_id=context.slot_id, ad_id=ad_id).to_dict())
            elif event.event_type == "CONVERSION":
                conversion = fsm.record_conversion(event_context)
                current = self.model.transition_event(current, event_context, "CONVERSION", event.timestamp)
                generated.append(JointGeneratedEvent(current.user_id, "CONVERSION", event.timestamp, context.impression_id, click_id=conversion.click_id, slot_id=context.slot_id, ad_id=ad_id).to_dict())
            elif event.event_type == "NO_CLICK_EXPIRED":
                fsm.record_no_click(event_context)
                current = self.model.transition_event(current, event_context, "NO_CLICK_EXPIRED", event.timestamp)
                generated.append(JointGeneratedEvent(current.user_id, "NO_CLICK_EXPIRED", event.timestamp, context.impression_id, slot_id=context.slot_id, ad_id=ad_id).to_dict())
            elif event.event_type == "CONVERSION_EXPIRED":
                expired = fsm.record_conversion_expired(event_context)
                current = self.model.transition_event(current, event_context, "CONVERSION_EXPIRED", event.timestamp)
                generated.append(JointGeneratedEvent(current.user_id, "CONVERSION_EXPIRED", event.timestamp, context.impression_id, click_id=expired.click_id, slot_id=context.slot_id, ad_id=ad_id).to_dict())
        return generated
