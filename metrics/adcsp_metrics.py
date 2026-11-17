"""Evaluation metrics that do not require real data."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

from inputs.adcsp.schema import Event, EventType
from models.adcsp.constraints import validate_ledger


def event_counts(events: Iterable[Event]) -> dict[str, int]:
    return dict(Counter(event.event_type.value for event in events))


def ledger_metrics(events: list[Event]) -> dict[str, float | int]:
    errors = validate_ledger(events)
    users = len({event.user_id for event in events})
    exposures = sum(event.event_type is EventType.EXPOSURE for event in events)
    clicks = sum(event.event_type is EventType.CLICK for event in events)
    conversions = sum(event.event_type is EventType.CONVERSION for event in events)
    return {
        "users": users,
        "events": len(events),
        "exposures": exposures,
        "clicks": clicks,
        "conversions": conversions,
        "ctr": clicks / exposures if exposures else 0.0,
        "cvr_after_click": conversions / clicks if clicks else 0.0,
        "constraint_violations": len(errors),
    }
