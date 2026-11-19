"""Exposure-level finite-state machine and event ledger."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from inputs.adcsp.schema import ExposureContext


@dataclass
class ImpressionState:
    impression_id: str
    exposure_timestamp: int
    slot_id: str
    click_status: str = "IMPRESSED"
    click_id: Optional[str] = None
    click_deadline: Optional[int] = None
    conversion_deadline: Optional[int] = None


@dataclass
class ClickState:
    click_id: str
    impression_id: str
    click_timestamp: int
    conversion_status: str = "PENDING"
    conversion_deadline: Optional[int] = None


@dataclass
class EventRecord:
    event_type: str
    event_id: str
    timestamp: int
    impression_id: str
    click_id: Optional[str] = None
    slot_id: Optional[str] = None


class EventLedger:
    def __init__(self):
        self.events: List[EventRecord] = []

    def append(self, event: EventRecord) -> None:
        self.events.append(event)

    def to_dicts(self) -> List[dict]:
        return [asdict(event) for event in self.events]


class FiniteStateMachine:
    """Enforce exposure -> click -> conversion ordering and deadlines."""

    def __init__(self, click_window_seconds: int = 30, conversion_window_seconds: int = 3600):
        self.click_window_seconds = int(click_window_seconds)
        self.conversion_window_seconds = int(conversion_window_seconds)
        self.active_impressions: Dict[str, ImpressionState] = {}
        self.active_clicks: Dict[str, ClickState] = {}
        self.ledger = EventLedger()

    def register_exposure(self, context: ExposureContext) -> ImpressionState:
        impression_id = str(context.impression_id)
        impression = ImpressionState(
            impression_id=impression_id,
            exposure_timestamp=int(context.event_timestamp),
            slot_id=str(context.slot_id),
            click_deadline=int(context.event_timestamp) + self.click_window_seconds,
        )
        self.active_impressions[impression_id] = impression
        self.ledger.append(EventRecord("EXPOSURE", impression_id, impression.exposure_timestamp, impression_id, slot_id=impression.slot_id))
        return impression

    def record_click(self, context: ExposureContext) -> ClickState:
        impression = self.active_impressions.get(str(context.impression_id))
        if impression is None:
            raise ValueError("CLICK must reference an existing impression")
        if impression.click_status != "IMPRESSED":
            raise ValueError(f"Impression {impression.impression_id} cannot transition from {impression.click_status} to CLICKED")
        if int(context.event_timestamp) < impression.exposure_timestamp:
            raise ValueError("CLICK timestamp cannot precede EXPOSURE")
        if impression.click_deadline is not None and int(context.event_timestamp) > impression.click_deadline:
            raise ValueError("CLICK is outside the click window")
        click_id = f"click:{impression.impression_id}:{int(context.event_timestamp)}"
        impression.click_status = "CLICKED"
        impression.click_id = click_id
        impression.conversion_deadline = int(context.event_timestamp) + self.conversion_window_seconds
        click = ClickState(click_id, impression.impression_id, int(context.event_timestamp), conversion_deadline=impression.conversion_deadline)
        self.active_clicks[click_id] = click
        self.ledger.append(EventRecord("CLICK", click_id, click.click_timestamp, impression.impression_id, click_id=click_id, slot_id=impression.slot_id))
        return click

    def record_no_click(self, context: ExposureContext) -> None:
        impression = self.active_impressions.get(str(context.impression_id))
        if impression is None:
            raise ValueError("NO_CLICK must reference an existing impression")
        if impression.click_status == "CLICKED":
            raise ValueError("CLICKED impression cannot transition to NO_CLICK")
        impression.click_status = "EXPIRED"
        self.ledger.append(EventRecord("NO_CLICK_EXPIRED", impression.impression_id, int(context.event_timestamp), impression.impression_id, slot_id=impression.slot_id))

    def record_conversion(self, context: ExposureContext) -> EventRecord:
        impression = self.active_impressions.get(str(context.impression_id))
        if impression is None or impression.click_id is None:
            raise ValueError("CONVERSION must reference a clicked impression")
        if int(context.event_timestamp) < impression.exposure_timestamp:
            raise ValueError("CONVERSION timestamp cannot precede EXPOSURE")
        click = self.active_clicks[impression.click_id]
        if int(context.event_timestamp) < click.click_timestamp:
            raise ValueError("CONVERSION timestamp cannot precede CLICK")
        if impression.conversion_deadline is not None and int(context.event_timestamp) > impression.conversion_deadline:
            raise ValueError("CONVERSION is outside the conversion window")
        click.conversion_status = "CONVERTED"
        event = EventRecord("CONVERSION", f"conversion:{impression.click_id}:{int(context.event_timestamp)}", int(context.event_timestamp), impression.impression_id, click_id=impression.click_id, slot_id=impression.slot_id)
        self.ledger.append(event)
        return event

    def record_conversion_expired(self, context: ExposureContext) -> EventRecord:
        """Close a clicked impression when its conversion window expires."""
        impression = self.active_impressions.get(str(context.impression_id))
        if impression is None or impression.click_id is None:
            raise ValueError("CONVERSION_EXPIRED must reference a clicked impression")
        click = self.active_clicks[impression.click_id]
        if int(context.event_timestamp) < click.click_timestamp:
            raise ValueError("CONVERSION_EXPIRED timestamp cannot precede CLICK")
        click.conversion_status = "EXPIRED"
        event = EventRecord(
            "CONVERSION_EXPIRED",
            f"conversion_expired:{impression.click_id}:{int(context.event_timestamp)}",
            int(context.event_timestamp),
            impression.impression_id,
            click_id=impression.click_id,
            slot_id=impression.slot_id,
        )
        self.ledger.append(event)
        return event

