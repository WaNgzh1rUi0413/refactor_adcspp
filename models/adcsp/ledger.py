"""In-memory event ledger used by simulation; only final records are persisted."""

from __future__ import annotations

from typing import Iterable, List

from inputs.adcsp.schema import Event, UserState
from .state import StateUpdater


class EventLedger:
    def __init__(self, state: UserState, updater: StateUpdater | None = None):
        self.state = state
        self.updater = updater or StateUpdater()
        self.events: List[Event] = []

    def append(self, event: Event) -> Event:
        event.event_seq_no = sum(item.event_timestamp == event.event_timestamp for item in self.events)
        self.updater.apply_event(self.state, event)
        self.events.append(event)
        return event

    def extend(self, events: Iterable[Event]) -> None:
        for event in events:
            self.append(event)
