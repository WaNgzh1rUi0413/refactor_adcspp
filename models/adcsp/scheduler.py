"""Timestamp ordered event queue for concurrent user trajectories."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(order=True)
class ScheduledEvent:
    timestamp: int
    sequence: int
    event_type: str = field(compare=False)
    impression_id: str | None = field(default=None, compare=False)
    payload: Any = field(default=None, compare=False)


class JointEventScheduler:
    """Stable queue for exposure, click, conversion, and expiry events."""

    def __init__(self):
        self._queue: list[ScheduledEvent] = []
        self._sequence = 0

    def schedule(self, event_type: str, timestamp: int, payload=None, impression_id: str | None = None) -> ScheduledEvent:
        event = ScheduledEvent(int(timestamp), self._sequence, str(event_type), impression_id, payload)
        self._sequence += 1
        heapq.heappush(self._queue, event)
        return event

    def has_pending(self) -> bool:
        return bool(self._queue)

    def peek(self) -> ScheduledEvent:
        if not self._queue:
            raise IndexError("event scheduler is empty")
        return self._queue[0]

    def pop(self) -> ScheduledEvent:
        if not self._queue:
            raise IndexError("event scheduler is empty")
        return heapq.heappop(self._queue)

    def run_until(self, end_timestamp: int, handler: Callable[[ScheduledEvent], Any]) -> list[Any]:
        results = []
        while self._queue and self._queue[0].timestamp <= int(end_timestamp):
            results.append(handler(self.pop()))
        return results

