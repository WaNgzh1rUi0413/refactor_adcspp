"""Business constraints for AdCSPP event chains."""

from __future__ import annotations

from typing import Iterable

from inputs.adcsp.schema import Event, EventType, UserState


class EventValidationError(ValueError):
    """Raised when an event violates an AdCSPP hard constraint."""


class EventValidator:
    """Validate event ordering and parent relationships before state mutation."""

    def validate(self, event: Event, state: UserState) -> None:
        if event.user_id != state.user_id:
            raise EventValidationError("event user_id does not match UserState")
        if any(item.event_id == event.event_id for item in state.event_history):
            raise EventValidationError(f"duplicate event_id: {event.event_id}")
        if state.event_history and event.event_timestamp < state.event_history[-1].event_timestamp:
            raise EventValidationError("event timestamp is earlier than the latest event")

        if event.event_type is EventType.EXPOSURE:
            if event.impression_id is None:
                raise EventValidationError("exposure requires impression_id")
            return

        if event.event_type is EventType.CLICK:
            if not event.impression_id or event.impression_id not in state.active_impressions:
                raise EventValidationError("click requires an active impression")
            impression = state.active_impressions[event.impression_id]
            if event.event_timestamp < impression.exposure_timestamp:
                raise EventValidationError("click cannot precede exposure")
            if not event.click_id:
                raise EventValidationError("click requires click_id")
            return

        if event.event_type is EventType.CONVERSION:
            if not event.click_id or event.click_id not in state.active_clicks:
                raise EventValidationError("conversion requires an active click")
            click = state.active_clicks[event.click_id]
            if event.event_timestamp < click.click_timestamp:
                raise EventValidationError("conversion cannot precede click")
            return

        if event.event_type is EventType.WINDOW_EXPIRED:
            if event.impression_id in state.active_impressions:
                return
            if event.click_id in state.active_clicks:
                return
            raise EventValidationError("window expiry requires an active impression or click")

        raise EventValidationError(f"unsupported event type: {event.event_type}")


def validate_ledger(events: Iterable[Event]) -> list[str]:
    """Return all violations in an already generated ledger."""
    errors: list[str] = []
    by_id = {event.event_id: event for event in events}
    impressions = {event.impression_id: event for event in events if event.event_type is EventType.EXPOSURE}
    clicks = {event.click_id: event for event in events if event.event_type is EventType.CLICK}
    previous = {}
    for event in events:
        if event.event_timestamp < previous.get(event.user_id, event.event_timestamp):
            errors.append(f"out_of_order:{event.event_id}")
        previous[event.user_id] = event.event_timestamp
        if event.event_type is EventType.CLICK:
            exposure = impressions.get(event.impression_id)
            if exposure is None:
                errors.append(f"orphan_click:{event.event_id}")
            elif event.event_timestamp < exposure.event_timestamp:
                errors.append(f"click_before_exposure:{event.event_id}")
        if event.event_type is EventType.CONVERSION:
            click = clicks.get(event.click_id)
            if click is None:
                errors.append(f"orphan_conversion:{event.event_id}")
            elif event.event_timestamp < click.event_timestamp:
                errors.append(f"conversion_before_click:{event.event_id}")
        if event.event_id in by_id and sum(item.event_id == event.event_id for item in events) > 1:
            errors.append(f"duplicate_event:{event.event_id}")
    return errors
