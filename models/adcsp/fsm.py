"""Exposure and click lifecycle state transitions."""

from inputs.adcsp.schema import Event, EventType, ImpressionState, ClickState, UserState


class ImpressionFSM:
    """Apply the legal state transitions for an impression chain."""

    def __init__(self, click_window_seconds: int = 30, conversion_window_seconds: int = 3_600):
        self.click_window_seconds = click_window_seconds
        self.conversion_window_seconds = conversion_window_seconds

    def transition(self, state: UserState, event: Event) -> None:
        if event.event_type is EventType.EXPOSURE:
            state.active_impressions[event.impression_id] = ImpressionState(
                impression_id=event.impression_id,
                exposure_timestamp=event.event_timestamp,
                click_deadline=event.event_timestamp + self.click_window_seconds,
                slot_id=event.slot_id or "",
                ad_id=event.ad_id or "",
                creative_id=event.creative_id or "",
                exposure_event_id=event.event_id,
            )
        elif event.event_type is EventType.CLICK:
            impression = state.active_impressions[event.impression_id]
            impression.click_status = "CLICKED"
            impression.click_id = event.click_id
            state.active_clicks[event.click_id] = ClickState(
                click_id=event.click_id,
                impression_id=event.impression_id,
                click_timestamp=event.event_timestamp,
                conversion_deadline=event.event_timestamp + self.conversion_window_seconds,
                click_event_id=event.event_id,
            )
        elif event.event_type is EventType.CONVERSION:
            click = state.active_clicks[event.click_id]
            click.conversion_status = "CONVERTED"
            state.active_impressions[click.impression_id].click_status = "CONVERTED"
        elif event.event_type is EventType.WINDOW_EXPIRED:
            if event.impression_id in state.active_impressions:
                impression = state.active_impressions[event.impression_id]
                impression.click_status = "EXPIRED"
            if event.click_id in state.active_clicks:
                state.active_clicks[event.click_id].conversion_status = "EXPIRED"
