"""Conditional V1 simulator with a priority queue and concurrent chains."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from inputs.adcsp.schema import Event, EventType, ExposureContext, SimulationConfig, UserState
from .ledger import EventLedger
from .model import AdCSPPModel


@dataclass
class SimulationTrajectory:
    user_id: str
    events: List[Event]
    final_state: UserState

    def to_records(self) -> list[dict]:
        return [event.to_dict() for event in self.events]


class ConditionalSimulator:
    def __init__(self, model: AdCSPPModel, config: SimulationConfig | None = None):
        self.model = model
        self.config = config or SimulationConfig()

    def simulate(self, user_state: UserState, exposure_plan: Sequence[ExposureContext]) -> SimulationTrajectory:
        rng = np.random.default_rng(self.config.seed + abs(hash(user_state.user_id)) % 10_000)
        ledger = EventLedger(user_state)
        queue = []
        counter = 0

        def schedule(timestamp: int, priority: int, kind: str, payload):
            nonlocal counter
            counter += 1
            heapq.heappush(queue, (int(timestamp), int(priority), counter, kind, payload))

        start = self.config.start_timestamp
        end = start + self.config.horizon_seconds
        for exposure in sorted(exposure_plan, key=lambda item: (item.event_timestamp, item.impression_id)):
            if start <= exposure.event_timestamp <= end:
                schedule(exposure.event_timestamp, 0, "exposure", exposure)

        while queue:
            timestamp, _, _, kind, payload = heapq.heappop(queue)
            if timestamp > end:
                break
            if kind == "exposure":
                exposure: ExposureContext = payload
                exposure_event = self._event_from_exposure(user_state.user_id, exposure)
                ledger.append(exposure_event)
                click_result = self.model.sample_click(user_state, exposure, self.config.click_window_seconds, rng)
                if click_result.clicked:
                    schedule(timestamp + int(click_result.delay_seconds), 1, "click", (exposure, click_result))
                else:
                    schedule(timestamp + self.config.click_window_seconds, 3, "impression_expiry", exposure)
            elif kind == "click":
                exposure, click_result = payload
                click_event = self._click_event(user_state.user_id, exposure, timestamp)
                ledger.append(click_event)
                conversion_result = self.model.sample_conversion(user_state, exposure, self.config.conversion_window_seconds, rng)
                if conversion_result.converted:
                    schedule(timestamp + int(conversion_result.delay_seconds), 2, "conversion", (exposure, click_event, conversion_result))
                else:
                    schedule(timestamp + self.config.conversion_window_seconds, 3, "click_expiry", click_event)
            elif kind == "conversion":
                exposure, click_event, conversion_result = payload
                ledger.append(self._conversion_event(user_state.user_id, exposure, click_event, timestamp, conversion_result))
            elif kind == "impression_expiry":
                exposure: ExposureContext = payload
                if exposure.impression_id in user_state.active_impressions:
                    ledger.append(self._expiry_event(user_state.user_id, exposure, timestamp, impression_id=exposure.impression_id))
            elif kind == "click_expiry":
                click_event: Event = payload
                if click_event.click_id in user_state.active_clicks:
                    exposure = self._context_from_state(user_state, click_event.impression_id)
                    ledger.append(self._expiry_event(user_state.user_id, exposure, timestamp, click_id=click_event.click_id))

        return SimulationTrajectory(user_state.user_id, ledger.events, user_state)

    @staticmethod
    def _event_from_exposure(user_id: str, exposure: ExposureContext) -> Event:
        return Event(
            user_id=user_id,
            event_id=f"EXP_{exposure.impression_id}",
            event_timestamp=exposure.event_timestamp,
            event_type=EventType.EXPOSURE,
            impression_id=exposure.impression_id,
            slot_id=exposure.slot_id,
            ad_id=exposure.ad_id,
            creative_id=exposure.creative_id,
            campaign_id=exposure.campaign_id,
            advertiser_id=exposure.advertiser_id,
            ad_category=exposure.ad_category,
            objective_type=exposure.objective_type,
        )

    @staticmethod
    def _click_event(user_id: str, exposure: ExposureContext, timestamp: int) -> Event:
        return Event(
            user_id=user_id,
            event_id=f"CLK_{exposure.impression_id}",
            event_timestamp=timestamp,
            event_type=EventType.CLICK,
            impression_id=exposure.impression_id,
            click_id=f"CLK_{exposure.impression_id}",
            parent_event_id=f"EXP_{exposure.impression_id}",
            slot_id=exposure.slot_id,
            ad_id=exposure.ad_id,
            creative_id=exposure.creative_id,
            campaign_id=exposure.campaign_id,
            advertiser_id=exposure.advertiser_id,
            ad_category=exposure.ad_category,
            objective_type=exposure.objective_type,
        )

    @staticmethod
    def _conversion_event(user_id: str, exposure: ExposureContext, click_event: Event, timestamp: int, result) -> Event:
        return Event(
            user_id=user_id,
            event_id=f"CNV_{exposure.impression_id}",
            event_timestamp=timestamp,
            event_type=EventType.CONVERSION,
            impression_id=exposure.impression_id,
            click_id=click_event.click_id,
            parent_event_id=click_event.event_id,
            slot_id=exposure.slot_id,
            ad_id=exposure.ad_id,
            creative_id=exposure.creative_id,
            campaign_id=exposure.campaign_id,
            advertiser_id=exposure.advertiser_id,
            ad_category=exposure.ad_category,
            objective_type=exposure.objective_type,
            conversion_type=result.conversion_type,
            conversion_target=result.conversion_target,
            conversion_value=result.conversion_value,
        )

    @staticmethod
    def _expiry_event(user_id: str, exposure: ExposureContext, timestamp: int, impression_id=None, click_id=None) -> Event:
        return Event(
            user_id=user_id,
            event_id=f"EXPIRY_{impression_id or click_id}",
            event_timestamp=timestamp,
            event_type=EventType.WINDOW_EXPIRED,
            impression_id=impression_id,
            click_id=click_id,
            parent_event_id=f"EXP_{exposure.impression_id}",
            slot_id=exposure.slot_id,
            ad_id=exposure.ad_id,
            creative_id=exposure.creative_id,
        )

    @staticmethod
    def _context_from_state(state: UserState, impression_id: str) -> ExposureContext:
        impression = state.active_impressions[impression_id]
        return ExposureContext(
            impression_id=impression.impression_id,
            event_timestamp=impression.exposure_timestamp,
            slot_id=impression.slot_id,
            slot_type="feed",
            slot_position="middle",
            ad_id=impression.ad_id,
            creative_id=impression.creative_id,
            campaign_id="campaign_0",
            advertiser_id="advertiser_0",
            ad_category="commerce",
            objective_type="purchase",
            creative_type="image",
            creative_size="square",
            bid_price=1.0,
            ecpm=1.0,
            ad_quality_score=0.8,
            landing_page_type="product",
            promotion_target="purchase",
            session_id="session_0",
            traffic_source="organic",
            app_category="shopping",
            page_context="home",
            network_type="wifi",
        )
