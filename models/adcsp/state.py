"""User state initialization and the single event update entry point."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np

from inputs.adcsp.schema import Event, EventType, UserProfile, UserState
from .constraints import EventValidator
from .fsm import ImpressionFSM


class UserStateBuilder:
    @staticmethod
    def build(
        profile: UserProfile,
        history_events: Optional[Iterable[Event]] = None,
        aggregate_features: Optional[Mapping[str, object]] = None,
        latent_dim: int = 16,
        static_embedding: Optional[np.ndarray] = None,
        feature_config: Optional[Mapping[str, object]] = None,
    ) -> UserState:
        state = UserState(
            user_id=profile.user_id,
            static_features=profile.to_dict(),
            static_embedding=np.zeros(8, dtype=np.float32)
            if static_embedding is None
            else np.asarray(static_embedding, dtype=np.float32).copy(),
            latent_state=np.zeros(latent_dim, dtype=np.float32),
            last_update_timestamp=0,
            dynamic_features=dict(aggregate_features or {}),
        )
        updater = StateUpdater(feature_config=feature_config)
        for event in sorted(history_events or [], key=lambda item: (item.event_timestamp, item.event_seq_no)):
            updater.apply_event(state, event)
        return state


class StateUpdater:
    """Mutate UserState only through validated events."""

    DEFAULT_SEQUENCE_FEATURES = {
        "hist_exp_slots": {"event_type": "EXPOSURE", "field": "slot_id", "max_len": 20},
        "hist_exp_ads": {"event_type": "EXPOSURE", "field": "ad_id", "max_len": 20},
        "hist_clk_slots": {"event_type": "CLICK", "field": "slot_id", "max_len": 20},
        "hist_clk_ads": {"event_type": "CLICK", "field": "ad_id", "max_len": 20},
        "hist_clk_categories": {"event_type": "CLICK", "field": "ad_category", "max_len": 20},
        "hist_conv_slots": {"event_type": "CONVERSION", "field": "slot_id", "max_len": 20},
        "hist_conv_targets": {"event_type": "CONVERSION", "field": "conversion_target", "max_len": 20},
    }

    def __init__(self, validator: Optional[EventValidator] = None, fsm: Optional[ImpressionFSM] = None, feature_config: Optional[Mapping[str, object]] = None):
        self.validator = validator or EventValidator()
        self.fsm = fsm or ImpressionFSM()
        config = dict(feature_config or {})
        self.dynamic_windows = config.get("dynamic_windows", {"1h": 3_600, "24h": 86_400, "7d": 7 * 86_400})
        self.sequence_config = dict(self.DEFAULT_SEQUENCE_FEATURES)
        self.sequence_config.update(config.get("sequence_features", {}))

    def apply_event(self, state: UserState, event: Event) -> UserState:
        self.validator.validate(event, state)
        previous_timestamp = state.last_update_timestamp
        event.event_time_delta = max(0.0, event.event_timestamp - previous_timestamp) if previous_timestamp else 0.0
        self.fsm.transition(state, event)
        state.event_history.append(event)
        state.state_version += 1
        event.state_version = state.state_version
        state.last_update_timestamp = max(state.last_update_timestamp, event.event_timestamp)
        decay = np.exp(-event.event_time_delta / 3_600.0) if event.event_time_delta else 1.0
        state.latent_state = np.asarray(state.latent_state, dtype=np.float32) * decay
        index = {
            EventType.EXPOSURE: 0,
            EventType.CLICK: 1,
            EventType.CONVERSION: 2,
            EventType.WINDOW_EXPIRED: 3,
        }[event.event_type]
        if index < len(state.latent_state):
            state.latent_state[index] += 1.0
        if event.event_type is EventType.EXPOSURE:
            state.user_exposure_cnt += 1
        elif event.event_type is EventType.CLICK:
            state.user_click_cnt += 1
        elif event.event_type is EventType.CONVERSION:
            key = event.conversion_type or "unknown"
            state.user_conversion_cnt[key] = state.user_conversion_cnt.get(key, 0) + 1
        if event.slot_id:
            slot_stat = state.user_slot_stats.setdefault(event.slot_id, {"exposure": 0, "click": 0, "conversion": 0})
            if event.event_type is EventType.EXPOSURE:
                slot_stat["exposure"] += 1
            elif event.event_type is EventType.CLICK:
                slot_stat["click"] += 1
            elif event.event_type is EventType.CONVERSION:
                slot_stat["conversion"] += 1
        state.dynamic_features = self.recompute_dynamic_features(state, event.event_timestamp)
        state.sequence_features = {
            name: list(values) for name, values in state.dynamic_features.items() if name.startswith("hist_")
        }
        return state

    def recompute_dynamic_features(self, state: UserState, timestamp: int) -> dict:
        events = state.event_history
        windows = {str(name): int(seconds) for name, seconds in self.dynamic_windows.items()}
        features = {}
        for name, seconds in windows.items():
            recent = [item for item in events if timestamp - seconds <= item.event_timestamp <= timestamp]
            features[f"user_exposure_cnt_{name}"] = sum(item.event_type is EventType.EXPOSURE for item in recent)
            features[f"user_click_cnt_{name}"] = sum(item.event_type is EventType.CLICK for item in recent)
            features[f"user_conversion_cnt_{name}"] = sum(item.event_type is EventType.CONVERSION for item in recent)
        real_events = [item for item in events if item.event_type is not EventType.WINDOW_EXPIRED]
        features["user_active_days_30d"] = len({item.event_timestamp // 86_400 for item in real_events if timestamp - item.event_timestamp <= 30 * 86_400})
        for event_type, key in ((EventType.EXPOSURE, "exposure"), (EventType.CLICK, "click"), (EventType.CONVERSION, "conversion")):
            matching = [item.event_timestamp for item in events if item.event_type is event_type]
            features[f"time_since_last_{key}"] = float(timestamp - max(matching)) if matching else None
        exposures = [item for item in events if item.event_type is EventType.EXPOSURE]
        repeated = sum(1 for item in exposures if sum(other.ad_id == item.ad_id for other in exposures) > 1)
        features["repeated_exposure_ratio"] = repeated / len(exposures) if exposures else 0.0
        features["last_exposure_slot"] = exposures[-1].slot_id if exposures else None
        features["last_click_slot"] = next((item.slot_id for item in reversed(events) if item.event_type is EventType.CLICK), None)
        for name, spec in self.sequence_config.items():
            event_type = EventType(str(spec.get("event_type", "CLICK")))
            field = str(spec.get("field", "slot_id"))
            max_len = max(1, int(spec.get("max_len", 20)))
            values = [getattr(item, field, None) for item in events if item.event_type is event_type]
            features[name] = [value for value in values if value not in (None, "")][-max_len:]
        return features
