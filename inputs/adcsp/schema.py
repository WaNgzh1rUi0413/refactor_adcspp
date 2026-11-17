"""Typed data contracts for the AdCSPP V1 conditional simulator."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


class EventType(str, Enum):
    EXPOSURE = "EXPOSURE"
    CLICK = "CLICK"
    CONVERSION = "CONVERSION"
    WINDOW_EXPIRED = "WINDOW_EXPIRED"


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    gender: str
    age_group: str
    region: str
    device_type: str
    os_type: str
    user_segment: str
    user_long_term_interest: Tuple[str, ...]
    user_registration_age: float
    user_value_segment: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExposureContext:
    impression_id: str
    event_timestamp: int
    slot_id: str
    slot_type: str
    slot_position: str
    ad_id: str
    creative_id: str
    campaign_id: str
    advertiser_id: str
    ad_category: str
    objective_type: str
    creative_type: str
    creative_size: str
    bid_price: float
    ecpm: float
    ad_quality_score: float
    landing_page_type: str
    promotion_target: str
    session_id: str
    traffic_source: str
    app_category: str
    page_context: str
    network_type: str
    is_holiday: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Event:
    user_id: str
    event_id: str
    event_timestamp: int
    event_type: EventType
    event_seq_no: int = 0
    event_time_delta: float = 0.0
    impression_id: Optional[str] = None
    click_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    slot_id: Optional[str] = None
    ad_id: Optional[str] = None
    creative_id: Optional[str] = None
    campaign_id: Optional[str] = None
    advertiser_id: Optional[str] = None
    ad_category: Optional[str] = None
    objective_type: Optional[str] = None
    conversion_type: Optional[str] = None
    conversion_target: Optional[str] = None
    conversion_value: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["event_type"] = self.event_type.value
        return result


@dataclass
class ImpressionState:
    impression_id: str
    exposure_timestamp: int
    click_deadline: int
    slot_id: str
    ad_id: str
    creative_id: str
    click_status: str = "IMPACTED"
    click_id: Optional[str] = None
    exposure_event_id: Optional[str] = None


@dataclass
class ClickState:
    click_id: str
    impression_id: str
    click_timestamp: int
    conversion_deadline: int
    conversion_status: str = "PENDING"
    click_event_id: Optional[str] = None


@dataclass
class UserState:
    user_id: str
    static_features: Dict[str, Any]
    static_embedding: np.ndarray
    latent_state: np.ndarray
    last_update_timestamp: int
    dynamic_features: Dict[str, Any] = field(default_factory=dict)
    event_history: List[Event] = field(default_factory=list)
    user_exposure_cnt: int = 0
    user_click_cnt: int = 0
    user_conversion_cnt: Dict[str, int] = field(default_factory=dict)
    user_slot_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    active_impressions: Dict[str, ImpressionState] = field(default_factory=dict)
    active_clicks: Dict[str, ClickState] = field(default_factory=dict)
    state_version: int = 0

    def clone(self) -> "UserState":
        return copy.deepcopy(self)


@dataclass(frozen=True)
class SimulationConfig:
    horizon_seconds: int = 86_400
    click_window_seconds: int = 30
    conversion_window_seconds: int = 3_600
    latent_dim: int = 16
    seed: int = 42
    start_timestamp: int = 1_700_000_000

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SimulationConfig":
        fields = {key: values[key] for key in cls.__dataclass_fields__ if key in values}
        return cls(**fields)


@dataclass(frozen=True)
class ClickResult:
    clicked: bool
    probability: float
    delay_seconds: Optional[int] = None


@dataclass(frozen=True)
class ConversionResult:
    converted: bool
    probability: float
    delay_seconds: Optional[int] = None
    conversion_type: str = "purchase"
    conversion_target: str = "purchase"
    conversion_value: float = 0.0
