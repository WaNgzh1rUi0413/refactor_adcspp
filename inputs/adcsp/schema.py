"""Model contracts for online AdCSPP inference and training."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

import numpy as np


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
class UserState:
    user_id: str
    static_features: Dict[str, Any]
    static_embedding: np.ndarray
    latent_state: np.ndarray
    last_update_timestamp: int
    dynamic_features: Dict[str, Any] = field(default_factory=dict)
    sequence_features: Dict[str, list] = field(default_factory=dict)
    initial_sequence_features: Dict[str, list] = field(default_factory=dict)

    def clone(self) -> "UserState":
        return copy.deepcopy(self)

