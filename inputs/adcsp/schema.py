"""Model contracts for online AdCSPP inference and training."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExposureContext:
    impression_id: str
    event_timestamp: int
    slot_id: str
    features: Dict[str, Any] = field(default_factory=dict)

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

