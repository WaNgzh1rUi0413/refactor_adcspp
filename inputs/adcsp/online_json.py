"""Read one production flat JSON traffic row for AdCSPP inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from inputs.adcsp.schema import ExposureContext, UserProfile, UserState
from models.adcsp.state import UserStateBuilder


_SEQUENCE_ALIASES = {
    "hist_exp_slot": "hist_exp_slots",
    "hist_exp_slots": "hist_exp_slots",
    "hist_exp_ad": "hist_exp_ads",
    "hist_exp_ads": "hist_exp_ads",
    "hist_clk_slot": "hist_clk_slots",
    "hist_clk_slots": "hist_clk_slots",
    "hist_clk_ad": "hist_clk_ads",
    "hist_clk_ads": "hist_clk_ads",
    "hist_clk_category": "hist_clk_categories",
    "hist_clk_categories": "hist_clk_categories",
    "hist_conv_slot": "hist_conv_slots",
    "hist_conv_slots": "hist_conv_slots",
    "hist_conv_target": "hist_conv_targets",
    "hist_conv_targets": "hist_conv_targets",
}


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return [item.strip().strip("'\"") for item in text.split(",") if item.strip()]
    return [value]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class OnlineTrafficRecord:
    """Canonical model input for one flat online JSON row."""

    raw: Dict[str, Any]
    user_id: str
    timestamp: int
    sequence_features: Dict[str, list[Any]]

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "OnlineTrafficRecord":
        raw = dict(row)
        user_id = str(row.get("auid", row.get("user_id", "unknown")))
        timestamp = int(row.get("time_stamp", row.get("event_timestamp", 0)))
        sequences: Dict[str, list[Any]] = {}
        for name, value in row.items():
            canonical = _SEQUENCE_ALIASES.get(str(name), str(name) if str(name).startswith("hist_") else None)
            if canonical:
                sequences[canonical] = _list_value(value)
        return cls(raw=raw, user_id=user_id, timestamp=timestamp, sequence_features=sequences)

    def profile(self) -> UserProfile:
        row = self.raw
        return UserProfile(
            user_id=self.user_id,
            gender=str(row.get("gender", "unknown")),
            age_group=str(row.get("age_group", "unknown")),
            region=str(row.get("region", "unknown")),
            device_type=str(row.get("device_type", "unknown")),
            os_type=str(row.get("os_type", "unknown")),
            user_segment=str(row.get("user_segment", "unknown")),
            user_long_term_interest=tuple(_list_value(row.get("user_long_term_interest"))),
            user_registration_age=_float(row.get("user_registration_age")),
            user_value_segment=str(row.get("user_value_segment", "unknown")),
        )

    def context(self) -> ExposureContext:
        row = self.raw
        slot_id = str(row.get("slot_id", "unknown"))
        ad_id = str(row.get("ad_id", "unknown"))
        impression_id = str(row.get("impression_id", f"{self.user_id}_{self.timestamp}_{slot_id}"))
        return ExposureContext(
            impression_id=impression_id,
            event_timestamp=self.timestamp,
            slot_id=slot_id,
            slot_type=str(row.get("slot_type", "unknown")),
            slot_position=str(row.get("slot_position", "unknown")),
            ad_id=ad_id,
            creative_id=str(row.get("creative_id", "unknown")),
            campaign_id=str(row.get("campaign_id", "unknown")),
            advertiser_id=str(row.get("advertiser_id", "unknown")),
            ad_category=str(row.get("ad_category", "unknown")),
            objective_type=str(row.get("objective_type", "unknown")),
            creative_type=str(row.get("creative_type", "unknown")),
            creative_size=str(row.get("creative_size", "unknown")),
            bid_price=_float(row.get("bid_price")),
            ecpm=_float(row.get("ecpm")),
            ad_quality_score=_float(row.get("ad_quality_score")),
            landing_page_type=str(row.get("landing_page_type", "unknown")),
            promotion_target=str(row.get("promotion_target", "unknown")),
            session_id=str(row.get("session_id", "unknown")),
            traffic_source=str(row.get("traffic_source", "unknown")),
            app_category=str(row.get("app_category", "unknown")),
            page_context=str(row.get("page_context", "unknown")),
            network_type=str(row.get("network_type", "unknown")),
            is_holiday=bool(row.get("is_holiday", False)),
        )

    def model_inputs(self, latent_dim: int = 16, feature_config: Optional[Mapping[str, Any]] = None) -> tuple[UserState, ExposureContext]:
        state = UserStateBuilder.build(
            self.profile(),
            aggregate_features=self.sequence_features,
            latent_dim=latent_dim,
            feature_config=feature_config,
        )
        return state, self.context()

