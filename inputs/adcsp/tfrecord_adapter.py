"""Convert platform-provided TFRecord batches to AdCSPP model contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from inputs.adcsp.schema import ExposureContext, UserProfile, UserState
from models.adcsp.state import UserStateBuilder


DEFAULT_MAPPING = {
    "user_id": "auid",
    "static": {
        "gender": "gender",
        "age_group": "age_group",
        "region": "region",
        "device_type": "device_type",
        "os_type": "os_type",
        "user_segment": "user_segment",
        "user_long_term_interest": "user_long_term_interest",
        "user_registration_age": "user_registration_age",
        "user_value_segment": "user_value_segment",
    },
    "context": {
        "slot_id": "slot_id",
        "slot_type": "slot_type",
        "slot_position": "slot_position",
        "ad_id": "ad_id",
        "creative_id": "creative_id",
        "campaign_id": "campaign_id",
        "advertiser_id": "advertiser_id",
        "ad_category": "ad_category",
        "objective_type": "objective_type",
        "creative_type": "creative_type",
        "creative_size": "creative_size",
        "bid_price": "bid_price",
        "ecpm": "ecpm",
        "ad_quality_score": "ad_quality_score",
        "landing_page_type": "landing_page_type",
        "promotion_target": "promotion_target",
        "session_id": "session_id",
        "traffic_source": "traffic_source",
        "app_category": "app_category",
        "page_context": "page_context",
        "network_type": "network_type",
        "event_timestamp": "time_stamp",
    },
    "sequence": {
        "hist_exp_slots": "hist_exp_slot",
        "hist_exp_ads": "hist_exp_ad",
        "hist_clk_slots": "hist_clk_slot",
        "hist_clk_ads": "hist_clk_ad",
        "hist_clk_categories": "hist_clk_category",
        "hist_conv_slots": "hist_conv_slot",
        "hist_conv_targets": "hist_conv_target",
    },
    "labels": {"click": "is_click", "conversion": "is_conversion"},
}


def _python(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _python(value.item())
        return [_python(item) for item in value.tolist()]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, list):
        return [_python(item) for item in value]
    return value


def _value(container: Mapping[str, Any], name: str | None, index: int, default: Any = None) -> Any:
    if not name or name not in container:
        return default
    value = container[name]
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.ndim == 0:
        return _python(array)
    return _python(array[index])


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AdCSPPExample:
    state: UserState
    context: ExposureContext
    is_click: float
    is_conversion: float


class AdCSPPTFRecordAdapter:
    def __init__(self, mapping: Mapping[str, Any] | None = None):
        self.mapping = dict(DEFAULT_MAPPING)
        for section in ("static", "context", "sequence", "labels"):
            self.mapping[section] = dict(DEFAULT_MAPPING[section])
            self.mapping[section].update((mapping or {}).get(section, {}))
        self.mapping["user_id"] = (mapping or {}).get("user_id", DEFAULT_MAPPING["user_id"])

    def example(self, features: Mapping[str, Any], labels: Mapping[str, Any] | None, index: int, latent_dim: int = 16, feature_config=None) -> AdCSPPExample:
        labels = labels or {}
        static = {field: _value(features, source, index, "unknown") for field, source in self.mapping["static"].items()}
        static["user_long_term_interest"] = tuple(_as_list(static.get("user_long_term_interest")))
        profile = UserProfile(
            user_id=str(_value(features, self.mapping["user_id"], index, "unknown")),
            gender=str(static["gender"]), age_group=str(static["age_group"]), region=str(static["region"]),
            device_type=str(static["device_type"]), os_type=str(static["os_type"]), user_segment=str(static["user_segment"]),
            user_long_term_interest=static["user_long_term_interest"],
            user_registration_age=_number(static["user_registration_age"]),
            user_value_segment=str(static["user_value_segment"]),
        )
        context_values = {field: _value(features, source, index, None) for field, source in self.mapping["context"].items()}
        context = ExposureContext(
            impression_id=str(context_values.get("impression_id") or f"{profile.user_id}_{index}"),
            event_timestamp=int(context_values.get("event_timestamp") or 0),
            slot_id=str(context_values.get("slot_id") or "unknown"), slot_type=str(context_values.get("slot_type") or "unknown"),
            slot_position=str(context_values.get("slot_position") or "unknown"), ad_id=str(context_values.get("ad_id") or "unknown"),
            creative_id=str(context_values.get("creative_id") or "unknown"), campaign_id=str(context_values.get("campaign_id") or "unknown"),
            advertiser_id=str(context_values.get("advertiser_id") or "unknown"), ad_category=str(context_values.get("ad_category") or "unknown"),
            objective_type=str(context_values.get("objective_type") or "unknown"), creative_type=str(context_values.get("creative_type") or "unknown"),
            creative_size=str(context_values.get("creative_size") or "unknown"), bid_price=_number(context_values.get("bid_price")),
            ecpm=_number(context_values.get("ecpm")), ad_quality_score=_number(context_values.get("ad_quality_score")),
            landing_page_type=str(context_values.get("landing_page_type") or "unknown"), promotion_target=str(context_values.get("promotion_target") or "unknown"),
            session_id=str(context_values.get("session_id") or "unknown"), traffic_source=str(context_values.get("traffic_source") or "unknown"),
            app_category=str(context_values.get("app_category") or "unknown"), page_context=str(context_values.get("page_context") or "unknown"),
            network_type=str(context_values.get("network_type") or "unknown"),
        )
        histories = {name: _as_list(_value(features, source, index, [])) for name, source in self.mapping["sequence"].items()}
        state = UserStateBuilder.build(profile, aggregate_features=histories, latent_dim=latent_dim, feature_config=feature_config)
        click_source = self.mapping["labels"].get("click", "is_click")
        conversion_source = self.mapping["labels"].get("conversion", "is_conversion")
        click_value = _value(labels, click_source, index, _value(features, click_source, index, 0))
        conversion_value = _value(labels, conversion_source, index, _value(features, conversion_source, index, 0))
        return AdCSPPExample(state, context, float(click_value or 0.0), float(conversion_value or 0.0))


def batch_size(features: Mapping[str, Any], labels: Mapping[str, Any] | None = None) -> int:
    for container in (features, labels or {}):
        for value in container.values():
            if hasattr(value, "shape") and len(value.shape) > 0:
                return int(value.shape[0])
    return 0

