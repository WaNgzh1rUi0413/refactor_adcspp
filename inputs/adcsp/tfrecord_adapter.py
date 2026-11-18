"""Convert platform-provided TFRecord batches to AdCSPP model contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

from inputs.adcsp.schema import ExposureContext, UserProfile, UserState
from models.adcsp.state import UserStateBuilder


DEFAULT_MAPPING = {
    "user_id": "auid",
    # The production schema has no direct static profile columns.  Keep this
    # section empty by default; long-term and real-time user state is mapped
    # below and preserved in UserProfile.features/UserState.dynamic_features.
    "static": {},
    "state": {
        "auid_site_click_cnt_90d": "auid_site_click_cnt_90d",
        "auid_normalizedt2_click_cnt_180d": "auid_normalizedt2_click_cnt_180d",
        "auid_ctr_90d": "auid_ctr_90d",
        "auid_show_cnt_180d": "auid_show_cnt_180d",
        "auid_app_show_cnt_180d": "auid_app_show_cnt_180d",
        "auid_app_click_cnt_90d": "auid_app_click_cnt_90d",
        "auid_app_ctr_90d": "auid_app_ctr_90d",
        "auid_appc3_ctr_180d": "auid_appc3_ctr_180d",
        "auid_slot_show_cnt_90d": "auid_slot_show_cnt_90d",
        "auid_slotid_imp_cnt_30d": "auid_slotid_imp_cnt_30d",
        "auid_slot_ctr_90d": "auid_slot_ctr_90d",
        "auid_slot_click_cnt_90d": "auid_slot_click_cnt_90d",
        "auid_site_show_cnt_90d": "auid_site_show_cnt_90d",
        "auid_site_ctr_90d": "auid_site_ctr_90d",
        "auid_normalizedt2_ctr_90d": "auid_normalizedt2_ctr_90d",
        "rt_auid_appid_alldsp_imp_cnt_10minute": "rt_auid_appid_alldsp_imp_cnt_10minute",
        "gender_appid_click_cnt_today": "gender_appid_click_cnt_today",
        "gender_slotid_ctr_today": "gender_slotid_ctr_today",
    },
    "context": {
        "slot_id": "slot_id",
        "slot_form": "slot_form",
        "slot_suited_creative_type": "slot_suited_creative_type",
        "slot_quality_score": "slot_quality_score",
        "slot_star_ranking_cn": "slot_star_ranking_cn",
        "slot_avg_click_bid_ecpm_3d": "slot_avg_click_bid_ecpm_3d",
        "slot_ctr_3rdpty_dsp_7d": "slot_ctr_3rdpty_dsp_7d",
        "slot_ctr_3rdpty_dsp_30d": "slot_ctr_3rdpty_dsp_30d",
        "ecpm_floor_price": "ecpm_floor_price",
        "price_strategy": "price_strategy",
        "creative_image_size": "creative_image_size",
        "is_video": "is_video",
        "is_short_video": "is_short_video",
        "creativetype": "creativetype",
        "creative_ocr_text_keywords": "creative_ocr_text_keywords",
        "creative_text_keywords": "creative_text_keywords",
        "title_text_keywords": "title_text_keywords",
        "video_prmt_label_class3": "video_prmt_label_class3",
        "app_introduction": "app_introduction",
        "app_class1": "app_class1",
        "app_class2": "app_class2",
        "app_class3": "app_class3",
        "app_type": "app_type",
        "game_type": "game_type",
        "site_app_allias_id": "site_app_allias_id",
        "site_unique_id": "site_unique_id",
        "media_type_new": "media_type_new",
        "site_click_cnt_30d": "site_click_cnt_30d",
        "site_paid_cnt_30d": "site_paid_cnt_30d",
        "site_paid_rate_smooth_3d": "site_paid_rate_smooth_3d",
        "site_activate_cnt_14d": "site_activate_cnt_14d",
        "site_browse_cnt_14d": "site_browse_cnt_14d",
        "site_click_cnt_14d": "site_click_cnt_14d",
        "site_register_count_14d": "site_register_count_14d",
        "site_retain_cnt_14d": "site_retain_cnt_14d",
        "site_show_cnt_14d": "site_show_cnt_14d",
        "site_register_hour_top_30d": "site_register_hour_top_30d",
        "site_paid_hour_top3_30d": "site_paid_hour_top3_30d",
        "indu_name": "indu_name",
        "task_name": "task_name",
        "developer_up_id": "developer_up_id",
        "developer_up_realname": "developer_up_realname",
        "crtvid_conversion_target": "crtvid_conversion_target",
        "unifiled_tag_level1_all_ssp": "unifiled_tag_level1_all_ssp",
        "unifiled_tag_level2_all_ssp": "unifiled_tag_level2_all_ssp",
        "unifiled_tag_level3": "unifiled_tag_level3",
        "normalized_tag2_id": "normalized_tag2_id",
        "slotid_click_unify_prmt_class1_freq_rank_list": "slotid_click_unify_prmt_class1_freq_rank_list",
        "slotid_click_unify_prmt_class3_freq_rank_list": "slotid_click_unify_prmt_class3_freq_rank_list",
        "rt_adid_hwdsp_pred_loan_clk_credit_score_bin_30minute": "rt_adid_hwdsp_pred_loan_clk_credit_score_bin_30minute",
        "rt_adid_hwdsp_pred_loan_clk_credit_score_30minute": "rt_adid_hwdsp_pred_loan_clk_credit_score_30minute",
        "event_timestamp": "logid",
    },
    "sequence": {
        "auid_hwdsp_clk_slotid_seq_time_365d": "auid_hwdsp_clk_slotid_seq_time_365d",
        "auid_hwdsp_clk_ts_seq_time_365d": "auid_hwdsp_clk_ts_seq_time_365d",
        "auid_hwdsp_conv_slotid_seq_time_365d": "auid_hwdsp_conv_slotid_seq_time_365d",
        "auid_hwdsp_conv_ts_seq_time_365d": "auid_hwdsp_conv_ts_seq_time_365d",
        "auid_hwdsp_conv_evttyp_seq_time_365d": "auid_hwdsp_conv_evttyp_seq_time_365d",
        "auid_hwdsp_conv_ocpx_trgt_seq_time_365d": "auid_hwdsp_conv_ocpx_trgt_seq_time_365d",
        "rt_auid_hwdsp_imp_unclk_3rd_indid_seq_time_12h": "rt_auid_hwdsp_imp_unclk_3rd_indid_seq_time_12h",
    },
    "labels": {"click": "label", "conversion": "label_keyaction"},
}


def _python(value: Any) -> Any:
    """Convert TensorFlow/NumPy values into plain Python values."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _python(value.item())
        return [_python(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _python(value.item())
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, list):
        return [_python(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_python(item) for item in value)
    return value


def _value(
    container: Mapping[str, Any],
    name: str | None,
    index: int,
    default: Any = None,
    *,
    squeeze_singleton: bool = False,
) -> Any:
    """Read one item from a batched feature tensor.

    Scalar fields may be parsed by TFRecordDataset with shape [1], yielding
    a per-example value such as array([123]). Set ``squeeze_singleton=True``
    for scalar contracts so that this becomes 123. Sequence fields keep their
    list dimension and should use the default ``False``.
    """
    if not name or name not in container:
        return default

    value = container[name]
    if hasattr(value, "numpy"):
        value = value.numpy()

    array = np.asarray(value)
    if array.ndim == 0:
        return _python(array)

    selected = array[index]
    if squeeze_singleton:
        selected_array = np.asarray(selected)
        if selected_array.size == 1:
            selected = selected_array.reshape(-1)[0]

    return _python(selected)


def _scalar_value(value: Any, default: Any = 0) -> Any:
    """Unwrap singleton/list/array values where a scalar is required."""
    value = _python(value)
    while isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    if value is None:
        return default
    return value


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    # Fixed-length TFRecord sequence features use empty-string/zero padding.
    # Padding is not a historical event and must not enter the state encoder.
    return [item for item in values if item not in (None, "", 0)]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _event_timestamp(value: Any, default: int = 0) -> int:
    """Convert the platform logid into a bounded chronological timestamp.

    Online ``logid`` values have the form ``YYYYMMDDHHMMSS`` followed by a
    long uniqueness suffix.  The full value is intentionally preserved as a
    feature, but must not be cast to int64/float32.  Only the 14-digit time
    prefix is used for ordering and state-time features.
    """
    value = _scalar_value(value, default=default)
    if value is None:
        return int(default)
    text = str(value).strip()
    digits = text[1:] if text.startswith("-") else text
    if len(digits) >= 14 and digits[:14].isdigit():
        try:
            parsed = datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
            return int(parsed.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            pass
    try:
        numeric = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return int(default)
    # Keep ordinary numeric timestamps bounded for TensorFlow consumers.
    return numeric if -(2**63) <= numeric <= 2**63 - 1 else int(default)


def _mapping_from_schema(schema: Mapping[str, Any], override: Mapping[str, Any] | None) -> dict:
    """Build an online mapping from YAML field lists.

    When a production ``feature_schema`` is supplied, this method constructs
    the sections directly from the online field lists.
    """
    override = override or {}
    result = {
        "user_id": override.get("user_id", schema.get("user_id_field", "auid")),
        "static": dict(override.get("static", {})),
        "state": dict(override.get("state", {})),
        "context": dict(override.get("context", {})),
        "sequence": dict(override.get("sequence", {})),
        "labels": {"click": "label", "conversion": "label_keyaction"},
    }
    result["labels"].update(schema.get("labels", {}))
    result["labels"].update(override.get("labels", {}))

    state_fields = schema.get("state_fields") or schema.get("static_numeric_fields") or []
    for name in state_fields:
        result["state"].setdefault(name, name)

    context_fields = list(schema.get("context_fields", []))
    if not context_fields:
        context_fields.extend(schema.get("context_categorical_fields", []))
        context_fields.extend(schema.get("context_numeric_fields", []))
    timestamp_field = str(schema.get("event_timestamp_field", "logid"))
    for name in ("slot_id", timestamp_field):
        if name not in context_fields:
            context_fields.append(name)
    for name in context_fields:
        result["context"].setdefault(name, name)
    # Keep the internal contract stable while allowing the online source
    # field to be ``logid`` (the current platform field).
    result["context"].setdefault("event_timestamp", timestamp_field)

    for name in schema.get("sequence_fields", []):
        result["sequence"].setdefault(name, name)
    for name in schema.get("static_categorical_fields", []):
        result["static"].setdefault(name, name)
    return result


@dataclass(frozen=True)
class AdCSPPExample:
    state: UserState
    context: ExposureContext
    is_click: float
    is_conversion: float


class AdCSPPTFRecordAdapter:
    def __init__(self, mapping: Mapping[str, Any] | None = None, feature_schema: Mapping[str, Any] | None = None):
        self.feature_schema = dict(feature_schema or (mapping or {}).get("feature_schema", {}) or {})
        if self.feature_schema:
            self.mapping = _mapping_from_schema(self.feature_schema, mapping)
        else:
            self.mapping = dict(DEFAULT_MAPPING)
            for section in ("static", "state", "context", "sequence", "labels"):
                self.mapping[section] = dict(DEFAULT_MAPPING[section])
                self.mapping[section].update((mapping or {}).get(section, {}))
            self.mapping["user_id"] = (mapping or {}).get("user_id", DEFAULT_MAPPING["user_id"])

    def example(self, features: Mapping[str, Any], labels: Mapping[str, Any] | None, index: int, latent_dim: int = 16, feature_config=None) -> AdCSPPExample:
        labels = labels or {}
        static = {field: _value(features, source, index, None, squeeze_singleton=True) for field, source in self.mapping["static"].items()}
        state_values = {
            field: _value(features, source, index, 0.0, squeeze_singleton=True)
            for field, source in self.mapping["state"].items()
        }
        profile = UserProfile(
            user_id=str(_value(features, self.mapping["user_id"], index, "unknown", squeeze_singleton=True)),
            features={
                **static,
                **{source: _value(features, source, index, None, squeeze_singleton=True) for source in self.mapping["state"].values()},
            },
        )
        context_values = {field: _value(features, source, index, None, squeeze_singleton=True) for field, source in self.mapping["context"].items()}
        context = ExposureContext(
            impression_id=str(context_values.get("impression_id") or f"{profile.user_id}_{index}"),
            event_timestamp=_event_timestamp(
                context_values.get("event_timestamp")
                if context_values.get("event_timestamp") is not None
                else context_values.get("logid"),
            ),
            slot_id=str(context_values.get("slot_id") or "unknown"),
            features={source: _value(features, source, index, None, squeeze_singleton=True) for source in self.mapping["context"].values()},
        )
        histories = {name: _as_list(_value(features, source, index, [])) for name, source in self.mapping["sequence"].items()}
        state = UserStateBuilder.build(
            profile,
            aggregate_features={**state_values, **histories},
            latent_dim=latent_dim,
            feature_config=feature_config,
        )
        click_source = self.mapping["labels"].get("click", "label")
        conversion_source = self.mapping["labels"].get("conversion", "label_keyaction")
        click_value = _value(labels, click_source, index, _value(features, click_source, index, 0, squeeze_singleton=True), squeeze_singleton=True)
        conversion_value = _value(labels, conversion_source, index, _value(features, conversion_source, index, 0, squeeze_singleton=True), squeeze_singleton=True)
        return AdCSPPExample(state, context, float(click_value or 0.0), float(conversion_value or 0.0))


def batch_size(features: Mapping[str, Any], labels: Mapping[str, Any] | None = None) -> int:
    for container in (features, labels or {}):
        for value in container.values():
            if hasattr(value, "shape") and len(value.shape) > 0:
                return int(value.shape[0])
    return 0

