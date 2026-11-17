"""Serialize internal simulated trajectories to the production flat-row format."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

from inputs.adcsp.schema import Event, EventType, ExposureContext, UserProfile


_OUTPUT_SEQUENCE_NAMES = {
    "hist_exp_slots": "hist_exp_slot",
    "hist_exp_ads": "hist_exp_ad",
    "hist_clk_slots": "hist_clk_slot",
    "hist_clk_ads": "hist_clk_ad",
    "hist_clk_categories": "hist_clk_category",
    "hist_conv_slots": "hist_conv_slot",
    "hist_conv_targets": "hist_conv_target",
}


def _limits(feature_config: Mapping[str, Any] | None) -> Dict[str, int]:
    config = (feature_config or {}).get("sequence_features", {})
    return {name: max(1, int(spec.get("max_len", 20))) for name, spec in config.items()}


def _canonical_history(initial: Mapping[str, Sequence[Any]] | None, prior_events: Iterable[Event], feature_config=None):
    """Build histories available immediately before the current exposure."""
    histories: Dict[str, list[Any]] = {}
    for name, values in (initial or {}).items():
        canonical = name
        if name == "hist_exp_slot":
            canonical = "hist_exp_slots"
        elif name == "hist_exp_ad":
            canonical = "hist_exp_ads"
        elif name == "hist_clk_slot":
            canonical = "hist_clk_slots"
        elif name == "hist_clk_ad":
            canonical = "hist_clk_ads"
        elif name == "hist_clk_category":
            canonical = "hist_clk_categories"
        elif name == "hist_conv_slot":
            canonical = "hist_conv_slots"
        elif name == "hist_conv_target":
            canonical = "hist_conv_targets"
        histories.setdefault(canonical, []).extend(list(values))

    for event in prior_events:
        if event.event_type is EventType.EXPOSURE:
            pairs = (("hist_exp_slots", event.slot_id), ("hist_exp_ads", event.ad_id))
        elif event.event_type is EventType.CLICK:
            pairs = (("hist_clk_slots", event.slot_id), ("hist_clk_ads", event.ad_id), ("hist_clk_categories", event.ad_category))
        elif event.event_type is EventType.CONVERSION:
            pairs = (("hist_conv_slots", event.slot_id), ("hist_conv_targets", event.conversion_target))
        else:
            pairs = ()
        for name, value in pairs:
            if value not in (None, ""):
                histories.setdefault(name, []).append(value)

    limits = _limits(feature_config)
    return {name: values[-limits.get(name, 20):] for name, values in histories.items()}


def trajectory_to_online_records(
    profile: UserProfile,
    exposure_plan: Sequence[ExposureContext],
    events: Sequence[Event],
    initial_sequence_features: Mapping[str, Sequence[Any]] | None = None,
    feature_config: Mapping[str, Any] | None = None,
) -> list[Dict[str, Any]]:
    """Convert an event trajectory into one JSON-ready row per exposure.

    The row contains the state available before the exposure and labels observed
    after the trajectory: ``is_click`` and ``is_conversion``. No ``event_type``
    or internal event identifiers are emitted.
    """
    contexts = {context.impression_id: context for context in exposure_plan}
    outcomes: Dict[str, Dict[str, bool]] = {}
    for event in events:
        if event.impression_id is None:
            continue
        outcome = outcomes.setdefault(event.impression_id, {"is_click": False, "is_conversion": False})
        if event.event_type is EventType.CLICK:
            outcome["is_click"] = True
        elif event.event_type is EventType.CONVERSION:
            outcome["is_conversion"] = True

    rows: list[Dict[str, Any]] = []
    prior_events: list[Event] = []
    for event in events:
        if event.event_type is not EventType.EXPOSURE:
            prior_events.append(event)
            continue
        context = contexts.get(event.impression_id)
        if context is None:
            prior_events.append(event)
            continue
        row: Dict[str, Any] = {"auid": profile.user_id}
        row.update(profile.to_dict())
        row.pop("user_id", None)
        row.update(context.to_dict())
        row.pop("impression_id", None)
        row.pop("event_timestamp", None)
        row["time_stamp"] = context.event_timestamp
        histories = _canonical_history(initial_sequence_features, prior_events, feature_config)
        # Keep a stable online schema: every configured history column is
        # present even for the first request of a user.
        configured_names = list((feature_config or {}).get("sequence_features", {}).keys())
        for name in configured_names or _OUTPUT_SEQUENCE_NAMES:
            histories.setdefault(name, [])
        for name, values in histories.items():
            row[_OUTPUT_SEQUENCE_NAMES.get(name, name)] = list(values)
        outcome = outcomes.get(event.impression_id, {"is_click": False, "is_conversion": False})
        row["is_click"] = int(outcome["is_click"])
        row["is_conversion"] = int(outcome["is_conversion"])
        rows.append(row)
        prior_events.append(event)
    return rows

