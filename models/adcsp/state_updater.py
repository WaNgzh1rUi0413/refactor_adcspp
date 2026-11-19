"""Explicit dynamic feature updater for AdCSPP user state.

The updater is the single observable-state mutation point.  The latent
transition remains trainable in ``AdCSPPModel``; this class owns sequence
append operations, counters, and last-event fields so an event cannot be
counted twice by different call sites.
"""

from __future__ import annotations

from typing import Any

from inputs.adcsp.schema import ExposureContext, UserState


class DynamicStateUpdater:
    def __init__(self, max_sequence_length: int = 20):
        self.max_sequence_length = int(max_sequence_length)

    def _append(self, state: UserState, name: str, value: Any) -> None:
        values = list(state.sequence_features.get(name, []))
        values.append(value)
        state.sequence_features[name] = values[-self.max_sequence_length:]

    def apply_exposure(self, state: UserState, context: ExposureContext) -> None:
        dynamic = state.dynamic_features
        dynamic["exposure_count"] = int(dynamic.get("exposure_count", 0)) + 1
        dynamic["last_exposure_slot"] = context.slot_id
        dynamic["last_exposure_timestamp"] = int(context.event_timestamp)

    def apply_click(self, state: UserState, context: ExposureContext) -> None:
        self._append(state, "auid_hwdsp_clk_slotid_seq_time_365d", context.slot_id)
        self._append(state, "auid_hwdsp_clk_ts_seq_time_365d", int(context.event_timestamp))
        dynamic = state.dynamic_features
        dynamic["click_count"] = int(dynamic.get("click_count", 0)) + 1
        dynamic["last_click_slot"] = context.slot_id
        dynamic["last_click_timestamp"] = int(context.event_timestamp)

    def apply_no_click(self, state: UserState, context: ExposureContext) -> None:
        category = context.features.get("indu_name", "unknown")
        self._append(state, "rt_auid_hwdsp_imp_unclk_3rd_indid_seq_time_12h", category)

    def apply_conversion(self, state: UserState, context: ExposureContext) -> None:
        target = context.features.get("promotion_target", "unknown")
        event_type = context.features.get("crtvid_conversion_target", target)
        self._append(state, "auid_hwdsp_conv_slotid_seq_time_365d", context.slot_id)
        self._append(state, "auid_hwdsp_conv_ts_seq_time_365d", int(context.event_timestamp))
        self._append(state, "auid_hwdsp_conv_evttyp_seq_time_365d", event_type)
        self._append(state, "auid_hwdsp_conv_ocpx_trgt_seq_time_365d", target)
        dynamic = state.dynamic_features
        dynamic["conversion_count"] = int(dynamic.get("conversion_count", 0)) + 1
        dynamic["last_conversion_timestamp"] = int(context.event_timestamp)

    def apply_outcome(
        self,
        state: UserState,
        context: ExposureContext,
        is_click: int,
        is_conversion: int,
        click_timestamp: int | None = None,
        conversion_timestamp: int | None = None,
    ) -> UserState:
        """Apply one exposure and its observed outcome exactly once."""
        self.apply_exposure(state, context)
        if is_click:
            click_context = context if click_timestamp is None else ExposureContext(
                impression_id=context.impression_id,
                event_timestamp=int(click_timestamp),
                slot_id=context.slot_id,
                features=dict(context.features),
            )
            self.apply_click(state, click_context)
        else:
            self.apply_no_click(state, context)
        if is_conversion and is_click:
            conversion_context = context if conversion_timestamp is None else ExposureContext(
                impression_id=context.impression_id,
                event_timestamp=int(conversion_timestamp),
                slot_id=context.slot_id,
                features=dict(context.features),
            )
            self.apply_conversion(state, conversion_context)
        return state

