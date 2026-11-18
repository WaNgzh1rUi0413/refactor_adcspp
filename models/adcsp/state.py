"""Build model state from online static and precomputed historical features."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from inputs.adcsp.schema import UserProfile, UserState


class UserStateBuilder:
    """Create the state consumed by the model from platform features.

    State keys are kept identical to the online feature schema.  No synthetic
    aliases are introduced, so a state can be passed back to the platform
    without a feature-name translation step.
    """
    PRODUCTION_SEQUENCE_FIELDS = {
        "rt_auid_hwdsp_imp_unclk_3rd_indid_seq_time_12h",
        "auid_hwdsp_clk_slotid_seq_time_365d",
        "auid_hwdsp_clk_ts_seq_time_365d",
        "auid_hwdsp_conv_slotid_seq_time_365d",
        "auid_hwdsp_conv_ts_seq_time_365d",
        "auid_hwdsp_conv_evttyp_seq_time_365d",
        "auid_hwdsp_conv_ocpx_trgt_seq_time_365d",
        "crtv_clstid_stream",
        "creative_ocr_text_keywords",
        "creative_text_keywords",
        "title_text_keywords",
        "video_prmt_label_class3",
        "slotid_click_unify_prmt_class1_freq_rank_list",
        "slotid_click_unify_prmt_class3_freq_rank_list",
    }

    @staticmethod
    def build(
        profile: UserProfile,
        aggregate_features: Optional[Mapping[str, object]] = None,
        latent_dim: int = 16,
        static_embedding: Optional[np.ndarray] = None,
        feature_config: Optional[Mapping[str, object]] = None,
    ) -> UserState:
        configured_sequences = set()
        if feature_config:
            configured_sequences.update(feature_config.get("sequence_fields", []))
            configured_sequences.update((feature_config.get("sequence_features", {}) or {}).keys())
        sequence_keys = UserStateBuilder.PRODUCTION_SEQUENCE_FIELDS | configured_sequences
        sequence_features = {}
        for key, value in (aggregate_features or {}).items():
            key = str(key)
            if key not in sequence_keys or not isinstance(value, (list, tuple, np.ndarray)):
                continue
            values = [item for item in list(value) if item not in (None, "", 0)]
            sequence_features[key] = list(values)
        dynamic_features = {
            str(key): value
            for key, value in (aggregate_features or {}).items()
            if str(key) not in sequence_keys
        }
        return UserState(
            user_id=profile.user_id,
            static_features=profile.to_dict(),
            static_embedding=np.zeros(8, dtype=np.float32)
            if static_embedding is None
            else np.asarray(static_embedding, dtype=np.float32).copy(),
            latent_state=np.zeros(latent_dim, dtype=np.float32),
            last_update_timestamp=0,
            dynamic_features=dynamic_features,
            sequence_features={key: list(value) for key, value in sequence_features.items()},
            initial_sequence_features={key: list(value) for key, value in sequence_features.items()},
        )

