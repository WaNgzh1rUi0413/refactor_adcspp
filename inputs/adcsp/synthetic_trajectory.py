"""Offline fixture using the online TFRecord feature names.

Only tests import this module. It emits production fields and production
labels (``label`` and ``label_keyaction``); no compatibility feature aliases
are generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import tensorflow as tf


CORE_SEQUENCE_FIELDS = [
    "auid_hwdsp_clk_slotid_seq_time_365d",
    "auid_hwdsp_clk_ts_seq_time_365d",
    "auid_hwdsp_conv_slotid_seq_time_365d",
    "auid_hwdsp_conv_ts_seq_time_365d",
    "auid_hwdsp_conv_evttyp_seq_time_365d",
    "auid_hwdsp_conv_ocpx_trgt_seq_time_365d",
    "rt_auid_hwdsp_imp_unclk_3rd_indid_seq_time_12h",
]


@dataclass(frozen=True)
class SyntheticTrajectoryConfig:
    num_users: int = 32
    exposures_per_user: int = 8
    sequence_length: int = 20
    seed: int = 7
    start_timestamp: int = 1_700_000_000
    exposure_gap_seconds: int = 300


class SyntheticTrajectoryGenerator:
    """Generate temporally ordered online-schema rows for local tests."""

    def __init__(self, config: SyntheticTrajectoryConfig | None = None):
        self.config = config or SyntheticTrajectoryConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def generate_records(self) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for user_index in range(self.config.num_users):
            user_id = f"synthetic_user_{user_index:05d}"
            counters = {
                "auid_site_click_cnt_90d": 0.0, "auid_normalizedt2_click_cnt_180d": 0.0,
                "auid_ctr_90d": 0.0, "auid_show_cnt_180d": 0.0,
                "auid_app_show_cnt_180d": 0.0, "auid_app_click_cnt_90d": 0.0,
                "auid_app_ctr_90d": 0.0, "auid_appc3_ctr_180d": 0.0,
                "auid_slot_show_cnt_90d": 0.0, "auid_slotid_imp_cnt_30d": 0.0,
                "auid_slot_ctr_90d": 0.0, "auid_slot_click_cnt_90d": 0.0,
                "auid_site_show_cnt_90d": 0.0, "auid_site_ctr_90d": 0.0,
                "auid_normalizedt2_ctr_90d": 0.0,
                "rt_auid_appid_alldsp_imp_cnt_10minute": 0.0,
                "gender_appid_click_cnt_today": 0.0, "gender_slotid_ctr_today": 0.0,
            }
            history = {name: [] for name in CORE_SEQUENCE_FIELDS}
            timestamp = self.config.start_timestamp + user_index * 17
            for step in range(self.config.exposures_per_user):
                slot_id = ["slot_a", "slot_b", "slot_c"][step % 3]
                category = ["commerce", "content", "finance"][step % 3]
                p_click = 1.0 / (1.0 + np.exp(-(-1.1 + 0.18 * len(history[CORE_SEQUENCE_FIELDS[0]]))))
                label = int(self.rng.random() < p_click)
                p_conversion = 1.0 / (1.0 + np.exp(-(-2.0 + 1.1 * label)))
                label_keyaction = int(label and self.rng.random() < p_conversion)
                row = {
                    "auid": user_id, **counters, "slot_id": slot_id,
                    "slot_form": ["feed", "search", "detail"][step % 3],
                    "slot_suited_creative_type": "video", "slot_star_ranking_cn": f"rank_{step % 3}",
                    "slot_avg_click_bid_ecpm_3d": 5.0 + step % 4, "slot_ctr_3rdpty_dsp_7d": 0.02,
                    "slot_ctr_3rdpty_dsp_30d": 0.025, "slot_quality_score": 0.7 + 0.05 * (step % 4),
                    "ecpm_floor_price": 1.0, "price_strategy": "stable",
                    "developer_up_id": f"advertiser_{step % 4}", "developer_up_realname": f"advertiser_name_{step % 4}",
                    "task_name": f"campaign_{step % 5}", "indu_name": category,
                    "objective_type": "purchase", "promotion_target": "purchase",
                    "crtvid_conversion_target": "purchase", "creative_image_size": "square",
                    "is_video": int(step % 2 == 1), "is_short_video": int(step % 3 == 1),
                    "creativetype": ["image", "video", "native"][step % 3],
                    "creative_ocr_text_keywords": [category], "creative_text_keywords": [category],
                    "title_text_keywords": ["offer"], "video_prmt_label_class3": category,
                    "crtv_clstid_stream": [f"cluster_{step % 5}"], "app_introduction": ["shopping"],
                    "app_class1": "shopping", "app_class2": "commerce", "app_class3": category,
                    "app_type": "product", "game_type": "none", "site_app_allias_id": f"site_{user_index % 3}",
                    "site_unique_id": f"site_{user_index % 3}", "media_type_new": "search",
                    "site_click_cnt_30d": counters["auid_site_click_cnt_90d"],
                    "site_paid_cnt_30d": counters["auid_app_click_cnt_90d"], "site_paid_rate_smooth_3d": 0.1,
                    "site_activate_cnt_14d": 0.0, "site_browse_cnt_14d": float(step),
                    "site_click_cnt_14d": counters["auid_site_click_cnt_90d"], "site_register_count_14d": 0.0,
                    "site_retain_cnt_14d": 0.0, "site_show_cnt_14d": float(step + 1),
                    "site_register_hour_top_30d": 12.0, "site_paid_hour_top3_30d": 13.0,
                    "unifiled_tag_level1_all_ssp": "commerce", "unifiled_tag_level2_all_ssp": "retail",
                    "unifiled_tag_level3": category, "normalized_tag2_id": "retail",
                    "slotid_click_unify_prmt_class1_freq_rank_list": ["commerce"],
                    "slotid_click_unify_prmt_class3_freq_rank_list": [category],
                    "rt_adid_hwdsp_pred_loan_clk_credit_score_bin_30minute": 2.0,
                    "rt_adid_hwdsp_pred_loan_clk_credit_score_30minute": 0.4,
                    "logid": timestamp, **{name: list(values) for name, values in history.items()},
                    "label": label, "label_keyaction": label_keyaction,
                }
                records.append(row)
                counters["auid_show_cnt_180d"] += 1
                counters["auid_slot_show_cnt_90d"] += 1
                counters["auid_slotid_imp_cnt_30d"] += 1
                if label:
                    counters["auid_site_click_cnt_90d"] += 1
                    counters["auid_app_click_cnt_90d"] += 1
                    counters["auid_slot_click_cnt_90d"] += 1
                    history["auid_hwdsp_clk_slotid_seq_time_365d"].append(slot_id)
                    history["auid_hwdsp_clk_ts_seq_time_365d"].append(timestamp)
                else:
                    history["rt_auid_hwdsp_imp_unclk_3rd_indid_seq_time_12h"].append(category)
                if label_keyaction:
                    history["auid_hwdsp_conv_slotid_seq_time_365d"].append(slot_id)
                    history["auid_hwdsp_conv_ts_seq_time_365d"].append(timestamp)
                    history["auid_hwdsp_conv_evttyp_seq_time_365d"].append("purchase")
                    history["auid_hwdsp_conv_ocpx_trgt_seq_time_365d"].append("purchase")
                for name in history:
                    history[name] = history[name][-self.config.sequence_length:]
                timestamp += self.config.exposure_gap_seconds
        return records

    def as_dataset(self, batch_size: int = 32) -> tf.data.Dataset:
        records = self.generate_records()
        sequence_fields = set(CORE_SEQUENCE_FIELDS)
        string_fields = {
            "auid", "slot_id", "slot_form", "slot_suited_creative_type", "slot_star_ranking_cn", "price_strategy",
            "developer_up_id", "developer_up_realname", "task_name", "indu_name", "objective_type", "promotion_target",
            "crtvid_conversion_target", "creative_image_size", "creativetype", "video_prmt_label_class3",
            "app_class1", "app_class2", "app_class3", "app_type", "game_type", "site_app_allias_id",
            "site_unique_id", "media_type_new", "unifiled_tag_level1_all_ssp", "unifiled_tag_level2_all_ssp",
            "unifiled_tag_level3", "normalized_tag2_id",
        }
        features = {}
        for name in records[0]:
            if name in ("label", "label_keyaction"):
                continue
            values = [row[name] for row in records]
            if name in sequence_fields or isinstance(values[0], list):
                padded = np.full((len(values), self.config.sequence_length), "", dtype=str)
                for i, items in enumerate(values):
                    items = items[-self.config.sequence_length:]
                    if items:
                        padded[i, -len(items):] = [str(item) for item in items]
                features[name] = padded
            elif name in string_fields:
                features[name] = np.asarray(values, dtype=str)
            else:
                features[name] = np.asarray(values, dtype=np.float32)
        labels = {
            "label": np.asarray([row["label"] for row in records], dtype=np.float32),
            "label_keyaction": np.asarray([row["label_keyaction"] for row in records], dtype=np.float32),
        }
        return tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)
