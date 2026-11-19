import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import math

from inputs.adcsp.tfrecord_adapter import AdCSPPTFRecordAdapter
from tasks.adcsp_flow import AdCSPPFlowGenerateTask, AdCSPPFlowPredictTask, AdCSPPFlowTrainTask
from utils.easydict import EasyDict


def batch():
    features = {
        "auid": np.array(["u1", "u2"]),
        "slot_id": np.array(["slot_a", "slot_b"]),
        "logid": np.array([100, 101]),
        "auid_ctr_90d": np.array([0.1, 0.2], dtype=np.float32),
        "auid_hwdsp_clk_slotid_seq_time_365d": np.array([["slot_a"], ["slot_b"]]),
    }
    labels = {"label": np.array([1.0, 0.0]), "label_keyaction": np.array([1.0, 0.0])}
    return features, labels


class AdCSPPFlowTaskTest(unittest.TestCase):
    def test_tfrecord_adapter_builds_model_example(self):
        features, labels = batch()
        schema = {
            "user_id_field": "auid",
            "static_numeric_fields": ["auid_ctr_90d"],
            "sequence_fields": ["auid_hwdsp_clk_slotid_seq_time_365d"],
            "context_categorical_fields": ["slot_id", "slot_form", "creativetype"],
            "context_numeric_fields": ["slot_quality_score"],
            "labels": {"click": "label", "conversion": "label_keyaction"},
            "event_timestamp_field": "logid",
        }
        example = AdCSPPTFRecordAdapter(feature_schema=schema).example(features, labels, 0, feature_config=schema)
        self.assertEqual(example.state.user_id, "u1")
        self.assertEqual(example.context.slot_id, "slot_a")
        self.assertEqual(example.state.sequence_features["auid_hwdsp_clk_slotid_seq_time_365d"], ["slot_a"])
        self.assertEqual(example.is_click, 1.0)

    def test_tfrecord_adapter_matches_production_schema_and_labels(self):
        features = {
            "auid": np.array(["u1"]),
            "slot_id": np.array(["s6711"]),
            "auid_ctr_90d": np.array([0.12], dtype=np.float32),
            "rt_auid_click_advertiser_list_7d": np.array([["adv1", ""]]),
            "auid_hwdsp_clk_slotid_seq_time_365d": np.array([["old_slot", ""]]),
            "slot_form": np.array(["feed"]),
            "creativetype": np.array(["video"]),
            "logid": np.array(["20261001101156017206704654853"]),
        }
        labels = {"label": np.array([1.0]), "label_keyaction": np.array([0.0])}
        example = AdCSPPTFRecordAdapter().example(features, labels, 0)
        self.assertEqual(example.state.user_id, "u1")
        self.assertEqual(example.context.slot_id, "s6711")
        self.assertLess(example.context.event_timestamp, 2**63)
        self.assertGreater(example.context.event_timestamp, 0)
        self.assertEqual(example.context.features["slot_form"], "feed")
        self.assertAlmostEqual(example.state.dynamic_features["auid_ctr_90d"], 0.12, places=6)
        self.assertEqual(example.state.sequence_features["auid_hwdsp_clk_slotid_seq_time_365d"], ["old_slot"])
        self.assertEqual(example.is_click, 1.0)
        self.assertEqual(example.is_conversion, 0.0)

    def test_flow_train_and_predict_use_platform_dataset(self):
        features, labels = batch()
        dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(2)
        with tempfile.TemporaryDirectory(prefix="adcsp_flow_train_") as directory:
            checkpoint_path = Path(directory) / "model.weights.h5"
            train = AdCSPPFlowTrainTask(EasyDict({
                "type": "AdCSPPFlowTrainTask",
                "checkpoint_path": str(checkpoint_path),
                "epochs": 1,
                "model": {"latent_dim": 16, "static_dim": 16, "context_dim": 16, "hidden_dim": 8},
            }))
            train.train_dataset = dataset
            train.run()
            self.assertTrue(checkpoint_path.exists())
            prior_path = checkpoint_path.with_name("adcsp_flow_prior.json")
            self.assertTrue(prior_path.exists())

            predict = AdCSPPFlowPredictTask(EasyDict({
                "type": "AdCSPPFlowPredictTask",
                "checkpoint_path": str(checkpoint_path),
                "model": {"latent_dim": 16, "static_dim": 16, "context_dim": 16, "hidden_dim": 8},
            }))
            predict.test_dataset = dataset
            predict.run()
            self.assertEqual(len(predict.results["pred_click_probability"]), 2)
            self.assertEqual(len(predict.results["pred_conversion_probability"]), 2)
            for key in ("pred_click_probability", "pred_conversion_probability"):
                self.assertTrue(all(math.isfinite(value) for value in predict.results[key]))
                self.assertTrue(all(0.0 <= value <= 1.0 for value in predict.results[key]))

            generate = AdCSPPFlowGenerateTask(EasyDict({
                "type": "AdCSPPFlowGenerateTask",
                "checkpoint_path": str(checkpoint_path),
                "flow_prior_path": str(prior_path),
                "generate_new_contexts": True,
                "num_steps": 3,
                "sample": False,
                "model": {"latent_dim": 16, "static_dim": 16, "context_dim": 16, "hidden_dim": 8},
            }))
            generate.generate_dataset = dataset
            generate.run()
            self.assertEqual(generate.result["user_count"], 2)
            self.assertEqual(generate.result["step_count"], 6)

            prior_only = AdCSPPFlowGenerateTask(EasyDict({
                "type": "AdCSPPFlowGenerateTask",
                "checkpoint_path": str(checkpoint_path),
                "flow_prior_path": str(prior_path),
                "generate_from_prior": True,
                "num_users": 2,
                "num_steps": 3,
                "start_timestamp": 100,
                "sample": False,
                "model": {"latent_dim": 16, "static_dim": 16, "context_dim": 16, "hidden_dim": 8},
            }))
            prior_only.run()
            self.assertEqual(prior_only.result["user_count"], 2)
            self.assertEqual(prior_only.result["step_count"], 6)

    def test_flow_generate_rolls_real_request_contexts(self):
        features, labels = batch()
        dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(2)
        with tempfile.TemporaryDirectory() as directory:
            generate = AdCSPPFlowGenerateTask(EasyDict({
                "type": "AdCSPPFlowGenerateTask",
                "model": {"latent_dim": 16, "static_dim": 16, "context_dim": 16, "hidden_dim": 8},
                "sample": False,
                "output": {"save_traffic": True, "output_dir": directory, "file_name": "traffic.jsonl"},
            }))
            generate.test_dataset = dataset
            generate.generate_dataset = dataset
            generate.run()
            self.assertEqual(generate.result["step_count"], 2)
            self.assertTrue(Path(generate.result["traffic_path"]).exists())
            traffic_lines = Path(generate.result["traffic_path"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(traffic_lines), 2)
            traffic_row = json.loads(traffic_lines[0])
            self.assertEqual(traffic_row["auid"], "u1")
            self.assertIn(traffic_row["is_click"], (0, 1))
            self.assertIn("auid_ctr_90d", traffic_row)
            self.assertIn("auid_hwdsp_clk_slotid_seq_time_365d", traffic_row)
            self.assertNotIn("click_probability", traffic_row)
            self.assertNotIn("conversion_probability", traffic_row)
            self.assertNotIn("events", traffic_row)
            self.assertEqual(len(generate.results["trajectories"]), 2)
            for item in generate.results["trajectories"]:
                self.assertIn("state_before", item)
                self.assertIn("state_after", item)
                self.assertTrue(0.0 <= item["click_probability"] <= 1.0)
                self.assertTrue(0.0 <= item["conversion_probability"] <= 1.0)
                self.assertFalse(item["is_conversion"] and not item["is_click"])

    def test_opportunity_head_can_train_from_observed_gaps(self):
        features = {
            "auid": np.array(["u1", "u1", "u1"]),
            "slot_id": np.array(["slot_a", "slot_b", "slot_a"]),
            "logid": np.array([100, 110, 120]),
            "auid_ctr_90d": np.array([0.1, 0.1, 0.1], dtype=np.float32),
            "auid_hwdsp_clk_slotid_seq_time_365d": np.array([["slot_a"], ["slot_b"], ["slot_a"]]),
        }
        labels = {"label": np.array([1.0, 0.0, 1.0]), "label_keyaction": np.array([1.0, 0.0, 0.0])}
        dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(3)
        with tempfile.TemporaryDirectory(prefix="adcsp_joint_train_") as directory:
            checkpoint_path = Path(directory) / "model.weights.h5"
            train = AdCSPPFlowTrainTask(EasyDict({
                "type": "AdCSPPFlowTrainTask",
                "checkpoint_path": str(checkpoint_path),
                "epochs": 1,
                "opportunity_training": {"enabled": True},
                "model": {"latent_dim": 16, "static_dim": 16, "context_dim": 16, "hidden_dim": 8},
            }))
            train.train_dataset = dataset
            train.run()
            self.assertGreaterEqual(train.result["opportunity_pairs"], 2)
            self.assertTrue(math.isfinite(train.result["opportunity_loss"]))

