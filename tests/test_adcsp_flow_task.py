import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf
import math

from inputs.adcsp.tfrecord_adapter import AdCSPPTFRecordAdapter
from tasks.adcsp_flow import AdCSPPFlowPredictTask, AdCSPPFlowTrainTask
from utils.easydict import EasyDict


def batch():
    features = {
        "auid": np.array(["u1", "u2"]),
        "gender": np.array(["female", "male"]),
        "device_type": np.array(["phone", "phone"]),
        "slot_id": np.array(["slot_a", "slot_b"]),
        "time_stamp": np.array([100, 101]),
        "hist_clk_slot": np.array([["slot_a"], ["slot_b"]]),
    }
    labels = {"is_click": np.array([1.0, 0.0]), "is_conversion": np.array([1.0, 0.0])}
    return features, labels


class AdCSPPFlowTaskTest(unittest.TestCase):
    def test_tfrecord_adapter_builds_model_example(self):
        features, labels = batch()
        example = AdCSPPTFRecordAdapter().example(features, labels, 0)
        self.assertEqual(example.state.user_id, "u1")
        self.assertEqual(example.context.slot_id, "slot_a")
        self.assertEqual(example.state.sequence_features["hist_clk_slots"], ["slot_a"])
        self.assertEqual(example.is_click, 1.0)

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

