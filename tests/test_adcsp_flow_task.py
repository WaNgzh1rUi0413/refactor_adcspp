import json
import tempfile
import unittest
from pathlib import Path

from tasks.adcsp_flow import AdCSPPFlowPredictTask
from utils.easydict import EasyDict


class AdCSPPFlowTaskTest(unittest.TestCase):
    def test_online_flow_predict_calls_adcspp_model_and_writes_predictions(self):
        with tempfile.TemporaryDirectory(prefix="adcsp_flow_") as directory:
            input_path = Path(directory) / "input.jsonl"
            output_path = Path(directory) / "output.jsonl"
            input_path.write_text(json.dumps({
                "auid": "u1",
                "slot_id": "slot_a",
                "device_type": "phone",
                "hist_clk_slot": ["slot_b"],
                "is_click": 0,
                "is_conversion": 0,
                "time_stamp": 1700000000,
            }) + "\n", encoding="utf-8")
            task = AdCSPPFlowPredictTask(EasyDict({
                "type": "AdCSPPFlowPredictTask",
                "input_path": str(input_path),
                "output_path": str(output_path),
                "model": {"latent_dim": 16, "static_dim": 16, "context_dim": 16, "hidden_dim": 8},
            }))
            task.run()
            result = json.loads(output_path.read_text(encoding="utf-8").strip())
            self.assertIn("pred_click_probability", result)
            self.assertIn("pred_conversion_probability", result)
            self.assertIn("expected_future_conversion", result)
            self.assertGreaterEqual(result["pred_click_probability"], 0.0)
            self.assertLessEqual(result["pred_click_probability"], 1.0)

