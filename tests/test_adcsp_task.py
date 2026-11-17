import tempfile
import unittest
from pathlib import Path

from tasks.adcsp_simulation import AdCSPPSimulationTask
from utils.easydict import EasyDict


class AdCSPPTaskTest(unittest.TestCase):
    def test_simulation_task_persists_only_final_traffic(self):
        with tempfile.TemporaryDirectory(prefix="adcsp_task_") as directory:
            task = AdCSPPSimulationTask(EasyDict({
                "type": "simulate",
                "simulation": {
                    "num_users": 1,
                    "exposures_per_user": 2,
                    "output_dir": directory,
                    "seed": 4,
                },
            }))
            task.run()
            self.assertTrue(Path(task.result.output_path).exists())
            self.assertEqual([item.name for item in Path(directory).iterdir()], ["traffic.jsonl"])

