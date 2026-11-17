import json
import tempfile
import unittest
from pathlib import Path

from inputs.adcsp.synthetic_traffic import SyntheticTrafficConfig, SyntheticTrafficGenerator


class AdCSPPGeneratorTest(unittest.TestCase):
    def test_only_final_online_traffic_is_persisted(self):
        with tempfile.TemporaryDirectory(prefix="adcsp_test_") as directory:
            result = SyntheticTrafficGenerator(SyntheticTrafficConfig(num_users=2, exposures_per_user=3, output_dir=directory, seed=9)).generate()
            files = [path.name for path in Path(directory).iterdir()]
            self.assertEqual(files, ["traffic.jsonl"])
            with Path(result.output_path).open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(result.user_count, 2)
            self.assertEqual(result.event_count, len(records))
            self.assertTrue(records)
            self.assertTrue(all("auid" in record and "slot_id" in record for record in records))
            self.assertTrue(all("is_click" in record and "is_conversion" in record for record in records))
            self.assertTrue(all("time_stamp" in record and "event_type" not in record for record in records))

    def test_feature_vocab_is_read_from_config(self):
        with tempfile.TemporaryDirectory(prefix="adcsp_vocab_") as directory:
            config = SyntheticTrafficConfig(
                num_users=1,
                exposures_per_user=1,
                output_dir=directory,
                seed=3,
                feature_config={
                    "categorical_vocab": {
                        "slot_id": ["slot_x"],
                        "ad_category": ["commerce"],
                        "gender": ["female"],
                        "age_group": ["25_34"],
                        "region": ["east"],
                        "device_type": ["phone"],
                        "os_type": ["ios"],
                        "user_segment": ["active"],
                        "user_value_segment": ["medium"],
                    },
                },
            )
            result = SyntheticTrafficGenerator(config).generate()
            record = json.loads(Path(result.output_path).read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["slot_id"], "slot_x")

