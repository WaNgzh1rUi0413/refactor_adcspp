import json
import tempfile
import unittest
from pathlib import Path

from inputs.adcsp.synthetic_traffic import SyntheticTrafficConfig, SyntheticTrafficGenerator


class AdCSPPGeneratorTest(unittest.TestCase):
    def test_only_final_event_ledger_is_persisted(self):
        with tempfile.TemporaryDirectory(prefix="adcsp_test_") as directory:
            result = SyntheticTrafficGenerator(SyntheticTrafficConfig(num_users=2, exposures_per_user=3, output_dir=directory, seed=9)).generate()
            files = [path.name for path in Path(directory).iterdir()]
            self.assertEqual(files, ["traffic_event_ledger.jsonl"])
            with Path(result.output_path).open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(result.user_count, 2)
            self.assertEqual(result.event_count, len(records))
            self.assertTrue(records)
            self.assertTrue(all("event_type" in record and "impression_id" in record for record in records))

