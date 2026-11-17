import unittest

import torch

from inputs.adcsp.schema import Event, EventType
from losses.adcsp_losses import AdCSPPSurvivalLoss
from metrics.adcsp_metrics import ledger_metrics


class AdCSPPLossMetricsTest(unittest.TestCase):
    def test_survival_loss_returns_finite_components(self):
        result = AdCSPPSurvivalLoss()(torch.tensor([0.5, 0.8]), torch.tensor([0.0, 1.0]), torch.tensor([0.2, 0.7]), torch.tensor([0.0, 1.0]))
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertIn("click_loss", result)
        self.assertIn("conversion_loss", result)

    def test_ledger_metrics_report_zero_constraint_violations(self):
        events = [
            Event("u1", "e1", 100, EventType.EXPOSURE, impression_id="i1", slot_id="slot_a", ad_id="ad1"),
            Event("u1", "e2", 101, EventType.CLICK, impression_id="i1", click_id="c1", parent_event_id="e1"),
            Event("u1", "e3", 110, EventType.CONVERSION, impression_id="i1", click_id="c1", parent_event_id="e2"),
        ]
        metrics = ledger_metrics(events)
        self.assertEqual(metrics["constraint_violations"], 0)
        self.assertEqual(metrics["exposures"], 1)
