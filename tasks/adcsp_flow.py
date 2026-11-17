"""Tasks dedicated to AdCSPP traffic generation and online extrapolation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

from inputs.adcsp.online_json import OnlineTrafficRecord
from inputs.adcsp.synthetic_traffic import SyntheticTrafficConfig, SyntheticTrafficGenerator
from models.adcsp.model import AdCSPPModel
from utils import global_variables as gv
from utils.easydict import EasyDict
from .abstract_task import AbstractTask


def _root_config() -> Mapping[str, Any]:
    return getattr(gv, "TOTAL_CONFIG", {}) or {}


class AdCSPPFlowGenerateTask(AbstractTask):
    """Local-only flow generator using the AdCSPP behavior model."""

    step = "adcsp_flow_generate"
    self_managed = True

    def __init__(self, cfg: EasyDict | None = None):
        self.cfg = cfg or EasyDict()
        self.result = None

    def run(self) -> None:
        simulation_cfg = self.cfg.get("simulation", self.cfg)
        model_cfg = self.cfg.get("model", {})
        model = AdCSPPModel(
            latent_dim=int(model_cfg.get("latent_dim", simulation_cfg.get("latent_dim", 16))),
            static_dim=int(model_cfg.get("static_dim", 16)),
            context_dim=int(model_cfg.get("context_dim", 16)),
            hidden_dim=int(model_cfg.get("hidden_dim", 32)),
            seed=int(model_cfg.get("seed", simulation_cfg.get("seed", 42))),
            feature_config=model_cfg.get("feature_config", simulation_cfg.get("feature_config", {})),
        )
        self.result = SyntheticTrafficGenerator(SyntheticTrafficConfig(**dict(simulation_cfg)), model=model).generate()
        logging.info("AdCSPP flow generation finished: %s", self.result.output_path)


class AdCSPPFlowPredictTask(AbstractTask):
    """Online conditional prediction task; it does not train a supervised model."""

    step = "adcsp_flow_predict"
    self_managed = True

    def __init__(self, cfg: EasyDict | None = None):
        self.cfg = cfg or EasyDict()
        self.result = None

    def _config(self, name: str, default=None):
        if name in self.cfg:
            return self.cfg.get(name)
        return _root_config().get(name, default)

    def run(self) -> None:
        input_cfg = self._config("input", {}) or {}
        model_cfg = self._config("model", {}) or {}
        output_cfg = self.cfg.get("output", {}) or {}
        input_path = self.cfg.get("input_path", input_cfg.get("path"))
        output_path = self.cfg.get("output_path", output_cfg.get("path", "traffic_output/traffic_predictions.jsonl"))
        if not input_path:
            raise ValueError("AdCSPPFlowPredictTask requires input.path or input_path")

        model = AdCSPPModel(
            latent_dim=int(model_cfg.get("latent_dim", 16)),
            static_dim=int(model_cfg.get("static_dim", 16)),
            context_dim=int(model_cfg.get("context_dim", 16)),
            hidden_dim=int(model_cfg.get("hidden_dim", 32)),
            seed=int(model_cfg.get("seed", 42)),
            feature_config=model_cfg.get("feature_config", {}),
        )
        model.eval()
        click_window = int(self.cfg.get("click_window_seconds", 30))
        conversion_window = int(self.cfg.get("conversion_window_seconds", 3600))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with Path(input_path).open(encoding="utf-8") as source, output.open("w", encoding="utf-8") as target:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                record = OnlineTrafficRecord.from_dict(row)
                state, context = record.model_inputs(latent_dim=int(model_cfg.get("latent_dim", 16)), feature_config=model_cfg.get("feature_config", {}))
                p_click = model.click_probability(state, context, click_window)
                p_conversion = model.conversion_probability(state, context, conversion_window)
                result = dict(row)
                result["pred_click_probability"] = p_click
                result["pred_conversion_probability"] = p_conversion
                result["expected_future_conversion"] = p_click * p_conversion
                target.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        self.result = {"output_path": str(output), "record_count": count}
        logging.info("AdCSPP online flow prediction finished: %s (%d records)", output, count)

