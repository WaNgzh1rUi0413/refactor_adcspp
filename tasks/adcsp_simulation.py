"""Pipeline task for generating and persisting final synthetic traffic."""

from __future__ import annotations

from tasks.abstract_task import AbstractTask
from utils.easydict import EasyDict
from inputs.adcsp.synthetic_traffic import SyntheticTrafficConfig, SyntheticTrafficGenerator


class AdCSPPSimulationTask(AbstractTask):
    step = "simulate"
    self_managed = True

    def __init__(self, cfg: EasyDict | None = None):
        self.cfg = cfg or EasyDict()
        self.result = None

    def run(self) -> None:
        config = SyntheticTrafficConfig(**dict(self.cfg.get("simulation", self.cfg)))
        self.result = SyntheticTrafficGenerator(config).generate()
