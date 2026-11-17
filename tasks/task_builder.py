from utils.easydict import EasyDict
from .train import Train
from .adcsp_simulation import AdCSPPSimulationTask


mapping = {
    "train": Train,
    "Train": Train,
    "TrainTask": Train,
    "simulate": AdCSPPSimulationTask,
    "Simulate": AdCSPPSimulationTask,
    "AdCSPPSimulationTask": AdCSPPSimulationTask,
}


class TaskBuilder(object):
    @classmethod
    def build(cls, cfg: EasyDict):
        task_type = cfg.get("type")
        if task_type in ("predict", "Predict", "PredictTask"):
            from .predict import Predict
            return Predict(cfg)
        if task_type not in mapping:
            raise ValueError(f"Unknown task type: {task_type}")
        return mapping.get(task_type)(cfg)
