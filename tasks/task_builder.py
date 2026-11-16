from utils.easydict import EasyDict
from .predict import Predict
from .train import Train


mapping = {
    "train": Train,
    "Train": Train,
    "TrainTask": Train,
    "predict": Predict,
    "Predict": Predict,
    "PredictTask": Predict,
}


class TaskBuilder(object):
    @classmethod
    def build(cls, cfg: EasyDict):
        task_type = cfg.get("type")
        if task_type not in mapping:
            raise ValueError(f"Unknown task type: {task_type}")
        return mapping.get(task_type)(cfg)
