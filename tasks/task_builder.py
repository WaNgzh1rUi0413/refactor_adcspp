from utils.easydict import EasyDict
from .predict import PredictTask
from .train import TrainTask


mapping = {
    "train": TrainTask,
    "TrainTask": TrainTask,
    "predict": PredictTask,
    "PredictTask": PredictTask,
}


class TaskBuilder(object):
    @classmethod
    def build(cls, cfg: EasyDict):
        task_type = cfg.get("type")
        if task_type not in mapping:
            raise ValueError(f"Unknown task type: {task_type}")
        return mapping.get(task_type)(cfg)
