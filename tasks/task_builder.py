from utils.easydict import EasyDict
from .train import Train
from .adcsp_flow import AdCSPPSyntheticTrainTask, AdCSPPFlowGenerateTask, AdCSPPFlowPredictTask, AdCSPPFlowTrainTask


mapping = {
    "train": Train,
    "Train": Train,
    "TrainTask": Train,
    "adcsp_flow_predict": AdCSPPFlowPredictTask,
    "AdCSPPFlowPredictTask": AdCSPPFlowPredictTask,
    "adcsp_flow_train": AdCSPPFlowTrainTask,
    "AdCSPPFlowTrainTask": AdCSPPFlowTrainTask,
    "adcsp_flow_generate": AdCSPPFlowGenerateTask,
    "AdCSPPFlowGenerateTask": AdCSPPFlowGenerateTask,
    "adcsp_synthetic_train": AdCSPPSyntheticTrainTask,
    "AdCSPPSyntheticTrainTask": AdCSPPSyntheticTrainTask,
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
