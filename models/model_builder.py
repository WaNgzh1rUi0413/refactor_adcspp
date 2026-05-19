from utils.easydict import EasyDict


class ModelBuilder(object):
    """
    模型工厂类
    """

    @classmethod
    def build(cls, cfg: EasyDict):
        """
        :param cfg: EasyDict, 模型config示例
            {
                "type": "RankingModel",
                "network": {
                    "type": "DNN"
                },
                "feature": {
                    "type": "ToyFeatureOp",
                    "num_sparse": 2,
                    "output_dim": 8
                },
                "loss": {
                    "type": "BinaryCrossentropy"
                },
                "optimizer": {
                    "type": "Adam"
                },
                "metric": [
                    {"metric_name": 'AUC'},
                    {"metric_name": 'LogLoss'},
                    {"metric_name": 'PCLQ'}
                ]
            }
        :return: cfg.type指定的具体Model类的对象
        """
        model_type = cfg.type
        if model_type in ("generation_model", "FlowGenerationModel"):
            from .generation_model import GenerationModel

            return GenerationModel(cfg)
        if model_type == "RankingModel":
            from .ranking_model import RankingModel

            return RankingModel(cfg)
        raise ValueError(f"Unknown model_type: {model_type}")
