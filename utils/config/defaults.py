from utils.easydict import EasyDict

DEFAULTS = EasyDict(
    {
        "run": {
            "steps": ["train", "predict"],
            "log_level": "INFO",
        },
        "model": {
            "type": "RankingModel",
            "feature": {
                "type": "ToyFeatureOp",
                "num_sparse": 2,
                "output_dim": 8,
                "outputs": {},
            },
            "network": {
                "type": "DNN",
            },
            "loss": {
                "type": "BinaryCrossentropy",
            },
            "optimizer": {
                "type": "Adam",
            },
            "metrics": ["accuracy"],
        },
        "data": {
            "type": "TFRecordReader",
            "feature": "",
        },
        "train": {
            "type": "Train",
            "epochs": 2,
            "batch_size": 10000,
        },
        "predict": {
            "type": "Predict",
            "save_result": True,
            "batch_size": 10000,
        },
    }
)
