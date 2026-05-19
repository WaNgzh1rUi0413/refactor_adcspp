import logging
import os

from utils.easydict import EasyDict
from utils.global_variables import PATH_CONFIG
from .abstract_task import AbstractTask
from core.model_export import export_onnx
from utils.dfx import check_feature_set


class ModelCopy(AbstractTask):
    step = "mcp"

    def __init__(self, cfg: EasyDict = None):
        """Initialize MCP task config.

        cfg example:
        {
            "type": "ModelCopy",
            "save_onnx": true,
            "remove_dummy_opset": true
        }
        """
        super().__init__()
        self.cfg = cfg

    def run(self):
        """Execute model export/check process.

        The mcp step mainly keeps compatibility with historical shell scripts.
        """
        target_path = self.cfg.get("model_path") or PATH_CONFIG.model_export_path
        logging.info("Export model is at %s, found=%s", target_path, os.path.exists(target_path))

        # ONNX export needs loading the Keras model first.
        is_save_onnx = self.cfg.get("save_onnx", False)
        is_check_feature_set = self.cfg.get("check_feature_set", True)
        if is_save_onnx or is_check_feature_set:
            from tensorflow import keras

            # compile=False avoids loading loss/metric/optimizer dependencies.
            model = keras.models.load_model(target_path, compile=False)

        if is_save_onnx:
            remove_dummy_opset = self.cfg.get("remove_dummy_opset", True)
            dummy_prefix = self.cfg.get("dummy_prefix", "args_0/")
            export_onnx(model, target_path, remove_dummy_opset, dummy_prefix)

        if is_check_feature_set:
            check_feature_set(model, PATH_CONFIG.feature_set_path)