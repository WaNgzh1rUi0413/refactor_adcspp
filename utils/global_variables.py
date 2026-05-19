"""
需要在运行过程中动态修改的全局变量，放在此文件中，常量请放在constants.py中
"""

import os
from enum import Enum

from utils.config import constants

IS_VERBOSE = True
UT_FEATURE_MAX_INDEX = 1000
UT_RAND_DATA_BATCH_SIZE = 100
RANDOM_SEED = 0
TOTAL_CONFIG = None

DEFAULT_INPUT_TYPE_MAPPING = {
    "sparse": "int64",
    "dense": "float32",
    "sparse_seq": "int64",
    "labels": "float32",
    "extra_input": "float32",
}

# 根据/opt/huawei路径是否存在，判断是否为MTP环境
IS_MTP = os.path.exists(constants.MTP_ALG_DIR)


class SaveFile(Enum):
    MONITOR_FILE = "monitor.csv"
    MCR_FEATURE_IMPORTANCE_FILE = "mcr_feature_importance.csv"


class MTPConfig(object):
    def __init__(self, date=None, version=None, scene_name=None, model_id=None):
        self._date = date
        self._version = version
        self._scene_name = scene_name
        self._model_id = model_id

    @property
    def date(self):
        return self._date

    @property
    def version(self):
        return self._version

    @property
    def scene_name(self):
        return self._scene_name

    @property
    def model_id(self):
        return self._model_id

    def __repr__(self):
        return str(
            dict(
                date=self.date,
                version=self.version,
                scene_name=self.scene_name,
                model_id=self.model_id,
            )
        )

    def set_value(self, date="", version="", scene_name="", model_id=""):
        self._date = date
        self._version = "" if version == "not_use" else version
        self._scene_name = scene_name
        self._model_id = model_id


MTP_TASK_CONFIG = MTPConfig(date="", version="", scene_name="", model_id="")


class PathConfig:
    def __init__(self):
        pass

    @property
    def checkpoint_path(self):
        if IS_MTP:
            path = os.path.join(
                constants.MTP_DATA_DIR,
                MTP_TASK_CONFIG.scene_name,
                MTP_TASK_CONFIG.model_id,
                "model",
                MTP_TASK_CONFIG.version,
                MTP_TASK_CONFIG.date,
                "checkpoints",
            )
        else:
            path = os.path.join(constants.LOCAL_OUTPUT_DIR, "checkpoints")
        return path

    @property
    def model_export_path(self):
        if IS_MTP:
            path = os.path.join(
                constants.MTP_DATA_DIR,
                MTP_TASK_CONFIG.scene_name,
                MTP_TASK_CONFIG.model_id,
                "model",
                MTP_TASK_CONFIG.version,
                MTP_TASK_CONFIG.date,
                "modelfile",
            )
        else:
            path = os.path.join(constants.LOCAL_OUTPUT_DIR, "modelfile")
        return path

    @property
    def visualization_file_path(self):
        if IS_MTP:
            path = os.path.join(
                constants.MTP_DATA_DIR,
                MTP_TASK_CONFIG.scene_name,
                MTP_TASK_CONFIG.model_id,
                "model",
                MTP_TASK_CONFIG.version,
                MTP_TASK_CONFIG.date,
                "visualization_file",
            )
        else:
            path = os.path.join(constants.LOCAL_OUTPUT_DIR, "visualization_file")
        return path

    @property
    def model_train_config_path(self):
        return os.path.join(
            constants.MTP_DATA_DIR,
            MTP_TASK_CONFIG.scene_name,
            MTP_TASK_CONFIG.model_id,
            "config",
            MTP_TASK_CONFIG.date,
            "modelTrain.json",
        )

    @property
    def feature_map_path(self):
        if IS_MTP:
            return os.path.join(
                constants.MTP_DATA_DIR,
                MTP_TASK_CONFIG.scene_name,
                MTP_TASK_CONFIG.model_id,
                "config",
                MTP_TASK_CONFIG.date,
                "featureMap.json",
            )
        return os.path.join(
            constants.LOCAL_OUTPUT_DIR,
            "config",
            MTP_TASK_CONFIG.date,
            constants.DPP_FEATURE_MAP_CONFIG,
        )

    @property
    def feature_set_path(self):
        if IS_MTP:
            return constants.DPP_FEATURE_SET_PATH
        return os.path.join(constants.LOCAL_OUTPUT_DIR, "modelFeatureSet.json")

    @property
    def assemble_feature_set_path(self):
        if IS_MTP:
            return constants.ASSEMBLE_FEATURE_SET_PATH
        return os.path.join(constants.LOCAL_OUTPUT_DIR, "assembleFeatureSet.json")

    @property
    def monitor_csv_path(self):
        if IS_MTP:
            path = os.path.join(
                constants.MTP_DATA_DIR,
                MTP_TASK_CONFIG.scene_name,
                MTP_TASK_CONFIG.model_id,
                "model",
                MTP_TASK_CONFIG.version,
                MTP_TASK_CONFIG.date,
                SaveFile.MONITOR_FILE.value,
            )
        else:
            path = os.path.join(constants.LOCAL_OUTPUT_DIR, SaveFile.MONITOR_FILE.value)
        return path

    @property
    def mcr_feature_importance_csv_path(self):
        if IS_MTP:
            path = os.path.join(
                constants.MTP_DATA_DIR,
                MTP_TASK_CONFIG.scene_name,
                MTP_TASK_CONFIG.model_id,
                "model",
                MTP_TASK_CONFIG.version,
                MTP_TASK_CONFIG.date,
                SaveFile.MCR_FEATURE_IMPORTANCE_FILE.value,
            )
        else:
            path = os.path.join(constants.LOCAL_OUTPUT_DIR, SaveFile.MCR_FEATURE_IMPORTANCE_FILE.value)
        return path

    @property
    def result_file_path(self):
        if IS_MTP:
            path = os.path.join(
                constants.MTP_DATA_DIR,
                MTP_TASK_CONFIG.scene_name,
                MTP_TASK_CONFIG.model_id,
                "model",
                MTP_TASK_CONFIG.version,
                MTP_TASK_CONFIG.date,
                "result.txt",
            )
        else:
            path = os.path.join(constants.LOCAL_OUTPUT_DIR, "result.txt")
        return path

    @property
    def model_version_path(self):
        if IS_MTP:
            path = os.path.join(
                constants.MTP_DATA_DIR,
                MTP_TASK_CONFIG.scene_name,
                MTP_TASK_CONFIG.model_id,
            )
        else:
            path = os.path.join(constants.LOCAL_OUTPUT_DIR, "modelfile")
        return path


PATH_CONFIG = PathConfig()

if __name__ == "__main__":
    cfg = MTPConfig(date="1", version="2", scene_name="3", model_id="4")
