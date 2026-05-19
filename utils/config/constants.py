from enum import Enum

MODEL_BUILD_MODES = ("trainable", "predictable")
DATASET_SPLITS = ("train", "valid", "test")
PIPELINE_STEPS = ("train", "predict", "mcp", "clean")

DPP_FEATURE_CONFIG_DIR: str = "/opt/huawei/schedule-train/algorithm/featureConfig"
DPP_FEATURE_SET_PATH = (
    "/opt/huawei/schedule-train/algorithm/featureConfig/modelFeatureSet.json"
)
DPP_FEATURE_CONFIG: list = [
    "/featureSetConfig.json",
    "/featureMultiGroupConfig.json",
]
DPP_MODEL_TRAIN_CONFIG = "modelTrain.json"
DPP_FEATURE_MAP_CONFIG = "featureMap.json"
DPP_DATA_SUBDIR = ["train", "valid", "test"]
SEQ_MAX_LENGTH_KEY = "seqMaxLength"
MAX_INDEX_KEY = "maxIndex"
MTP_OUTPUT_DIR: str = "/opt/huawei/schedule-train/output/"
FEATURE_MAP_OUTPUT_NAME: str = "adrcm3-feature-map.featuremap"

MTP_ALG_DIR = "/opt/huawei/schedule-train/algorithm"
MTP_DATA_DIR = "/opt/huawei/dataset/dataDir/"

TENSORBOARD_LOG_DIR: str = "/opt/huawei/schedule-train/log/tensorboard"

LOCAL_OUTPUT_DIR = "test_outputs/"

NON_TRAIN_INPUT_TYPES = ["labels", "extra_input"]
DEFAULT_UNIFORM_MINVAL = -0.05
DEFAULT_UNIFORM_MAXVAL = 0.05
DEFAULT_NORMAL_STDDEV = 0.02
DEFAULT_NORMAL_MEAN = 0
CONFIG_CROSS_SEP = "#"
CROSS_FEATURE_SEPARATORS = ("|", "*", "$", "#")
ACTUAL_CROSS_SEP = "__"
HSTU_PADDING_MASK = 0
HSTU_MASK_VALUE = -1e9
HSTU_VECTOR_NUM = 4

GENERATED_STR_LENGTH = 5
MAX_INDEX_LIMIT = 200
VOCAB_SIZE = 2
DEFAULT_RANDOM_FACTOR = 0.3

EPS = 1e-9
EPS6 = 1e-6


class FeatureType(Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HASH = "hash"
    JOINT = "joint"
    WEIGHT_SEQ = "weight_seq"
    SPARSE_SEQ = "sparse_seq"
    DENSE_SEQ = "dense_seq"


FEATURE_TYPE_SET = {feature_type.value for feature_type in FeatureType}
