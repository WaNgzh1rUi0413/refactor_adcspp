import glob
import os
from typing import Dict, List, Optional

import pandas as pd
import tensorflow as tf

from utils.config.constants import DPP_DATA_SUBDIR, MTP_DATA_DIR
from utils.easydict import EasyDict
from utils.global_variables import MTP_TASK_CONFIG
from utils import global_variables as gv
from .dataset import Dataset

try:
    from utils.get_model_version import get_latest_dataset_period
except ImportError:
    get_latest_dataset_period = None


class FlowTFRecordDataset(Dataset):
    """Load flow-generation source data from .tfrecord.gz files."""

    def __init__(self, cfg: EasyDict):
        self.cfg = cfg
        self.date = MTP_TASK_CONFIG.date
        self.schema = cfg.get("schema", [])
        self.data_path = {}
        self.field_alias = cfg.get(
            "field_alias",
            {
                "app_pkg": "apppkg",
                "slot_id": "slotid",
                "slot_size": "slotsize",
                "ad_type": "adtype",
                "device_type": "devicetype",
            },
        )
        self._init_platform_data_path()

    def _init_platform_data_path(self) -> None:
        scene_name = MTP_TASK_CONFIG.scene_name
        model_id = MTP_TASK_CONFIG.model_id
        period = self.date
        if not (scene_name and model_id and period):
            return

        if gv.TOTAL_CONFIG.get("predict", {}).get("is_use_model_version", False) and get_latest_dataset_period:
            period = get_latest_dataset_period(gv.TOTAL_CONFIG, period)

        base = os.path.join(MTP_DATA_DIR, scene_name, model_id, "data", period)
        self.data_path = {split: os.path.join(base, split) for split in DPP_DATA_SUBDIR}

    @staticmethod
    def _expand_path_vars(path: str) -> str:
        if not path:
            return ""
        mappings = {
            "$dataDir": MTP_DATA_DIR,
            "${dataDir}": MTP_DATA_DIR,
            "$scene_name": MTP_TASK_CONFIG.scene_name,
            "${scene_name}": MTP_TASK_CONFIG.scene_name,
            "$model_id": MTP_TASK_CONFIG.model_id,
            "${model_id}": MTP_TASK_CONFIG.model_id,
            "$date": MTP_TASK_CONFIG.date,
            "${date}": MTP_TASK_CONFIG.date,
        }
        expanded = os.path.expandvars(path)
        for key, value in mappings.items():
            expanded = expanded.replace(key, value or "")
        return os.path.normpath(expanded)

    def build(self, dataset_type_list: List, dataset_path_list: List = None):
        dataframes = []
        for index, dataset_type in enumerate(dataset_type_list):
            split_path = None
            if dataset_path_list and index < len(dataset_path_list):
                split_path = dataset_path_list[index]
            dataframes.append(self._load_dataframe(dataset_type=dataset_type, tfrecord_dir=split_path))
        return dataframes

    def _parse_example(self, ser_data: tf.Tensor):
        del ser_data
        raise NotImplementedError("FlowTFRecordDataset parses records via Example API.")

    def _get_feature_desc(self):
        return {}

    def _resolve_file_list(self, dataset_type: str = None, tfrecord_dir: Optional[str] = None) -> List[str]:
        split_name = (dataset_type or "train").lower()

        split_file_list = self.cfg.get(f"{split_name}_file_list", [])
        if split_file_list:
            return split_file_list

        if split_name in ("train",):
            file_list = self.cfg.get("file_list", [])
            if file_list:
                return file_list

        file_pattern = self.cfg.get("file_pattern", "*.tfrecord.gz")

        candidate_dirs = []
        if tfrecord_dir:
            candidate_dirs.append(self._expand_path_vars(tfrecord_dir))

        base_dir = self.cfg.get("tfrecord_dir", "") or self.cfg.get("tfecord_dir", "")
        if not base_dir:
            base_dir = self.data_path.get(split_name, "")
        base_dir = self._expand_path_vars(base_dir)

        split_dir = self.cfg.get(f"{split_name}_tfrecord_dir", "")
        split_dir = self._expand_path_vars(split_dir)

        if split_dir:
            candidate_dirs.append(split_dir)
        if base_dir:
            # Prefer explicit split subfolder under a common tfrecord root.
            candidate_dirs.append(os.path.join(base_dir, split_name))
            # Support passing train folder as tfrecord_dir and finding sibling valid/test.
            if split_name in ("valid", "test"):
                candidate_dirs.append(os.path.join(os.path.dirname(base_dir), split_name))
            candidate_dirs.append(base_dir)

        seen = set()
        dedup_dirs = []
        for d in candidate_dirs:
            if not d:
                continue
            key = os.path.normpath(d)
            if key in seen:
                continue
            seen.add(key)
            dedup_dirs.append(d)

        for d in dedup_dirs:
            files = sorted(glob.glob(os.path.join(d, file_pattern)))
            if files:
                return files

        if split_name in ("valid", "test"):
            return []

        if dedup_dirs:
            raise ValueError(
                f"No tfrecord files found for split={split_name} under {dedup_dirs} with pattern {file_pattern}"
            )
        raise ValueError("flow_generation.input.tfrecord_dir is required")

    @staticmethod
    def _to_python_value(feature: tf.train.Feature):
        kind = feature.WhichOneof("kind")
        if kind == "int64_list":
            values = list(feature.int64_list.value)
            return values[0] if len(values) == 1 else values
        if kind == "float_list":
            values = list(feature.float_list.value)
            return values[0] if len(values) == 1 else values
        if kind == "bytes_list":
            raw_values = list(feature.bytes_list.value)
            values = [v.decode("utf-8", errors="ignore") for v in raw_values]
            return values[0] if len(values) == 1 else values
        return None

    def _record_iterator(self, file_path: str, compression_type: Optional[str]):
        if compression_type:
            return tf.compat.v1.io.tf_record_iterator(file_path, options=compression_type)
        return tf.compat.v1.io.tf_record_iterator(file_path)

    def _load_dataframe(self, dataset_type: str = None, tfrecord_dir: Optional[str] = None) -> pd.DataFrame:
        compression_type = self.cfg.get("compression_type", "GZIP")
        rows: List[Dict] = []
        files = self._resolve_file_list(dataset_type=dataset_type, tfrecord_dir=tfrecord_dir)

        if not files:
            return pd.DataFrame(columns=self.schema if self.schema else None)

        for file_path in files:
            for record in self._record_iterator(file_path, compression_type):
                example = tf.train.Example()
                example.ParseFromString(record)
                item = {
                    key: self._to_python_value(val)
                    for key, val in example.features.feature.items()
                }
                rows.append(item)

        if not rows:
            raise ValueError("No records loaded from tfrecord files")

        df = pd.DataFrame(rows)
        if self.field_alias:
            df = df.rename(columns=self.field_alias)
        if self.schema:
            missing = [c for c in self.schema if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in tfrecord data: {missing}")
            return df[self.schema].copy()
        return df
