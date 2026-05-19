import glob
import os
from typing import List, Optional

import pandas as pd
import tensorflow as tf

from utils.easydict import EasyDict
from .dataset import Dataset


class FlowCSVDataset(Dataset):
    """Load flow-generation source data from CSV files."""

    def __init__(self, cfg: EasyDict):
        self.cfg = cfg
        self.schema = cfg.get("schema", [])
        self.field_alias = cfg.get("field_alias", {})
        self.encoding_list = cfg.get("encoding_list", ["utf-8", "gb18030", "gbk", "latin-1"])

    def build(self, dataset_type_list: List, dataset_path_list: List = None):
        dataframes = []
        for index, dataset_type in enumerate(dataset_type_list):
            split_path = None
            if dataset_path_list and index < len(dataset_path_list):
                split_path = dataset_path_list[index]
            dataframes.append(self._load_dataframe(dataset_type=dataset_type, csv_dir=split_path))
        return dataframes

    def _parse_example(self, ser_data: tf.Tensor):
        del ser_data
        raise NotImplementedError("FlowCSVDataset parses records via pandas.")

    def _get_feature_desc(self):
        return {}

    def _resolve_file_list(self, dataset_type: str = None, csv_dir: Optional[str] = None) -> List[str]:
        split_name = (dataset_type or "train").lower()

        split_file_list = self.cfg.get(f"{split_name}_file_list", [])
        if split_file_list:
            return split_file_list

        if split_name in ("train",):
            file_list = self.cfg.get("file_list", [])
            if file_list:
                return file_list

        file_pattern = self.cfg.get("file_pattern", "*.csv")

        # For optional eval splits (valid/test), only use explicit sources or discover sibling folders.
        # If none exists, return empty and let upper-level task fall back safely.
        if split_name in ("valid", "test"):
            split_file_list = self.cfg.get(f"{split_name}_file_list", [])
            if split_file_list:
                return split_file_list

            split_dir = self.cfg.get(f"{split_name}_csv_dir", "")
            if split_dir:
                files = sorted(glob.glob(os.path.join(split_dir, file_pattern)))
                if files:
                    return files

            base_for_split = csv_dir or self.cfg.get("csv_dir", "")
            if base_for_split:
                sibling_split = os.path.join(os.path.dirname(base_for_split), split_name)
                files = sorted(glob.glob(os.path.join(sibling_split, file_pattern)))
                if files:
                    return files
            return []

        candidate_dirs = []
        if csv_dir:
            candidate_dirs.append(csv_dir)

        base_dir = self.cfg.get("csv_dir", "")
        split_dir = self.cfg.get(f"{split_name}_csv_dir", "")

        if split_dir:
            candidate_dirs.append(split_dir)
        if base_dir:
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
                f"No csv files found for split={split_name} under {dedup_dirs} with pattern {file_pattern}"
            )
        raise ValueError("flow_generation.input.csv_dir is required")

    def _read_csv_with_fallback(self, file_path: str) -> pd.DataFrame:
        last_error = None
        for encoding in self.encoding_list:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError as err:
                last_error = err
        if last_error is not None:
            raise last_error
        return pd.read_csv(file_path)

    def _load_dataframe(self, dataset_type: str = None, csv_dir: Optional[str] = None) -> pd.DataFrame:
        files = self._resolve_file_list(dataset_type=dataset_type, csv_dir=csv_dir)

        if not files:
            return pd.DataFrame(columns=self.schema if self.schema else None)

        parts = [self._read_csv_with_fallback(f) for f in files]
        df = pd.concat(parts, axis=0, ignore_index=True)

        if self.field_alias:
            df = df.rename(columns=self.field_alias)

        if self.schema:
            missing = [c for c in self.schema if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in csv data: {missing}")
            return df[self.schema].copy()
        return df
