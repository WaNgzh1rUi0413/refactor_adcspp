import gzip
import json
import logging
import os
import random
import string
from typing import Dict, List, Union

import numpy as np
import tensorflow as tf

from utils.config.constants import ACTUAL_CROSS_SEP
from utils.config.constants import CONFIG_CROSS_SEP
from utils.config.constants import DEFAULT_RANDOML_FACTOR
from utils.config.constants import DPP_FEATURE_MAP_CONFIG
from utils.config.constants import DPP_MODEL_TRAIN_CONFIG
from utils.config.constants import GENERATED_STR_LENGTH
from utils.config.constants import LOCAL_OUTPUT_DIR
from utils.config.constants import MAX_INDEX_LIMIT
from utils.config.constants import VOCAB_SIZE
from utils.easydict import EasyDict
from .generator import Generate


class TFRecordGenerate(Generate):
    """生成TFRecord格式数据集的实现类"""

    def __init__(self, input_conf: EasyDict, data_size: int = 1000, files_per_split: int = 5, date: str = ''):
        super().__init__()
        self.input_column_conf = input_conf.input_column
        self.data_size = data_size
        self.files_per_split = files_per_split
        self.feature_config = self._parse_input_config()
        self.data_dir = os.path.join(LOCAL_OUTPUT_DIR, "data", date)
        self.train_config_path = os.path.join(LOCAL_OUTPUT_DIR, "config", date, DPP_MODEL_TRAIN_CONFIG)
        self.featuremap_path = os.path.join(LOCAL_OUTPUT_DIR, "config", date, DPP_FEATURE_MAP_CONFIG)

    def _parse_input_config(self):
        """解析输入的特征配置情况"""
        parsed_config = {}
        self.label_names = []
        for feature_type, feature_conf in self.input_column_conf.items():
            for feature in feature_conf:
                if isinstance(feature, str):
                    feature_setting = self._default_feature_setting(feature_type)
                    name = feature
                elif isinstance(feature, dict):
                    feature_setting = feature
                    name = feature_setting.get("name")
                else:
                    raise ValueError("invalid feature ...")

                if feature_type == 'labels':
                    self.label_names.append(name)

                parsed_config[name] = {
                    "data_type": feature_setting.get("data_type", self._default_data_type(feature_type)),
                    "seqMaxLength": feature_setting.get("seqMaxLength", 1),
                    "maxIndex": feature_setting.get("maxIndex", 100),
                    "joint_name": feature_setting.get("joint_name"),
                    "cross_name": name.replace(CONFIG_CROSS_SEP, ACTUAL_CROSS_SEP)
                }
        return parsed_config

    @staticmethod
    def _default_feature_setting(feature_type: str) -> Dict:
        if feature_type == "sparse":
            return {"data_type": "int64", "seqMaxLength": 1, "maxIndex": 100}
        elif feature_type == "dense" or feature_type == "labels":
            return {"data_type": "float32", "seqMaxLength": 1}
        elif feature_type == "sparse_seq":
            return {"data_type": "int64", "seqMaxLength": 20, "maxIndex": 100}
        return {}

    @staticmethod
    def _default_data_type(data_type: str) -> str:
        if data_type == "sparse" or data_type == "sparse_seq":
            return "int64"
        elif data_type == "dense" or data_type == "labels":
            return "float32"
        return ""

    @staticmethod
    def _float_feature(value: Union[float, List[float]]) -> tf.train.Feature:
        """从value返回float的list
        :param value:
        :return:
        """
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _int64_feature(value: Union[int, List[int]]) -> tf.train.Feature:
        """从value返回int64的list
        :param value:
        :return:
        """
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _bytes_feature(value: Union[str, bytes, List[str], List[bytes]]) -> tf.train.Feature:
        """
        将字符串或字节列表转换为TFRecord的bytes_feature
        :param value: 字符串、字节或它们的列表
        :return: tf.train.Feature 对象
        """
        if isinstance(value, str):
            value = value.encode('utf-8')
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        if isinstance(value, bytes):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            value = [v.encode('utf-8') for v in value]
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        if isinstance(value, list) and all(isinstance(v, bytes) for v in value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        raise ValueError(f"Unsupported type: {type(value)}")

    def serialize_example(self, feature_dict: Dict[str, np.ndarray]) -> bytes:
        """创建TFRecord格式的训练样本
        :param feature_dict:
        :return:
        """
        features = {}
        for feature_name, values in feature_dict.items():
            metadata = self.feature_config.get(feature_name, {})
            feature_type = metadata.get('data_type', 'int64')

            if feature_type == 'float32':
                if values.dtype != np.float32:
                    values = values.astype(np.float32)
                features[metadata.get("cross_name")] = self._float_feature(values.tolist())
            elif feature_type == 'int64':
                if values.dtype != np.int64:
                    values = values.astype(np.int64)
                features[metadata.get("cross_name")] = self._int64_feature(values.tolist())
            elif feature_type == 'string':
                if isinstance(values, np.ndarray):
                    if values.dtype.kind == 'U':
                        values = [v.encode('utf-8') for v in values.tolist()]
                    elif values.dtype.kind == 'S':
                        values = values.tolist()
                    else:
                        raise ValueError(f"Unsupported string dtype: {values.dtype}")
                elif isinstance(values, str):
                    values = [values.encode('utf-8')]
                elif isinstance(values, bytes):
                    values = [values]
                else:
                    raise ValueError(f"Unsupported string type: {type(values)}")

                features[metadata.get("cross_name")] = self._bytes_feature(values)
            else:
                raise ValueError(f"Unknown feature_type '{feature_type}'")

        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return example_proto.SerializeToString()

    def _generate_data(self, data_size: int) -> Dict[str, np.ndarray]:
        """生成特征和标签模拟数据
        :param data_size:
        :return:
        """
        features = {}
        for feature_name, metadata in self.feature_config.items():
            feature_type = metadata.get('data_type', 'int64')
            seq_length = metadata.get('seqMaxLength', 1)
            max_index = metadata.get('maxIndex', 100)
            if max_index > MAX_INDEX_LIMIT:
                raise ValueError("the given max index is too large...")
            if feature_name not in self.label_names + ['column_id']:
                if feature_type == 'float32':
                    features[feature_name] = np.random.rand(data_size, seq_length).astype(np.float32)
                elif feature_type == 'int64':
                    features[feature_name] = np.random.randint(
                        low=0,
                        high=max_index + 1,
                        size=(data_size, seq_length),
                        dtype=np.int64
                    )
                elif feature_type == 'string':
                    chars = string.ascii_lowercase
                    vocab_size_new = min(max_index + 1, VOCAB_SIZE)
                    vocab = [''.join(random.choices(chars, k=GENERATED_STR_LENGTH)) for _ in range(vocab_size_new)]
                    strings = np.random.choice(vocab, size=(data_size, seq_length))
                    features[feature_name] = strings.astype('U')
                else:
                    raise ValueError(f"Unknown feature_type '{feature_type}'")
            elif feature_name in self.label_names:
                values = np.random.randint(0, 2, size=data_size, dtype=np.int64)
                if metadata.get('enable_empty') and np.random.random() < DEFAULT_RANDOML_FACTOR:
                    values = -1
                if feature_type == 'float32':
                    values = values.astype(np.float32)
                features[feature_name] = values
            elif feature_name == 'column_id':
                chars = string.ascii_lowercase
                vocab = [''.join(random.choices(chars, k=GENERATED_STR_LENGTH)) for _ in range(VOCAB_SIZE)]
                aid = np.random.choice(vocab, size=data_size)
                task_id = np.random.choice(vocab, size=data_size)
                logid = np.random.choice(vocab, size=data_size)
                label = [''.join(random.choices('12', k=1)) for _ in range(data_size)]
                features[feature_name] = np.array([
                    f'{p1}#{p2}#{p3}#{p4}' for p1, p2, p3, p4 in zip(aid, task_id, logid, label)
                ], dtype='U')

        return features

    def save_data(self, features: Dict[str, np.ndarray], split_name: str) -> None:
        """保存数据为多个TFRecord文件
        :param features:
        :param split_name:
        """
        split_dir = os.path.join(self.data_dir, 'valid' if split_name == 'val' else split_name)
        os.makedirs(split_dir, exist_ok=True)

        samples_per_file = self.data_size // self.files_per_split
        for file_idx in range(self.files_per_split):
            start_idx = file_idx * samples_per_file
            end_idx = start_idx + samples_per_file if file_idx < self.files_per_split - 1 else self.data_size

            tfrecord_path = os.path.join(split_dir, f'{split_name}_{file_idx}.tfrecord.gz')
            tfrecord_temp_path = os.path.join(split_dir, f'{split_name}_{file_idx}_temp.tfrecord')
            with tf.io.TFRecordWriter(tfrecord_path) as writer:
                for i in range(start_idx, end_idx):
                    feature_dict = {name: values[i] for name, values in features.items()}
                    example = self.serialize_example(feature_dict)
                    writer.write(example)

            try:
                with tf.io.TFRecordWriter(tfrecord_temp_path) as writer:
                    for i in range(start_idx, end_idx):
                        feature_dict = {name: values[i] for name, values in features.items()}
                        example = self.serialize_example(feature_dict)
                        writer.write(example)

                with open(tfrecord_temp_path, 'rb') as f_in, gzip.open(tfrecord_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            finally:
                if os.path.exists(tfrecord_temp_path):
                    os.remove(tfrecord_temp_path)

    def save_train_config_json(self) -> None:
        """保存配置文件，格式为(seqMaxLength:{}, maxIndex:{})"""
        os.makedirs(os.path.dirname(self.train_config_path), exist_ok=True)

        config_data = {
            "seqMaxLength": {},
            "maxIndex": {}
        }

        for feature_name, metadata in self.feature_config.items():
            if feature_name not in self.label_names:
                config_data["seqMaxLength"][metadata.get("cross_name")] = metadata.get("seqMaxLength", 1)
                config_data["maxIndex"][metadata.get("joint_name") or metadata.get("cross_name")] = metadata.get(
                    "maxIndex",
                    100)

        with open(self.train_config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    def save_featuremap_json(self) -> None:
        """保存配置文件，格式为(seqMaxLength:{}, maxIndex:{})"""
        os.makedirs(os.path.dirname(self.featuremap_path), exist_ok=True)

        featuremap = {}

        for _, metadata in self.feature_config.items():
            cross_name = metadata.get("cross_name")
            if metadata.get("data_type") == "int64":
                max_index = metadata.get("maxIndex", 0)
                featuremap['_'.join([cross_name, "missing"])] = 0
                for index in range(max_index + 1):
                    featuremap['_'.join([cross_name, str(index)])] = index + 1

        with open(self.featuremap_path, "w") as f:
            json.dump(featuremap, f, indent=2)

    def generate(self) -> None:
        """生成并保存TFRecord格式的数据集"""
        os.makedirs(self.data_dir, exist_ok=True)

        dataset_sizes = {
            'train': self.data_size,
            'valid': self.data_size,
            'test': self.data_size
        }

        for split_name, data_size in dataset_sizes.items():
            features = self._generate_data(data_size)
            self.save_data(features, split_name)
        logging.info(f'TFRecord datasets have been saved to: {self.data_dir}')
