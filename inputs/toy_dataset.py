from typing import List

import numpy as np
import tensorflow as tf

from utils.easydict import EasyDict
from utils.ut_utils import gen_random_features
from .dataset import Dataset


class ToyDataset(Dataset):
    def __init__(self, cfg: EasyDict):
        self.cfg = cfg

    def build(self, dataset_type_list: List, dataset_path_list: List = None):
        data_size = 1000
        return_datasets = []
        for split in dataset_type_list:
            split_cfg = self.cfg.get(split, EasyDict)
            inputs = gen_random_features(
                2,
                data_size)
            labels = np.random.random([data_size]) > 0.5
            labels = labels.astype('float32')
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
            return_datasets.append(dataset)
        return return_datasets

    def _parse_example(self, ser_data: tf.Tensor):
        pass

    def _get_feature_desc(self):
        pass