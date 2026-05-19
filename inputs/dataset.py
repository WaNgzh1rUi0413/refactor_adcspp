import abc
from typing import List
import tensorflow as tf

class Dataset(abc.ABC):
    @abc.abstractmethod
    def build(self, dataset_type_list: List, dataset_path_list: List = None):
        pass

    @abc.abstractmethod
    def _parse_example(self, ser_data: tf.Tensor):
        pass

    @abc.abstractmethod
    def _get_feature_desc(self):
        pass