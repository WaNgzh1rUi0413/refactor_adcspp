from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class Generate(ABC):
    """数据集生成基础抽象类"""

    @abstractmethod
    def generate(self) -> None:
        """生成数据集的抽象方法"""
        pass

    @abstractmethod
    def save_data(self, features: Dict[str, np.ndarray], split_name: str) -> None:
        """
        保存数据的抽象方法
        :param features:
        :param labels:
        :param split_name:
        """
        pass

    @abstractmethod
    def save_train_config_json(self) -> None:
        """保存配置文件的抽象方法"""
        pass

    @abstractmethod
    def save_featuremap_json(self) -> None:
        """保存featuremap的抽象方法"""
        pass
