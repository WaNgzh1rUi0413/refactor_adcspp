from abc import ABC, abstractmethod
from typing import Dict

import tensorflow as tf
from tensorflow import keras


class AbstractModel(ABC, keras.models.Model):
    """
    继承keras Model类，详细接口参阅keras官方文档
    """

    @abstractmethod
    def build_predictable_model(self):
        """
        构建特征层和网络结构，组合成可用于预测的模型
        """
        pass

    @abstractmethod
    def build_feature_op(self):
        """
        通过工厂类构建特征层对象
        """
        pass

    @abstractmethod
    def build_network_op(self):
        """
        通过工厂类构建网络结构层对象
        """
        pass

    @abstractmethod
    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> Dict[str, tf.Tensor]:
        """
        定义计算图
        :param inputs: Dict[str, tf.Tensor], 未经载入的原始特征字典
        :param training: bool，训练状态
        :return network_out: Dict[str, tf.Tensor], 输出字典
        """
        pass

    @abstractmethod
    def build_loss_op(self):
        """
        通过工厂类构建loss对象
        """
        pass

    @abstractmethod
    def build_optimizer_op(self):
        """
        通过工厂类构建optimizer对象
        """
        pass

    def build_trainable_model(self):
        """
        构建特征层和网络结构，组合成可用于训练的模型
        """
        pass

    @abstractmethod
    def build_metric_op(self):
        """
        构建metric集合，这些metric会作为keras Model类的fit入参
        """
        pass
