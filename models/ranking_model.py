from typing import Dict, Union
import tensorflow as tf
from tensorflow import keras

from features.feature_builder import FeatureBuilder
from losses.loss_builder import LossBuilder
from metrics.metric_builder import MetricBuilder
from networks.network_builder import NetworkBuilder
from optimizers.optimizer_builder import OptimizerBuilder
from utils.easydict import EasyDict
from utils.logger import log_decorator
from .abstract_model import AbstractModel


class RankingModel(AbstractModel):
    """
    父类为AbstractModel和keras.Model，详细接口参阅keras官方文档
    """

    def __init__(self, cfg: EasyDict) -> None:
        """
        构造一个Model对象
        :param cfg: EasyDict，示例
            {
                "type": "RankingModel"
                "feature": {
                    "type": "ToyFeatureOp"
                },
                "network": {
                    "type": "DNN"
                },
                "loss": {
                    "type": "BinaryCrossentropy"
                }
            }
        """
        super(RankingModel, self).__init__()
        self.cfg = cfg
        self.network_op = None
        self.feature_op = None
        self.loss_op = None
        self.optimizer_op = None
        self.metric_op = None
        self.is_predictable = False
        # 使用可以构建模式，初始化时只构建feature、network层，确保模型可用于推理
        self.build_predictable_model()
        self.is_trainable = False
        self._cache_input = None
        self._cache_output = None

    @property
    def export_outputs(self):
        return self.cfg.output_names

    @log_decorator
    def build_predictable_model(self) -> None:
        """
        构建特征层和网络结构，组合成可用于预测的模型。在__init__时默认执行
        """
        self.build_feature_op()
        self.build_network_op()
        self.is_predictable = True

    @log_decorator
    def build_trainable_model(self):
        """
        构建loss、optimizer、metric，组合成可用于训练的模型。需要在创建实例后显式调用
        """
        self.build_optimizer_op()
        self.build_loss_op()
        self.build_metric_op()
        self.compile(
            optimizer=self.optimizer_op,
            loss=self.loss_op,
            metrics=self.metric_op
        )

        self.is_trainable = True

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> Dict[str, tf.Tensor]:
        """
        定义计算图
        :param inputs: dict，未经载入的原始特征字典
        :param training: bool，训练状态
        :return: network_out
        """
        self._cache_input = inputs
        features: Union[Dict[str, Union[tf.Tensor, Dict[str, tf.Tensor]]]] = self.feature_op(inputs, training=training)
        network_out: Union[tf.Tensor, Dict[str, tf.Tensor]] = self.network_op(features, training=training)

        self._cache_output = network_out
        return network_out

    @log_decorator
    def build_feature_op(self):
        """
        通过工厂类构建特征层对象
        """
        self.feature_op = FeatureBuilder.build(self.cfg.feature)

    @log_decorator
    def build_network_op(self):
        """
        通过工厂类构建网络结构层对象
        """
        self.network_op = NetworkBuilder.build(self.cfg.network)

    @log_decorator
    def build_loss_op(self):
        """
        通过工厂类构建loss对象
        """
        self.loss_op = LossBuilder.build(self.cfg.loss)

    @log_decorator
    def build_optimizer_op(self):
        """
        通过工厂类构建optimizer对象
        """
        self.optimizer_op = OptimizerBuilder.build(self.cfg.optimizer)

    @log_decorator
    def build_metric_op(self):
        """
        构建metric集合
        """
        self.metric_op = MetricBuilder.build(self.cfg.metric)

    def get_functional_model(self):
        inputs = {k: keras.Input(name=k, shape=v.shape.as_list()[1:]) for k, v in self._cache_input.items()}
        output = self(inputs, training=False)

        f_model = keras.Model(inputs=inputs, outputs=output)
        return f_model

    def train_step(self, data) -> dict:
        """
        :param data:
        :return:
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y, _ = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.loss_op(y, y_pred, sample_weight=y)
            model_loss = tf.reduce_sum(self.network_op.losses)
            loss += model_loss
            feature_loss = tf.reduce_sum(self.feature_op.losses)
            loss += feature_loss
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        # If y and y_pred are dicts, then we update the metrics for label and pred
        y = y['label'] if isinstance(y, dict) else y
        y_pred = y_pred['pred'] if isinstance(y_pred, dict) else y_pred
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        history = {m.name: m.result() for m in self.metrics}
        history['loss'] = loss
        history['feature_loss'] = feature_loss
        history['model_loss'] = model_loss
        try:
            history.update(self.loss_op.loss_result)
        except AttributeError:
            pass
        return history

    def test_step(self, data) -> dict:
        """
        定义在fit时evaluate的行为
        :param data:
        :return:
        """
        x, y, _ = data

        y_pred = self(x, training=False)
        loss = self.loss_op(y, y_pred, sample_weight=y)

        y = y['label'] if isinstance(y, dict) else y
        y_pred = y_pred['pred'] if isinstance(y_pred, dict) else y_pred
        self.compiled_metrics.update_state(y, y_pred)
        history = {m.name: m.result() for m in self.metrics}
        history['loss'] = loss
        try:
            history.update(self.loss_op.loss_result)
        except AttributeError:
            pass
        return history
