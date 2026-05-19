import inspect
import logging
from typing import List, Dict, Any, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import Loss

from core.registry import ClassRegistry


def dict_loss_wrapper(loss_fn, pred_key='pred', label_key='label'):
    """
    封装现有Keras原生损失函数，使其能够处理字典格式的输入

    :param loss_fn: 原始损失函数
    :param pred_key: 字典中对应的键名
    :param label_key: 字典中对应的键名
    """

    def wrapped_loss(y_true, y_pred):
        """处理字典输入的包装损失函数"""
        if isinstance(y_pred, dict):
            y_pred = y_pred[pred_key]
        if isinstance(y_true, dict):
            y_true = y_true[label_key]
        return loss_fn(y_true, y_pred)

    return wrapped_loss

class LossRegistry(ClassRegistry[Loss]):
    """Keras损失注册器"""

    def __init__(self):
        super().__init__(base_class=Loss, registry_name="Loss")

        # 自动注册Keras原生层
        self._register_keras_native_losses()
        # 自动注册所有模块
        self.auto_register_from_module(module_path='losses', exclude_patterns=['loss_builder'])

    def _register_keras_native_losses(self) -> None:
        """注册Keras原生损失函数"""
        import tensorflow.keras.losses as keras_losses

        for name, obj in inspect.getmembers(keras_losses):
            if (
                inspect.isclass(obj)
                and issubclass(obj, Loss)
                and obj != Loss
            ):
                self.register(name, obj)
                logging.debug(msg="Registered Keras native loss: %s", args=name)


class CustomLoss(Loss):
    def __init__(self, loss_config: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.config = loss_config

        # 初始化losses字典
        self.loss_funcs = {}

    class CustomLoss(Loss):
        def __init__(self, loss_config: Dict[str, Any], **kwargs):
            self.loss_funcs = {}

            # 从当前包(loss)获取注册器
            self.registry = LossRegistry()

            # 构建模块
            self._build_losses(self.config.get('losses', []))

            # 初始化结果字典
            self.loss_result = {}

        def __call__(self, y_true: Dict[str, tf.Tensor], y_pred: Dict[str, tf.Tensor], sample_weight=None) -> tf.Tensor:
            self.loss_result = {}

            for loss_config in self.config.get('losses', []):
                loss_name = loss_config.get('name')
                loss_func = self.loss_funcs.get(loss_name)

                inputs_config = loss_config.get('inputs', {})
                y_true_config = inputs_config.get('y_true')
                y_pred_config = inputs_config.get('y_pred')
                sample_weight_config = inputs_config.get('sample_weight')

                y_true_input = self._resolve_loss_input(y_true_config, y_true)
                y_pred_input = self._resolve_loss_input(y_pred_config, y_pred)
                sample_weight_input = None
                if sample_weight_config and sample_weight:
                    sample_weight_input = self._resolve_loss_input(sample_weight_config,
                                                                   sample_weight)  # _resolve函数中做了none空值保护
                loss_output = loss_func(y_true=y_true_input, y_pred=y_pred_input, sample_weight=sample_weight_input)
                self.loss_result[loss_name] = loss_output

            total_loss = 0.0
            for loss_config in self.config.get('losses', []):
                loss_name = loss_config.get('name')
                loss_weight = loss_config.get('loss_weight', 1.0)
                total_loss += loss_weight * self.loss_result.get(loss_name)

            return total_loss
    def get_config(self):
        super_config = super().get_config()
        super_config.update({"loss_config": self.config})
        return super_config

    def _build_losses(self, losses_config: List[Dict[str, Any]]):
        """构建损失函数"""
        has_strategy = tf.distribute.has_strategy()
        for loss_config in losses_config:
            loss_class_name = loss_config.get('type')
            loss_name = loss_config.get('name')
            loss_params = loss_config.get('params', {})

            # 根据TF官方文档，分布式训练，需要把loss reduction设置为None
            # 这样reduction会延后执行，按照多卡总batch size归一
            if 'reduction' not in loss_params and has_strategy:
                loss_params['reduction'] = 'none'

            loss_class = self.registry.get(loss_class_name)
            loss_func = loss_class(name=loss_name, **loss_params)
            self.loss_funcs[loss_name] = loss_func

        loss_func_name_dict = {}
        for loss_func_name, loss_func in self.loss_funcs.items():
            loss_func_name_dict[loss_func_name] = type(loss_func).__name__
        logging.info(msg="Built loss_funcs: %s", args=loss_func_name_dict)

    def _resolve_loss_input(self, inputs_config: Union[Dict[str, str], List[str]],
                            inputs: Union[tf.Tensor, Dict[str, tf.Tensor]]) -> \
            List[tf.Tensor]:
        """解析输入张量"""
        if isinstance(inputs_config, str) and isinstance(inputs, tf.Tensor):
            return inputs
        if inputs_config is None:
            return None
        if isinstance(inputs_config, dict):
            return self._resolve_loss_input_dict(inputs_config, inputs)
        elif isinstance(inputs_config, (list, tuple)):
            return self._resolve_loss_input_list(inputs_config, inputs)
        elif isinstance(inputs_config, str):
            return inputs[inputs_config]
        else:
            raise ValueError(f"Unsupported loss input type: {type(inputs_config)}")

    def _resolve_loss_input_dict(self, inputs_config: Dict[str, str], inputs: Dict[str, tf.Tensor]) -> List[tf.Tensor]:
        """解析输入张量dict"""
        tensors = {}
        for input_name, source in inputs_config.items():
            if source not in inputs:
                raise ValueError(f"Input '{source}' not found in current values")
            tensors[input_name] = inputs[source]
        return tensors

    def _resolve_loss_input_list(self, inputs_config: List[str], inputs: Dict[str, tf.Tensor]) -> List[tf.Tensor]:
        """解析输入张量list"""
        tensors = []
        for source in inputs_config:
            if source not in inputs:
                raise ValueError(f"Input '{source}' not found in current values")
            tensors.append(inputs[source])
        return tensors

        return tensors

class LossBuilder(object):
    @classmethod
    def build(cls, loss_config: Dict[str, Any]) -> Loss:
        reduction = 'none' if tf.distribute.has_strategy() else 'auto'
        if loss_config.get('type') == 'BinaryCrossentropy':
            return dict_loss_wrapper(keras.losses.BinaryCrossentropy(reduction=reduction))
        else:
            return CustomLoss(loss_config)


if __name__ == "__main__":
    config = {
        "name": "custom_loss",
        "losses": [
            {
                "name": "output_regularization",
                "type": "L2RegLoss",
                "loss_weight": 0.01,
                "inputs": {
                    "y_true": None,
                    "y_pred": "pred"
                }
            }
        ]
    }
    loss = LossBuilder.build(config)

    pred = tf.constant([0.5, 0.5, 0.5])
    logit = tf.constant([0.0, 0.0, 0.0])
    label = tf.constant([1.0, 0.0, 1.0])

    loss_value = loss({"label": label}, {"pred": pred, "logit": logit})
    logging.info(loss_value)
    logging.info(loss.loss_result)