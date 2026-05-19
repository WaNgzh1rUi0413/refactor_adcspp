import logging
from typing import Dict, Any, Callable
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from core.registry import ClassRegistry, FunctionRegistry
from utils.verify import validate_tensor_function

class MetricClassRegistry(ClassRegistry[Metric]):
    """Keras指标类注册器"""

    def __init__(self):
        super().__init__(base_class=Metric, registry_name="MetricClass")

        # 自动注册Keras原生指标
        self._register_keras_native_metrics()
        # 自动注册所有指标
        self.auto_register_from_module(module_path='metrics', exclude_patterns=['metric_builder', 'callback_funcs'])

    def _register_keras_native_metrics(self) -> None:
        """注册Keras原生指标"""
        import tensorflow.keras.metrics as keras_metrics

        for name, obj in inspect.getmembers(keras_metrics):
            if (
                inspect.isclass(obj)
                and issubclass(obj, Metric)
                and obj != Metric
            ):
                self.register(name, obj)
                logging.debug(msg="Registered Keras native metric: %s", args=name)

class MetricFuncRegistry(FunctionRegistry[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]):
    """Keras指标函数注册器"""

    def __init__(self):
        super().__init__(registry_name="MetricClass", validator=validate_tensor_function)

        # 自动注册Keras原生指标
        self._register_keras_native_metrics()
        # 自动注册所有指标
        self.auto_register_from_module(module_path='metrics', exclude_patterns=['metric_builder'])

    def _register_keras_native_metrics(self) -> None:
        """注册Keras原生指标"""
        import tensorflow.keras.metrics as keras_metrics

        for name, obj in inspect.getmembers(keras_metrics):
            if self._is_valid(obj):
                self.register(name, obj)
                logging.debug("Registered Keras native optimizer: %s", name)

class MetricBuilder(object):
    """
    指标构建器，用于根据配置动态创建Keras指标实例，用于训练与验证集的指标计算
    """
    @classmethod
    def build(cls, metric_configs: Dict[str, Any]) -> tf.keras.metrics.Metric:
        """
    根据配置字典构建指标

        :param
        metric_configs: 指标列表，内容为包含指标配置的字典，应包含以下键:
        -
        class: 指标类名
        - params: 指标参数
        :return:
        """
        metric_class_registry = MetricClassRegistry()
        metric_class_dict = metric_class_registry.list_all()
        metric_func_registry = MetricFuncRegistry()
        metric_func_dict = metric_func_registry.list_all()

        metric_list = []
        for metric_config in metric_configs:
            metric_type = metric_config.get('type', '')

            if metric_type in metric_class_dict:
                metric_params = metric_config.get('params', {})
                metric_class = metric_class_registry.get(metric_type)
                metric_list.append(metric_class(**metric_params))
            elif metric_type in metric_func_dict:
                metric_func = metric_func_registry.get(metric_type)
                metric_list.append(metric_func)
            else:
                raise ValueError(f"Unknown metric: {metric_type}")

    def format_metric(metric):
        """将指标对象格式化为易读字符串"""
        if hasattr(metric, 'name'):
            return metric.name
        if callable(metric):
            return metric.__name__
        return str(metric)
        logging.info(f"Built metrics: [{', '.join(map(format_metric, metric_list))}]")
        return metric_list

if __name__ == "__main__":
    metric_configs = [{'type': 'AUC'},
                      {'type': 'mse'},
                      {'type': 'LogLoss'},
                      {'type': 'PCOC'}]

    metrics = MetricBuilder.build(metric_configs)
    logging.info(metrics)

