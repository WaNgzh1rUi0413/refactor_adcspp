import inspect
import logging
from typing import Dict, Any

import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from core.registry import ClassRegistry


class OptimizerRegistry(ClassRegistry[Optimizer]):
    """Keras优化器注册器"""

    def __init__(self):
        super().__init__(base_class=Optimizer, registry_name="Optimizer")

        # 自动注册Keras原生优化器
        self._register_keras_native_optimizers()
        # 自动注册所有优化器
        self.auto_register_from_module(module_path='optimizers', exclude_patterns=['optimizer_builder'])

    def _register_keras_native_optimizers(self) -> None:
        """注册Keras原生优化器"""
        import tensorflow.keras.optimizers as keras_optimizers

        for name, obj in inspect.getmembers(keras_optimizers):
            if (
                inspect.isclass(obj)
                and issubclass(obj, Optimizer)
                and obj != Optimizer
            ):
                self.register(name, obj)
                logging.debug("Registered Keras native optimizer: %s", name)


class ScheduleRegistry(ClassRegistry[LearningRateSchedule]):
    """Keras学习率调度器注册器"""

    def __init__(self):
        super().__init__(base_class=LearningRateSchedule, registry_name="LearningRateSchedule")

        # 自动注册Keras原生学习率调度器
        self._register_keras_native_schedules()
        # 自动注册所有学习率调度器
        self.auto_register_from_module(module_path='optimizers', exclude_patterns=['optimizer_builder'])

    def _register_keras_native_schedules(self) -> None:
        """注册Keras原生学习率调度器"""
        import tensorflow.keras.optimizers.schedules as keras_schedules

        for name, obj in inspect.getmembers(keras_schedules):
            if (
                inspect.isclass(obj)
                and issubclass(obj, LearningRateSchedule)
                and obj != LearningRateSchedule
            ):
                self.register(name, obj)
                logging.debug("Registered Keras native schedule: %s", name)


class OptimizerBuilder(object):
    """
    优化器构建器，用于根据配置动态创建Keras优化器实例
    """

    @classmethod
    def build(cls, optimizer_config: Dict[str, Any]) -> tf.keras.optimizers.Optimizer:
        """
        根据配置字典构建优化器

        :param optimizer_config: 包含优化器配置的字典，应包含以下键:
                - class: 优化器类名
                - params: 优化器参数
        :return:
        """
        optimizer_registry = OptimizerRegistry()
        optimizer_class_name = optimizer_config.get('type', 'Adam')
        optimizer_params = optimizer_config.get('params', {})

        lr_scheduler = optimizer_params.pop('lr_scheduler', None)
        if lr_scheduler:
            schedule_registry = ScheduleRegistry()
            schedule_class_name = lr_scheduler.get('type', 'PiecewiseConstantDecay')
            schedule_params = lr_scheduler.get('params', {})
            schedule_class = schedule_registry.get(schedule_class_name)
            schedule = schedule_class(**schedule_params)
            if optimizer_params.get('learning_rate', None):
                logging.info("Built learning rate schedule, overwrited learning rate: %s", schedule)
            optimizer_params['learning_rate'] = schedule

        optimizer_class = optimizer_registry.get(optimizer_class_name)
        optimizer = optimizer_class(**optimizer_params)

        logging.info("Built optimizer: %s", optimizer.get_config())
        return optimizer


if __name__ == '__main__':
    # 构建Adam优化器
    adam_config = {
        'type': 'Adam',
        'params': {
            'name': 'adam_optimizer',
            'learning_rate': 0.001,
            'beta_1': 0.9,
            'beta_2': 0.999
        }
    }
    adam_optimizer = OptimizerBuilder.build(adam_config)
    logging.info("Adam优化器配置: %s", adam_optimizer.get_config())
