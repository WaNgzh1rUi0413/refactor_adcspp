import tensorflow as tf


class StrategyBuilder:
    @classmethod
    def build(cls, name: str, **params) -> tf.distribute.Strategy:
        """
        获取分布式策略
        :param name: str, 策略名
        :return: tf.distribute.Strategy, 策略实例
        """
        strategy_mapping = tf.distribute.__dict__
        if name not in strategy_mapping:
            raise ValueError(f"Unknown strategy '{name}'")
        return strategy_mapping.get(name)(**params)