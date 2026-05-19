import tensorflow as tf


class AdamW(tf.keras.optimizers.Optimizer):
    """
    AdamW优化器实现，继承自tf.keras.optimizers.Optimizer。
    AdamW是一种带有权重衰减的Adam优化器。

    :param learning_rate: 学习率，默认为0.001。
    :param weight_decay: 权重衰减因子，默认为0.0。
    :param beta_1: 用于计算一阶矩的指数衰减率，默认为0.9。
    :param beta_2: 用于计算二阶矩的指数衰减率，默认为0.999。
    :param epsilon: 用于数值稳定性的常数，默认为1e-7。
    :param name: 优化器名称，默认为"AdamW"。
    """

    def __init__(self, learning_rate=0.001, weight_decay=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-7, name="AdamW"):
        super().__init__(name=name)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # 初始化状态变量m和v
        self.m = []
        self.v = []
        self.updates = []

    def build(self, var_list):
        """
        初始化优化器的状态变量。

        :param var_list: 需要优化的参数列表。
        """
        # 释放旧的状态变量
        if self.m:
            for var in self.m + self.v:
                var.experimental_ref().deref()
            self.m = []
            self.v = []

        # 初始化新的状态变量
        for var in var_list:
            self.m.append(tf.Variable(tf.zeros_like(var), trainable=False))
            self.v.append(tf.Variable(tf.zeros_like(var), trainable=False))

    def get_updates(self, loss, params):
        """
        获取优化器的更新操作。

        :param loss: 损失函数。
        :param params: 需要优化的参数列表。
        :return: 更新操作列表。
        """
        if not self.m or len(self.m) != len(params):
            self.build(params)

        grads = self.get_gradients(loss, params)
        self.updates = []

        t = self.iterations
        lr = self.learning_rate
        wd = self.weight_decay
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        epsilon = self.epsilon

        for i, (param, grad) in enumerate(zip(params, grads)):
            m = self.m[i]
            v = self.v[i]

            # 更新一阶矩和二阶矩
            self.updates.append(tf.assign(m, beta_1 * m + (1 - beta_1) * grad))
            self.updates.append(tf.assign(v, beta_2 * v + (1 - beta_2) * tf.square(grad)))

            # 偏置修正
            m_hat = m / (1 - beta_1 ** (t + 1))
            v_hat = v / (1 - beta_2 ** (t + 1))

            # 计算参数更新值
            param_update = m_hat / (tf.sqrt(v_hat) + epsilon) + wd * param
            self.updates.append(tf.assign(param, param - lr * param_update))

        return self.updates

    def get_config(self):
        """
        获取优化器的配置。

        :return: 配置字典，包含学习率、权重衰减、beta_1、beta_2和epsilon。
        """
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
        }
