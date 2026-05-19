import tensorflow as tf


class LogLoss(tf.keras.metrics.Metric):
    """
    对数损失 (LogLoss/Binary Cross-Entropy) 指标类
    支持从logits或概率值计算，并处理多批次累积计算
    """

    def __init__(self, name='logloss', eps=1e-15, from_logits=False, **kwargs):
        super(LogLoss, self).__init__(name=name, **kwargs)
        self.eps = eps
        self.from_logits = from_logits
        self.total_loss = self.add_weight(name='total_loss', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """更新状态以反映新的预测和标签"""
        # 确保输入是张量
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 如果从logits计算，应用sigmoid函数
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        # 限制预测值范围，避免 log(0) 或 log(1)
        y_pred = tf.clip_by_value(y_pred, self.eps, 1 - self.eps)

        # 计算二元交叉熵
        term_0 = (1 - y_true) * tf.math.log(1 - y_pred)
        term_1 = y_true * tf.math.log(y_pred)
        batch_loss = -tf.reduce_mean(term_0 + term_1)

        # 更新累积损失和样本数
        self.total_loss.assign_add(batch_loss)
        self.count.assign_add(1.0)

    def result(self):
        """返回当前累积的对数损失"""
        return tf.math.divide_no_nan(self.total_loss, self.count)

    def reset_state(self):
        """重置所有状态变量"""
        self.total_loss.assign(0.0)
        self.count.assign(0.0)


class PCOC(tf.keras.metrics.Metric):
    """
    计算 PCOC (Predicted Click Over Click) 指标
    衡量模型预测的点击率与实际点击率的校准程度
    """

    def __init__(self, name='pcoc', **kwargs):
        super(PCOC, self).__init__(name=name, **kwargs)
        self.predicted_clicks = self.add_weight(name='predicted_clicks', initializer='zeros')
        self.actual_clicks = self.add_weight(name='actual_clicks', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        更新指标状态
        :param y_true: 实际点击标签 (0/1)
        :param y_pred: 预测的点击概率
        :param sample_weight: 样本权重
        :return:
        """
        # 确保输入是张量
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 计算预测的点击数总和
        batch_predicted_clicks = tf.reduce_sum(y_pred)

        # 计算实际的点击数总和
        batch_actual_clicks = tf.reduce_sum(y_true)

        # 更新累积值
        self.predicted_clicks.assign_add(batch_predicted_clicks)
        self.actual_clicks.assign_add(batch_actual_clicks)

    def result(self):
        """返回当前 PCOC 值"""
        return tf.math.divide_no_nan(self.predicted_clicks, self.actual_clicks)

    def reset_state(self):
        """重置所有状态变量"""
        self.predicted_clicks.assign(0.0)
        self.actual_clicks.assign(0.0)
