import logging
import time

from tensorflow import keras


class LogLossCallback(keras.callbacks.Callback):
    def __init__(self, log_interval=10):
        """
        初始化 LogLossCallback
        :param log_interval: 每隔多少步记录一次 loss，默认为 100 步
        """
        super(LogLossCallback, self).__init__()
        self.log_interval = log_interval
        self.start_time = None

    def on_train_begin(self, logs=None):
        """
        训练开始时调用。
        """
        self.start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        """
        在每个训练批次结束时调用
        :param batch: 当前批次的索引
        :param logs: 包含当前批次的 loss 和其他指标的字典
        """
        if batch % self.log_interval == 0:
            loss = logs.get('loss')
            elapsed_time = time.time() - self.start_time
            time_per_batch = elapsed_time / (batch + 1)
            logging.info(f"Batch {batch}: Loss = {loss:.6f} Speed = {time_per_batch:.4f}s/Batch")