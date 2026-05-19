import tensorflow as tf
from tensorflow.keras.losses import Loss
class MaskedBinaryFocalLoss(Loss):
    """
    二分类 Focal Loss (负反馈)
    - y_true: [label, mask]
        label: {1, 0} (1=负反馈, 0=无反馈)
        mask : {0,1}
    - y_pred: logits, shape (B, 1) or (B,)
    """

    def __init__(self, alpha=0.95, gamma=2.0,
                 reduction='auto',
                 name="masked_binary_focal_loss"):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        label, mask = y_true

        # ---- 类型 & shape ----
        label = tf.cast(label, tf.float32)
        mask = tf.cast(mask, tf.float32)
        y_pred = tf.reshape(y_pred, (-1,))

        y = tf.cast(label, tf.float32)

        # ---- sigmoid(logits) ----
        p = tf.sigmoid(y_pred)

        # ---- focal loss ----
        pt = y * p + (1 - y) * (1 - p)
        alpha_t = y * self.alpha + (1 - y) * (1 - self.alpha)

        loss = -alpha_t * tf.pow(1 - pt, self.gamma) * tf.math.log(pt + 1e-8)

        # ---- mask ----
        loss = loss * mask
        return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)