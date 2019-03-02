import tensorflow as tf

class class_balanced_cross_entropy():
    """
    :param labels: the target
    :param logtis: the log probability
    :param beta: the class balance parameter

    :arg loss: the loss
    """
    def __init__(self, labels, logtis, beta):
        self._labels = labels
        self._logtis = logtis
        self._beta = beta
        self.loss = self.get_loss()

    def get_loss(self):
        prediction_softmax = tf.nn.softmax(self._logtis)
        log_prediction = tf.math.log(prediction_softmax)
        class_balance = tf.Variable(tf.constant([self._beta, -1], dtype=tf.float32))
        loss = tf.reduce_sum(tf.multiply(tf.multiply(log_prediction, self._labels), class_balance), axis=1)
        return loss