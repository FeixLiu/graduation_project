import tensorflow as tf

class LinearRelu3d():
    """
    :param inputs: the input vector [batch size, input length, input size]
    :param input_length: the 2nd dimension of the inputs
    :param inputs_size: the 3rd dimension of the inputs
    :param outputs_size: the output size of the linear relu
    :param keepProb: whether do the dropout

    :arg relu: the tensor after relu(wx_plus_b)
    """
    def __init__(self, inputs, input_length, inputs_size, outputs_size, keepProb=None):
        self._inputs = inputs
        self._input_length = input_length
        self._inputs_size = inputs_size
        self._outputs_size = outputs_size
        self._keepProb = keepProb
        self.relu = self.LR()

    def LR(self):
        inputs = tf.reshape(self._inputs, [-1, self._inputs_size])
        weights = tf.Variable(tf.random_normal(shape=[self._inputs_size, self._outputs_size]))
        biases = tf.Variable(tf.constant(0.1, shape=[self._outputs_size]))
        wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)
        relu = tf.nn.relu(wx_plus_b)
        if self._keepProb is not None:
            relu = tf.nn.dropout(relu, self._keepProb)
        relu = tf.reshape(relu, [-1, self._input_length, self._outputs_size])
        return relu


class Lineartanh2d():
    """
    :param inputs: the input vector [batch size, input size]
    :param inputs_size: the 2rd dimension of the inputs
    :param output_size: the output size of the linear relu
    :param keepProb: whether do the dropout

    :arg relu: the tensor after relu(wx_plus_b)
    """
    def __init__(self, inputs, inputs_size, outputs_size, keepProb=None):
        self._inputs = inputs
        self._inputs_size = inputs_size
        self._outputs_size = outputs_size
        self._keepProb = keepProb
        self.relu = self.LT()

    def LT(self):
        weights = tf.Variable(tf.random_normal(shape=[self._inputs_size, self._outputs_size]))
        biases = tf.Variable(tf.constant(0.1, shape=[self._outputs_size]))
        wx_plus_b = tf.add(tf.matmul(self._inputs, weights), biases)
        relu = tf.nn.tanh(wx_plus_b)
        if self._keepProb is not None:
            relu = tf.nn.dropout(relu, self._keepProb)
        return relu