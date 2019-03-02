import tensorflow as tf


class BiLSTM():
    def __init__(self, inputs, input_size, time_steps, hidden_units, batch_size, project=False):
        """
        :param inputs: the input tensor shape is [batch size, time steps, inputs size]
        :param input_size: the 3rd dimension of the inputs
        :param time_steps: the 2ne dimension of the inputs
        :param hidden_units: the units of the lstm
        :param batch_size: the 1st dimension of the inputs
        :param project: whether do the wx_plus_b or not

        :arg outputs: the all hidden states of the lstm
        :arg states: the final states of the lstm
        """
        self._inputs = inputs
        self._input_size = input_size
        self._time_steps = time_steps
        self._hidden_units = hidden_units
        self._batch_size = batch_size
        self._project = project
        self.outputs, self.states = self.lstm()

    def lstm(self):
        if self._project:
            weights = tf.Variable(tf.random_normal(shape=[self._input_size, self._hidden_units]))
            biases = tf.Variable(tf.constant(0.1, shape=[self._hidden_units]))
            x = tf.reshape(self._inputs, [-1, self._input_size])
            x_in = tf.add(tf.matmul(x, weights), biases)
            x_in = tf.reshape(x_in, [-1, self._time_steps, self._hidden_units])
        else:
            x_in = self._inputs
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self._hidden_units, forget_bias=1.0, state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self._hidden_units, forget_bias=1.0, state_is_tuple=True)
        initial_state_fw = cell_fw.zero_state(self._batch_size, dtype=tf.float32)
        initial_state_bw = cell_bw.zero_state(self._batch_size, dtype=tf.float32)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=x_in,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            time_major=False
        )
        return outputs, states