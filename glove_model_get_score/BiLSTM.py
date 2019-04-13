import tensorflow as tf


class BiLSTM():
    def __init__(self, inputs, time_steps, hidden_units, batch_size, name):
        self._inputs = inputs
        self._time_steps = time_steps
        self._hidden_units = hidden_units
        self._batch_size = batch_size
        self._name = name
        self._outputs, self._states = self.lstm()
        self.result = tf.concat([self._outputs[0], self._outputs[1]], axis=2)

    def lstm(self):
        x_in = self._inputs
        cell_fw = tf.nn.rnn_cell.LSTMCell(
            num_units=self._hidden_units,
            forget_bias=1.0,
            state_is_tuple=True,
            name=self._name + '_fw')
        cell_bw = tf.nn.rnn_cell.LSTMCell(
            num_units=self._hidden_units,
            forget_bias=1.0,
            state_is_tuple=True,
            name=self._name + '_bw')
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