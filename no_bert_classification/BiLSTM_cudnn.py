from tensorflow.contrib import cudnn_rnn


class BiLSTM():
    def __init__(self, inputs, hidden_units, dropout):
        self._inputs = inputs
        self._hidden_units = hidden_units
        self._dropout = dropout
        self.result = self.lstm()

    def lstm(self):
        lstm_cell = cudnn_rnn.CudnnLSTM(
            num_layers=1,
            num_units=self._hidden_units,
            direction='bidirectional',
            dropout=self._dropout
        )
        outputs, (_, _) = lstm_cell(inputs=self._inputs)
        return outputs