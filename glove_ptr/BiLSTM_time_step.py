from tensorflow.contrib import cudnn_rnn

class BiLSTM_TS():
    def __init__(self, hidden_units, dropout, name):
        self._hidden_units = hidden_units
        self._dropout = dropout
        self._name = name
        self._lstm_cell = self._lstm()

    def _lstm(self):
        lstm_cell = cudnn_rnn.CudnnLSTM(
            num_layers=1,
            num_units=self._hidden_units,
            direction='bidirectional',
            dropout=self._dropout,
            name=self._name
        )
        return lstm_cell

    def get_embd(self, inputs):
        outputs, (_, _) = self._lstm_cell(inputs=inputs)
        return outputs[-1]