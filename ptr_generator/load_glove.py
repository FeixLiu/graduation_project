import sys
from hyperparameters import Hyperparameters as hp


class Load_glove():
    def __init__(self, path):
        self._path = path
        self._load_glove()

    def _load_glove(self):
        index = 2
        self.vocab2index = {}
        self.index2vocab = {}
        self.vocab2index['<UNK>'] = 0
        self.index2vocab[0] = '<UNK>'
        self.vocab2index['<Start>'] = 1
        self.index2vocab[1] = '<Start>'
        self.vocab2index['<End>'] = 2
        self.index2vocab[2] = '<End>'
        with open(self._path, 'r', encoding='utf-8') as file:
            for line in file:
                row = line.strip().split(' ')
                index += 1
                self.vocab2index[row[0]] = index
                self.index2vocab[index] = row[0]
