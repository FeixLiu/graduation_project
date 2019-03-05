import sys


class Load_glove():
    def __init__(self, path):
        self._path = path
        self._load_glove()

    def _load_glove(self):
        index = 0
        self.vocab2index = {}
        self.index2vocab = {}
        self.vocab2index['unk'] = 0
        self.index2vocab[0] = 'unk'
        with open(self._path, 'r') as file:
            for line in file:
                row = line.strip().split(' ')
                index += 1
                self.vocab2index[row[0]] = index
                self.index2vocab[index] = row[0]
        print('Loaded vocabulary dictionary.', file=sys.stderr)
