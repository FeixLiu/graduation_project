import sys


class load_dict():
    """
    self._path (string): the path of the words set
    self.vocab2index (dictionary): the vocab to index dictionary
    self.index2vocab (dictionary): the index to vocab dictionary
    """
    def __init__(self, path):
        """
        function: initialize the class
        :param path (string): the path of the words set
        """
        self._path = path
        self._load_dict()

    def _load_dict(self):
        """
        function: load the word index dictionary
        """
        index = 2
        self.vocab2index = {}
        self.index2vocab = {}
        self.vocab2index['<UNK>'] = 0
        self.index2vocab[0] = '<UNK>'
        self.vocab2index['<Start>'] = 1
        self.index2vocab[1] = '<Start>'
        self.vocab2index['<End>'] = 2
        self.index2vocab[2] = '<End>'
        with open(self._path, 'r') as file:
            for line in file:
                row = line.strip()
                index += 1
                self.vocab2index[row] = index
                self.index2vocab[index] = row
        print('Loaded vocabulary from:', self._path, file=sys.stderr)
