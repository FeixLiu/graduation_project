import sys


class load_dict():
    """
    self._path (string): the path of the words set
    self.vocab2index (dictionary): the vocab to index dictionary
    self.index2vocab (dictionary): the index to vocab dictionary
    self.embd (list): the embeddings of the words
    """
    def __init__(self, path, embedding_size):
        """
        function: initialize the class
        :param path (string): the path of the words set
        :param embedding_size (int): the embedding size
        """
        self._path = path
        self._embedding_size = embedding_size
        self._load_dict()

    def _load_dict(self):
        """
        function: load the word index dictionary
        """
        index = 0
        self.vocab2index = {}
        self.index2vocab = {}
        self.embd = []
        self.vocab2index['unk'] = 0
        self.index2vocab[0] = 'unk'
        self.vocab2index['<start>'] = 1
        self.index2vocab[1] = '<start>'
        self.vocab2index['<end>'] = 2
        self.index2vocab[2] = '<end>'
        self.embd.append([0. for _ in range(self._embedding_size)])
        with open(self._path, 'r') as file:
            for line in file:
                row = line.strip().split(' ')
                index += 1
                self.vocab2index[row[0]] = index
                self.index2vocab[index] = row[0]
                self.embd.append(row[1:])
        print('Loaded vocabulary from:', self._path, file=sys.stderr)
