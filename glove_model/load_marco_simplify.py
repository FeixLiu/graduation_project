import json
import sys
import numpy as np
import nltk


class load_marco():
    """
    self._path (string): the path of the marco dataset
    self._vocab (dictionary): the word index dictionary
    self._max_seq_length: max sequence length of the answer, query and paragraph
    self._max_para: max paragraphs each passage has
    self.passage (list): the text of the passage,
        saved as [[para1, para2, ..., para10], ..., [para1, para2, ..., para10]]
    self.label (list): whether the paragraph can answer the query or not,
        saved as [[para1_label, para2_label, ..., para10_label], ..., [para1_label, para2_label, ..., para10_label]]
    self.answer (list): the answer for the query, saved as [answer1, answer2, ..., answerN]
    self.answer_word (list): the words level answer,
        saved as [[[word1], [word2], ..., [wordQ]], ..., [[word1], [word2], ..., [wordP]]]
    self.question (list): the question list, saved as [query1, query2, ..., queryN]
    self.passage_index (list): the word index of the passage
        saved as [[[word1, ..., word64], ..., [word1, ..., word64]], ..., [[word1, ..., word64], ..., [word1, ..., word64]]]
    self.query_index (list): the word index of the query
        saved as [[word1, ..., word64], ..., [word1, ..., word64]]
    self.answer_index (list): the word index of the answer,
        saved as [[[0, word1], [0, word2], ..., [0, wordQ], ..., [0, 0]], ..., [[0, word1], [0, word2], ..., [0, word64]]]
    self.total (int): how many query-answer-answer_index-label-passage pairs the dataset has
        equal with len(query), len(answer), len(answer_index), len(label), len(passage)
    """
    def __init__(self, path, vocab, max_seq_length, max_para):
        """
        function: initialize the class
        :param path (string): the path of the marco dataset
        :param vocab (class object): the vocab class instance
        :param max_seq_length (int): max sequence length of the answer, query and paragraph
        :param max_para (int): max paragraphs each passage has
        """
        self._path = path
        self._vocab = vocab
        self._max_seq_length = max_seq_length
        self._max_para = max_para
        self._load_marco()

    def _load_marco(self):
        """
        function: load the marco dataset
        """
        self.label = []
        self.passage_index = []
        self.query_index = []
        self.answer_index = []
        self.answer_indice = []
        with open(self._path, 'r') as file:
            data = json.load(file)
        self.total = len(data['answers'])
        for i in range(0, self.total, 40):
            i = str(i)
            query = data['query'][i]
            answer = data['answers'][i][0]
            passage = data['passages'][i]
            label_temp, para_temp = self._convert_para(passage)
            para_index_temp = []
            for i in para_temp:
                para_index = self._get_index(i)
                para_index_temp.append(para_index)
            para_index_temp = np.array(para_index_temp)
            self.passage_index.append(para_index_temp)
            self.query_index.append(self._get_index(query))
            self.label.append(label_temp)
            self.answer_index.append(self._get_index(answer, True))
            para_word = self._para_index(para_temp, label_temp)
            answer_index = self._convert2index(answer, para_word)
            self.answer_indice.append(np.array(answer_index))
        self.total = len(self.answer_index)
        self.label = np.array(self.label)
        self.answer_index = np.array(self.answer_index)
        self.answer_indice = np.array(self.answer_indice)
        self.query_index = np.array(self.query_index)
        self.passage_index = np.array(self.passage_index)
        print('Loaded MS Marco', self._path.split('/')[4].split('_')[0], 'set from:', self._path, file=sys.stderr)

    def _get_index(self, inputs, answer=False):
        """
        function: convert the inputs to its index representation
        :param inputs (string): input sentence
        :return temp(list): the index representation of inputs
        """
        words = nltk.word_tokenize(inputs)
        temp = []
        if answer:
            temp.append(0)
        for word in words:
            try:
                index = self._vocab.vocab2index[word]
            except KeyError:
                index = 0
            temp.append(index)
            if len(temp) == self._max_seq_length:
                break
        while len(temp) < self._max_seq_length:
            temp.append(0)
        return temp

    def _convert_para(self, passage):
        """
        function: convert the passage dictionary to two lists
        :param passage (dictionary): all paragraphs of the passage
        note: each passage has up to 10 paragraphs, if not enough, padding with 0
        :return label_temp (list): whether the paragraph can answer the query or not
        :return para_temp (list): the paragraph
        """
        label_temp = []
        para_temp = []
        for j in range(len(passage)):
            if j == self._max_para:
                break
            label_temp.append([float(passage[j]['is_selected'])])
            para_temp.append(passage[j]['passage_text'])
        while len(label_temp) < 10:
            label_temp.append([0.])
            temp = ''
            for _ in range(self._max_seq_length):
                temp += 'unk '
            para_temp.append(temp)
        return label_temp, para_temp

    def _convert2index(self, answer, para_word):
        """
        function: convert the answer to the word index
        :param answer (string): the answer
        :param para_word (dictionary): the word index dictionary of current passage
        :return answer_index (list): the word index of the input answer
        """
        words = nltk.word_tokenize(answer)
        answer_index = []
        id = 0
        for word in words:
            try:
                index = self._vocab.vocab2index[word]
            except KeyError:
                try:
                    index = para_word['word2index'][word]
                except KeyError:
                    index = 0
            answer_index.append([id, index])
            id += 1
            if id == 64:
                break
        while id < self._max_seq_length:
            answer_index.append([id, 0])
            id += 1
        return answer_index

    def _para_index(self, para, label):
        """
        function: get the word index dictionary of current passage
        :param para (list): all paragraphs of the passage
        :param label (list): whether the paragraph can answer the query or not
        :return para_word: the word index dictionary of current passage
        """
        index = 0
        word2index = {}
        index2word = {}
        for j in range(len(para)):
            if label[j] == 1:
                words = para[j].split(' ')
                for word in words:
                    word2index[word] = index
                    index2word[index] = word
                    index += 1
        para_word = {
            'word2index': word2index,
            'index2word': index2word
        }
        return para_word
