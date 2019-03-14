import nltk
import json
from sklearn.utils import shuffle
import sys
import numpy as np

from hyperparameters import Hyperparameters as hp
import load_dict


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
    self.question (list): the question list, saved as [query1, query2, ..., queryN]
    self.answer_index (list): the word index of the answer,
        saved as [[word1, word2, ..., wordQ], ..., [word1, word2, workP]]
    self.total (int): how many query-answer-answer_index-label-passage pairs the dataset has
        equal with len(query), len(answer), len(answer_index), len(label), len(passage)
    """
    def __init__(self, path, vocab, max_seq_length, max_para):
        """
        function: initialize the class
        :param path (string): the path of the marco dataset
        :param vocab (dictionary): the word index dictionary
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
        self.passage = []
        self.label = []
        self.answer = []
        self.question = []
        self.answer_index = []
        with open(self._path, 'r') as file:
            data = json.load(file)
        self.total = len(data['answers'])
        for i in range(self.total):
            i = str(i)
            query = data['query'][i]
            answer = data['answers'][i][0]
            answer = self._conver_answer(answer)
            passage = data['passages'][i]
            label_temp, para_temp = self._conver_para(passage)
            self.passage.append(para_temp)
            self.label.append(label_temp)
            self.answer.append(answer)
            self.question.append(query)
            para_word = self._para_index(para_temp, label_temp)
            answer_index = self._conver2index(answer, para_word)
            self.answer_index.append(answer_index)
        self.passage, self.label, self.answer, self.question, self.answer_index = shuffle(
            self.passage,
            self.label,
            self.answer,
            self.question,
            self.answer_index
        )
        self.label = np.array(self.label)
        self.answer_index = np.array(self.answer_index)
        print('Loaded MS Marco', self._path.split('/')[4].split('_')[0], 'set from:', self._path, file=sys.stderr)

    def _conver_para(self, passage):
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
                temp += '<UNK> '
            para_temp.append(temp)
        return label_temp, para_temp

    def _conver_answer(self, answer):
        """
        function: tokenize the answer and add <Start> and <End> token and add space between word and punctuation
        :param answer (string): the answer
        :return answer (string): modified answer
        """
        answer_token = nltk.word_tokenize(answer)
        answer_token.insert(0, '<Start>')
        answer_token.append('<End>')
        answer = ''
        for j in answer_token:
            answer += j
            answer += ' '
        answer = answer.strip()
        return answer

    def _conver2index(self, answer, para_word):
        """
        function: convert the answer to the word index
        :param answer (string): the answer
        :param para_word (dictionary): the word index dictionary of current passage
        :return answer_index (list): the word index of the input answer
        """
        answer_index = []
        for word in answer.split(' '):
            try:
                index = self._vocab.vocab2index[word]
            except KeyError:
                try:
                    index = para_word['word2index'][word]
                except KeyError:
                    index = 0
            answer_index.append(index)
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
