from hyperparameters import Hyperparameters as hp
from sklearn.utils import shuffle
import json
import sys
import nltk
from tqdm import tqdm
from load_glove import Load_glove


class Marco_dataset():
    def __init__(self, path, vocab):
        self._path = path
        self._vocab = vocab
        self._load_data()

    def _load_data(self):
        self.paragraph = []
        self._paragraph_word = []
        self.query = []
        self.answer = []
        self.label = []
        self.answer_index = []
        with open(self._path, 'r') as file:
            data = json.load(file)
        self.total = len(data['answers'])
        index = 0
        for i in range(self.total):
            i = str(i)
            query = data['query'][i]
            answer = data['answers'][i][0]
            answer_token = nltk.word_tokenize(answer)
            answer_token.insert(0, '<Start>')
            answer_token.append('<End>')
            answer = ''
            for j in answer_token:
                answer += j
                answer += ' '
            answer = answer.strip()
            passage = data['passages'][i]
            positive, negative = self._figure_pn(passage)
            for j in positive:
                index += 1
                self._paragraph_word = self._para_index(j)
                answer_index = self._convert2index(answer_token)
                print(answer_index)
                self.paragraph.append(j)
                self.query.append(query)
                self.answer.append(answer)
                self.answer_index.append(answer_index)
                self.label.append([0., 1.])
            '''
            answer = 'No Answer Present.'
            answer_token = nltk.word_tokenize(answer)
            answer_token.insert(0, '<Start>')
            answer_token.append('<End>')
            answer_index = self._convert2index(answer_token)
            answer = ''
            for j in answer_token:
                answer += j
                answer += ' '
            answer = answer.strip()
            for i in negative:
                self.paragraph.append(i)
                self.query.append(query)
                self.answer.append(answer)
                self.answer_index.append(answer_index)
                self.label.append([1., 0.])
            '''
        self.total = index
        self.paragraph, self.query, self.answer, self.label = shuffle(self.paragraph, self.query, self.answer, self.label)
        print('Loaded MS Marco', self._path.split('/')[4].split('_')[0], 'set.', file=sys.stderr)

    def _para_index(self, j):
        words = j.split(' ')
        index = 0
        word2index = {}
        index2word = {}
        for word in words:
            word2index[word] = index
            index2word[index] = word
            index += 1
        para = {
            'word2index': word2index,
            'index2word': index2word
        }
        return para

    def _figure_pn(self, passage):
        positive = []
        negative = []
        for i in range(len(passage)):
            if passage[i]['is_selected'] == 1:
                positive.append(passage[i]['passage_text'])
            else:
                negative.append(passage[i]['passage_text'])
        return positive, negative

    def _convert2index(self, answer):
        rst = []
        for i in answer:
            try:
                index = self._vocab.vocab2index[i]
            except KeyError:
                try:
                    index = self._paragraph_word['word2index'][i] + hp.vocab_size
                except KeyError:
                    index = 0
            rst.append(index)
        return rst
