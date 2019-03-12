from hyperparameters import Hyperparameters as hp
from load_dict import Load_dict
import nltk
import json
from sklearn.utils import shuffle


class load_marco():
    def __init__(self, path, vocab):
        self._path = path
        self._vocab = vocab
        self._load_marco()

    def _load_marco(self):
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
            label_temp = []
            para_temp = []
            for j in range(len(passage)):
                if j == hp.max_para:
                    break
                label_temp.append(passage[j]['is_selected'])
                para_temp.append(passage[j]['passage_text'])
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
            

    def _conver_answer(self, answer):
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


vocab = Load_dict(hp.word)
load_marco(hp.marco_dev_path, vocab)
#load_marco(hp.marco_train_path, vocab)
