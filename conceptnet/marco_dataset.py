from hyper import Hyperparameters as hp
from sklearn.utils import shuffle
import json
import sys
from conceptNet import ConcetNet


class Marco_dataset():
    def __init__(self, path):
        self._path = path
        self._net = ConcetNet(hp.conceptNet_filter)
        self._load_data()

    def _load_data(self):
        self.paragraph = []
        self.query = []
        self.answer = []
        self.label = []
        with open(self._path, 'r') as file:
            data = json.load(file)
        self.total = len(data['answers'])
        for i in range(self.total):
            i = str(i)
            query = data['query'][i]
            answer = data['answers'][i][0]
            passage = data['passages'][i]
            positive, negative = self._figure_pn(passage)
            for i in positive:
                self._get_extension(i, query)
                self.paragraph.append(i)
                self.query.append(query)
                self.answer.append(answer)
                self.label.append([0., 1.])
            answer = 'Cannot answer the question from the passage.'
            for i in negative:
                #self._get_extension(i, query)
                self.paragraph.append(i)
                self.query.append(query)
                self.answer.append(answer)
                self.label.append([1., 0.])
        self.paragraph, self.query, self.answer, self.label = shuffle(self.paragraph, self.query, self.answer, self.label)
        print('Loaded MS Marco', self._path.split('/')[4].split('_')[0], 'set.', file=sys.stderr)

    def _figure_pn(self, passage):
        positive = []
        negative = []
        for i in range(len(passage)):
            if passage[i]['is_selected'] == 1:
                positive.append(passage[i]['passage_text'])
            else:
                negative.append(passage[i]['passage_text'])
        return positive, negative

    def _get_extension(self, paragraph, qeustion):
        question_extension = self._net.a_and_b_relation(paragraph, qeustion)
        if question_extension != qeustion:
            print(paragraph.encode('utf-8'))
            print(question_extension.encode('utf-8'))
        print('\n\n\n\n\n')
        return ''

Marco_dataset(hp.marco_train_path)
