from hyperparameters import Hyperparameters as hp
from sklearn.utils import shuffle
import json
import sys


class Marco_dataset():
    def __init__(self, path):
        self._path = path
        self.load_data()

    def load_data(self):
        self.paragraph = []
        self.query = []
        self.answer = []
        self.label = []
        with open(self._path, 'r') as file:
            data = json.load(file)
        for i in range(len(data['answers'])):
            i = str(i)
            query = data['query'][i]
            answer = data['answers'][i][0]
            passage = data['passages'][i]
            positive, negative = self.figure_pn(passage)
            for i in positive:
                self.paragraph.append(i)
                self.query.append(query)
                self.answer.append(answer)
                self.label.append([0., 1.])
            answer = 'Cannot answer the question from the passage.'
            for i in negative:
                self.paragraph.append(i)
                self.query.append(query)
                self.answer.append(answer)
                self.label.append([1., 0.])
        self.paragraph, self.query, self.answer, self.label = shuffle(self.paragraph, self.query, self.answer, self.label)
        print('Loaded MS Marco', self._path.split('/')[4].split('_')[0], 'set.', file=sys.stderr)

    def figure_pn(self, passage):
        positive = []
        negative = []
        for i in range(len(passage)):
            if passage[i]['is_selected'] == 1:
                positive.append(passage[i]['passage_text'])
            else:
                negative.append(passage[i]['passage_text'])
        return positive, negative
