from hyperparameters import Hyperparameters as hp
from bert import Bert_server
from tqdm import tqdm
import nltk
import json


class Marco_dataset():
    def __init__(self, path):
        self._path = path
        self._bert = Bert_server()
        self.load_data()

    def load_data(self):
        self.paragraph = []
        self.query = []
        self.answer = []
        self.label = []
        self.query_embd = []
        self.answer_embd = []
        self.paragraph_embd = []
        with open(self._path, 'r') as file:
            data = json.load(file)
        for i in tqdm(range(len(data['answers']))):
            i = str(i)
            query = data['query'][i]
            query_token = nltk.word_tokenize(query)
            print(query_token)
            query_embd = self._bert.convert2vector(query_token, tokenized=True, show=True)
            print(query_embd)
            """
            answer = data['answers'][i][0]
            answer_token = nltk.word_tokenize(answer)
            passage = data['passages'][i]
            positive, negative = self.figure_np(passage)
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
            """

    def figure_np(self, passage):
        positive = []
        negative = []
        for i in range(len(passage)):
            if passage[i]['is_selected'] == 1:
                positive.append(passage[i]['passage_text'])
            else:
                negative.append(passage[i]['passage_text'])
        return positive, negative

marco_train = Marco_dataset(path=hp.marco_train_path)
#marco_eval = Marco_dataset(path=hp.marco_eval_path)
#marco_dev = Marco_dataset(path=hp.marco_dev_path)
