from bert_serving.client import BertClient


class Bert_server():
    def __init__(self):
        self._bc = BertClient()

    def convert2vector(self, sentence):
        return self._bc.encode(sentence)