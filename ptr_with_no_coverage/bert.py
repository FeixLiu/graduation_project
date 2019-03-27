from bert_serving.client import BertClient


class bert_server():
    """
    self._bc: the bert service api
    """
    def __init__(self):
        """
        function: initialize the class
        """
        self._bc = BertClient(check_length=False)

    def convert2vector(self, sentence, show=False):
        """
        function: convert the input sentences to the embedding with bert
        :param sentence (list): the sentences need to convert
        :param show (bool): whether return the tokens from bert
        :return: the embedding of the input
        """
        return self._bc.encode(sentence, show_tokens=show)