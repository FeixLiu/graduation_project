from nltk.translate.bleu_score import sentence_bleu


class get_blue_1():
    def __init__(self):
        self.score = 0.
        self.total = 0

    def update_bleu(self, target, inputs):
        self.score += sentence_bleu(target, inputs, weights=(1, 0, 0, 0))
        self.total += 1

'''
gb1 = get_blue_1()
a = [['he', 'is', 'a', 'boy']]
b = ['he', 'is', 'not', 'a', 'boy']
gb1.update_bleu(a, b)
print(gb1.score)
'''
