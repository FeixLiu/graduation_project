import numpy as np


class classification_f1():
    def __init__(self):
        self.total_para = 0.
        self.total_right_para = 0.
        self.total_pos = 0.
        self.pos_right = 0.

    def update_acc_recall(self, target, prediction):
        self.total_para += len(target)
        self.total_pos += np.sum(target)
        self.total_right_para += len(target) - np.sum(np.absolute(target - prediction))
        for i in range(len(target)):
            if target[i] == 1 and prediction[i] == 1:
                self.pos_right += 1

'''
cf = classification_f1()
for i in range(50):
    a = np.random.randint(low=0, high=2, size=10)
    b = np.random.randint(low=0, high=2, size=10)
    cf.update_acc_recall(a, b)
precision = cf.total_right_para / cf.total_para
recall = cf.pos_right / cf.total_right_para
f1 = 2 * ((precision * recall) / (precision + recall))
print(f1)
'''
