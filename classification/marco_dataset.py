from .hyperparameters import Hyperparameters as hp
import json


class Marco_dataset():
    def __init__(self, path):
        self._path = path
        self.load_data()

    def load_data(self):
        with open(self._path, 'r') as file:
            data = json.load(file)

marco_train = Marco_dataset(path=hp.marco_train_path)
#marco_eval = Marco_dataset(path=hp.marco_eval_path)
#marco_dev = Marco_dataset(path=hp.marco_dev_path)