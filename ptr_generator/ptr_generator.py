import tensorflow as tf
from hyperparameters import Hyperparameters as hp


class PTR_Gnerator():
    def __init__(self, qadr, ar, converge, ab):
        self.ptr = []
        self.attention = []

