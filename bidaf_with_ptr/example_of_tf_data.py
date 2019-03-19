from load_marco import load_marco
from load_dict import load_dict
from hyperparameters import Hyperparameters as hp
import tensorflow as tf

vocab = load_dict(hp.word)
marco_train = load_marco(
    vocab=vocab,
    path=hp.marco_dev_path,
    max_seq_length=hp.max_seq_length,
    max_para=hp.max_para
)