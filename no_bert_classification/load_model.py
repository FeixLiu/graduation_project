import tensorflow as tf
from load_marco import load_marco
from load_dict import load_dict
from hyperparameters import Hyperparameters as hp

vocab = load_dict(hp.word, hp.embedding_size)
marco_train = load_marco(
    vocab=vocab,
    path=hp.marco_train_path,
    max_seq_length=hp.max_seq_length,
    max_para=hp.max_para
)

ckpt = tf.train.get_checkpoint_state('./bidaf_classify/model')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
