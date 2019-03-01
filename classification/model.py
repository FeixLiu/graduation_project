from hyperparameters import Hyperparameters as hp
import numpy as np
import tensorflow as tf
from BiDAF import BiDAF
#from marco_dataset import Marco_dataset as md
from bert import Bert_server as bs
from LinearReLU import *

'''
bert = bs()
marco_train = md(path=hp.marco_train_path)
marco_eval = md(path=hp.marco_eval_path)
marco_dev = md(path=hp.marco_dev_path)
'''

para_input = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.bert_embedding_size])
qas_input = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.bert_embedding_size])
label = tf.placeholder(dtype=tf.float32, shape=[None, hp.classes])
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope('biAttention', reuse=tf.AUTO_REUSE):
    fuse_vector = BiDAF(
        refc=para_input,
        refq=qas_input,
        cLength=hp.max_seq_length,
        qLength=hp.max_seq_length,
        hidden_units=hp.bert_embedding_size
    ).fuse_vector

with tf.variable_scope('attentionBiGRU', reuse=tf.AUTO_REUSE):
    relu_fuze_vector = LinearRelu3d(
        inputs=fuse_vector,
        input_length=hp.max_seq_length,
        inputs_size=4 * hp.bert_embedding_size,
        outputs_size=hp.bidaf_lstm_hidden_units,
        keepProb=None
    ).relu

