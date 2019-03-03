from BiDAF import BiDAF
from marco_dataset import Marco_dataset
from hyperparameters import Hyperparameters as hp
from bert import Bert_server
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.device('/gpu:1'):
    '''
    with tf.variable_scope('bert_service', reuse=tf.AUTO_REUSE):
        bert = Bert_server()
    marco_train = Marco_dataset(path=hp.marco_train_path)
    marco_dev = Marco_dataset(path=hp.marco_dev_path)
    '''
    para_input = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.bert_embedding_size])
    qas_input = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.bert_embedding_size])
    answer = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.bert_embedding_size])
    keep_prob = tf.placeholder(tf.float32)

    with tf.variable_scope('biAttention', reuse=tf.AUTO_REUSE):
        fuse_vector = BiDAF(
            refc=para_input,
            refq=qas_input,
            cLength=hp.max_seq_length,
            qLength=hp.max_seq_length,
            hidden_units=hp.bert_embedding_size
        ).fuse_vector
    print(fuse_vector)