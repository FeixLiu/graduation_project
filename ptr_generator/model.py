from BiDAF import BiDAF
from marco_dataset import Marco_dataset
from hyperparameters import Hyperparameters as hp
from bert import Bert_server
import tensorflow as tf
import os
from load_glove import Load_glove
from ptr_generator import PTR_Gnerator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.device('/gpu:1'):
    '''
    with tf.variable_scope('bert_service', reuse=tf.AUTO_REUSE):
        bert = Bert_server()
    marco_train = Marco_dataset(path=hp.marco_train_path)
    marco_dev = Marco_dataset(path=hp.marco_dev_path)
    vocab = Load_glove(hp.glove_path)
    vocab_size = len(vocab.index2vocab)
    '''
    para_input = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.bert_embedding_size])
    qas_input = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.bert_embedding_size])
    answer = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.bert_embedding_size])
    current_answer = ['' for _ in range(hp.batch_size)]
    keep_prob = tf.placeholder(tf.float32)

    with tf.variable_scope('biAttention', reuse=tf.AUTO_REUSE):
        fuse_vector = BiDAF(
            refc=para_input,
            refq=qas_input,
            cLength=hp.max_seq_length,
            qLength=hp.max_seq_length,
            hidden_units=hp.bert_embedding_size
        ).fuse_vector

    converge_vector = tf.Variable(tf.zeros(shape=[hp.batch_size, hp.max_seq_length]))
    word = ['' for _ in range(hp.batch_size)]
    loss = tf.Variable(trainable=False, initial_value=tf.constant(0., dtype=tf.float32), dtype=tf.float32)
    print(loss)
    index = 0
    while index < hp.max_seq_length:
        index += 1
        ptrg = PTR_Gnerator()
        ptr = ptrg.ptr
        converge_vector += ptrg.attention
