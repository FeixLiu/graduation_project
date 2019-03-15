"""
this version each time only handle with only one paragraph
when concat with the classification, the sequence length of each paragraph should multiple by two
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from hyperparameters import Hyperparameters as hp
import tensorflow as tf
from load_marco import load_marco
from load_dict import load_dict
from BiDAF import BiDAF
from bert import bert_server
from ptr_generator import PTR_Gnerator

'''
vocab = load_dict(path=hp.word)
marco_dev = load_marco(
    vocab=vocab,
    path=hp.marco_dev_path,
    max_seq_length=hp.max_seq_length,
    max_para=hp.max_para
)
'''
'''
marco_train = load_marco(
    vocab=vocab, 
    path=hp.marco_train_path, 
    max_seq_length=hp.max_seq_length, 
    max_para=hp.max_para
)
'''

with tf.device('/cpu'):
    # batch size is one
    passage = tf.placeholder(shape=[None, hp.max_seq_length, hp.bert_embedding_size], dtype=tf.float32)
    query = tf.placeholder(shape=[None, hp.max_seq_length, hp.bert_embedding_size], dtype=tf.float32)
    answer_embd = tf.placeholder(dtype=tf.float32, shape=[hp.max_seq_length, hp.bert_embedding_size])
    answer_word_embd = tf.placeholder(dtype=tf.float32, shape=[hp.max_seq_length, hp.bert_embedding_size])
    answer_indices = tf.placeholder(dtype=tf.int32, shape=[hp.max_seq_length, 2])
    keep_prob = tf.placeholder(tf.float32)

    fuse_vector = BiDAF(
        refc=passage,
        refq=query,
        cLength=hp.max_seq_length,
        qLength=hp.max_seq_length,
        hidden_units=hp.bert_embedding_size
    ).fuse_vector
    fuse_vector = tf.reshape(fuse_vector, shape=[hp.max_seq_length, 4 * hp.bert_embedding_size])

    ptrg = PTR_Gnerator(
        bert_embedding_size=hp.bert_embedding_size,
        max_seq_length=hp.max_seq_length,
        ptr_conv_beta=hp.ptr_conv_beta
    )

    coverage_vector_t = tf.Variable(tf.zeros(shape=[hp.max_seq_length, 1]))

    # all variables for attention
    Wh = tf.Variable(tf.random_normal(shape=[4 * hp.bert_embedding_size, hp.attention_inter_size]), dtype=tf.float32)
    Ws = tf.Variable(tf.random_normal(shape=[hp.bert_embedding_size, hp.attention_inter_size]), dtype=tf.float32)
    wc = tf.Variable(tf.random_normal(shape=[1, hp.attention_inter_size]), dtype=tf.float32)
    batten = tf.Variable(tf.constant(0.1, shape=[1, hp.attention_inter_size]), dtype=tf.float32)
    V = tf.Variable(tf.random_normal(shape=[hp.attention_inter_size, 1]), dtype=tf.float32)

    with tf.variable_scope('ptr_generator', reuse=tf.AUTO_REUSE):
        for index in range(hp.max_seq_length):
            indices = [index]
            st = tf.gather_nd(answer_embd, indices)
            set = tf.expand_dims(st, 0)
            xt = tf.gather_nd(answer_word_embd, indices)
            at = ptrg.attention(
                Wh=Wh,
                H=fuse_vector,
                Ws=Ws,
                st=set,
                wc=wc,
                coverage=coverage_vector_t,
                batten=batten,
                v=V
            )
            coverage_vector_t = tf.add(coverage_vector_t, at)
            attention_t = tf.tile(at, [1, 4 * hp.bert_embedding_size])
            h_star_t = tf.reduce_sum(tf.multiply(fuse_vector, attention_t), axis=0)