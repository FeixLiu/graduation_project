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
    vocab = Load_glove(hp.word)
    marco_dev = Marco_dataset(path=hp.marco_dev_path, vocab=vocab)
    #marco_train = Marco_dataset(path=hp.marco_train_path)
    '''
    para_input = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.bert_embedding_size])
    qas_input = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.bert_embedding_size])
    answer_embd = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.bert_embedding_size])
    answer_word_embd = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.bert_embedding_size])
    answer_indices = tf.placeholder(dtype=tf.int32, shape=[None, hp.max_seq_length, 3])
    keep_prob = tf.placeholder(tf.float32)

    with tf.variable_scope('biAttention', reuse=tf.AUTO_REUSE):
        fuse_vector = BiDAF(
            refc=para_input,
            refq=qas_input,
            cLength=hp.max_seq_length,
            qLength=hp.max_seq_length,
            hidden_units=hp.bert_embedding_size
        ).fuse_vector

    coverage_vector_t = tf.Variable(tf.zeros(shape=[hp.batch_size, hp.max_seq_length, 1]))
    index = 0

    # all variables for attention
    Wh = tf.Variable(tf.random_normal(shape=[4 * hp.bert_embedding_size, hp.attention_inter_size]), dtype=tf.float32)
    Ws = tf.Variable(tf.random_normal(shape=[hp.bert_embedding_size, hp.attention_inter_size]), dtype=tf.float32)
    wc = tf.Variable(tf.random_normal(shape=[1, hp.attention_inter_size]), dtype=tf.float32)
    batten = tf.Variable(tf.constant(0.1, shape=[1, hp.attention_inter_size]), dtype=tf.float32)
    V = tf.Variable(tf.random_normal(shape=[hp.attention_inter_size, 1]), dtype=tf.float32)

    # all variables for pvocab
    pvocab_w = tf.Variable(tf.random_normal(shape=[5 * hp.bert_embedding_size, hp.vocab_size]), dtype=tf.float32)
    pvocab_b = tf.Variable(tf.constant(0.1, shape=[1, hp.vocab_size]), dtype=tf.float32)

    # all variables for pointer
    wh = tf.Variable(tf.random_normal(shape=[4 * hp.bert_embedding_size, 1]), dtype=tf.float32)
    ws = tf.Variable(tf.random_normal(shape=[hp.bert_embedding_size, 1]), dtype=tf.float32)
    wx = tf.Variable(tf.random_normal(shape=[hp.bert_embedding_size, 1]), dtype=tf.float32)
    bptr = tf.Variable(tf.constant(0.1, shape=[1, 1]))

    with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        while index < hp.max_seq_length:
            index += 1
            indices = []
            for j in range(hp.batch_size):
                indices.append([j, index])
            st = tf.gather_nd(answer_embd, indices)
            set = tf.tile(tf.expand_dims(st, 1), [1, hp.max_seq_length, 1])
            xt = tf.gather_nd(answer_word_embd, indices)
            word_indece = tf.gather_nd(answer_indices, indices)
            ptrg = PTR_Gnerator()
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
            attention_t = tf.tile(at, [1, 1, 4 * hp.bert_embedding_size])
            hstar_t = tf.reduce_sum(tf.math.multiply(fuse_vector, attention_t), axis=1)
            pvocab = ptrg.pvocab(
                st=st,
                hstar_t=hstar_t,
                w=pvocab_w,
                b=pvocab_b
            )
            pgen = ptrg.pointer(
                wh=wh,
                hstar_t=hstar_t,
                ws=ws,
                st=st,
                wx=wx,
                xt=xt,
                bptr=bptr
            )

