"""
this version each time only handle with only one paragraph
when concat with the classification, the sequence length of each paragraph should multiple by two
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from hyperparameters import Hyperparameters as hp
import tensorflow as tf
from load_marco import load_marco
from load_dict import load_dict
from BiDAF import BiDAF
from ptr_generator_reconstruct import PTR_Gnerator
from BiLSTM import BiLSTM
import numpy as np


def convert2word_embd(words):
    temp = []
    for i in range(len(words)):
        temp.append(words[i][1])
    while len(temp) < hp.max_seq_length:
        temp.append([0. for _ in range(hp.embedding_size)])
    return np.array(temp)


vocab = load_dict(path=hp.word, embedding_size=hp.embedding_size)
marco_dev = load_marco(
    vocab=vocab,
    path=hp.marco_train_path,
    max_seq_length=hp.max_seq_length,
    max_para=hp.max_para
)

'''
marco_train = load_marco(
    vocab=vocab, 
    path=hp.marco_train_path, 
    max_seq_length=hp.max_seq_length, 
    max_para=hp.max_para
)
'''

with tf.device('/gpu:0'):
    with tf.variable_scope('embedding'):
        embedding_weight = tf.Variable(tf.constant(0.0, shape=[hp.vocab_size, hp.embedding_size]), trainable=False)
        embedding_placeholder = tf.placeholder(tf.float32, [hp.vocab_size, hp.embedding_size])
        embedding_init = embedding_weight.assign(embedding_placeholder)
        keep_prob = tf.placeholder(tf.float32)

        context_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, hp.max_seq_length])
        qas_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, hp.max_seq_length])
        #label = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        answer_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, hp.max_seq_length])
        #answer_word_ids = tf.placeholder(dtype=tf.int32, shape=[hp.max_seq_length])
        answer_indices = tf.placeholder(dtype=tf.int32, shape=[hp.max_seq_length, 2])
        context_embedding = tf.nn.embedding_lookup(embedding_weight, context_input_ids)
        qas_embedding = tf.nn.embedding_lookup(embedding_weight, qas_input_ids)
        answer_word_embedding = tf.nn.embedding_lookup(embedding_weight, answer_input_ids)
        #answer_word_embedding = tf.nn.embedding_lookup(embedding_weight, answer_word_ids)

    with tf.variable_scope('context_lstm', reuse=tf.AUTO_REUSE):
        context_lstm = BiLSTM(
            inputs=context_embedding,
            hidden_units=hp.embedding_size,
            name='context_lstm',
            dropout=hp.keep_prob
        ).result

    with tf.variable_scope('qas_lstm', reuse=tf.AUTO_REUSE):
        qas_lstm = BiLSTM(
            inputs=qas_embedding,
            hidden_units=hp.embedding_size,
            name='qas_lstm',
            dropout=hp.keep_prob
        ).result

    with tf.variable_scope('ans_lstm', reuse=tf.AUTO_REUSE):
        ans_lstm = BiLSTM(
            inputs=answer_word_embedding,
            hidden_units=hp.embedding_size,
            name='ans_lstm',
            dropout=hp.keep_prob
        ).result
        ans_lstm = tf.reshape(ans_lstm, shape=[hp.max_seq_length, 2 * hp.embedding_size])

    with tf.variable_scope('bidaf', reuse=tf.AUTO_REUSE):
        fuse_vector = BiDAF(
            refc=context_lstm,
            refq=qas_lstm,
            cLength=hp.max_seq_length,
            qLength=hp.max_seq_length,
            hidden_units=hp.embedding_size,
            name='bidaf'
        ).fuse_vector
        fuse_vector = tf.reshape(fuse_vector, shape=[hp.max_seq_length, 8 * hp.embedding_size])

    with tf.variable_scope('ptr_variables', reuse=tf.AUTO_REUSE):
        coverage_vector_t = tf.Variable(tf.constant(0., shape=[hp.max_seq_length, 1]),
                                        dtype=tf.float32,
                                        trainable=False,
                                        name='coverage_vector_t')

        # all variables for attention
        Wh = tf.Variable(tf.random_normal(shape=[8 * hp.embedding_size, hp.attention_inter_size]),
                         dtype=tf.float32,
                         name='Wh')
        Ws = tf.Variable(tf.random_normal(shape=[2 * hp.embedding_size, hp.attention_inter_size]),
                         dtype=tf.float32,
                         name='Ws')
        wc = tf.Variable(tf.random_normal(shape=[1, hp.attention_inter_size]),
                         dtype=tf.float32,
                         name='wc')
        batten = tf.Variable(tf.constant(0.1, shape=[1, hp.attention_inter_size]),
                             dtype=tf.float32,
                             name='batten')
        V = tf.Variable(tf.random_normal(shape=[hp.attention_inter_size, 1]),
                        dtype=tf.float32,
                        name='V')

        # all variables for pvocab
        pvocab_w = tf.Variable(tf.random_normal(shape=[10 * hp.embedding_size, hp.vocab_size]),
                               dtype=tf.float32,
                               name='pvocab_w')
        pvocab_b = tf.Variable(tf.constant(0.1, shape=[1, hp.vocab_size]),
                               dtype=tf.float32,
                               name='pvocab_b')

        # all variables for pointer
        wh = tf.Variable(tf.random_normal(shape=[8 * hp.embedding_size, 1]),
                         dtype=tf.float32,
                         name='wh')
        ws = tf.Variable(tf.random_normal(shape=[2 * hp.embedding_size, 1]),
                         dtype=tf.float32,
                         name='ws')
        wx = tf.Variable(tf.random_normal(shape=[hp.embedding_size, 1]),
                         dtype=tf.float32,
                         name='wx')
        bptr = tf.Variable(tf.constant(0.1, shape=[1, 1]),
                           dtype=tf.float32,
                           name='bptr')

        # variable for calculate loss
        vocab_size = tf.Variable(tf.constant(hp.vocab_size, shape=[1]),
                                 trainable=False,
                                 name='vocab_size')

    index = 0

    answer_pre = []
    losses = []

    with tf.variable_scope('ptr_generator', reuse=tf.AUTO_REUSE):
        ptrg = PTR_Gnerator(
            bert_embedding_size=hp.embedding_size,
            max_seq_length=hp.max_seq_length,
            ptr_conv_beta=hp.ptr_conv_beta,
            Wh=Wh,
            H=fuse_vector,
            Ws=Ws,
            wc=wc,
            batten=batten,
            v=V,
            w=pvocab_w,
            b=pvocab_b,
            wh=wh,
            ws=ws,
            wx=wx,
            bptr=bptr,
            vocab_size=vocab_size
        )
        while index < hp.max_seq_length:
            st = tf.gather_nd(ans_lstm, [index])
            set = tf.expand_dims(st, axis=0)
            xt = tf.gather_nd(answer_word_embedding, [0, index])
            word_indice = tf.gather_nd(answer_indices, [index])
            at = ptrg.attention(
                st=set,
                coverage=coverage_vector_t,
            )

            coverage_vector_t = tf.add(coverage_vector_t, at)
            attention_t = tf.tile(at, [1, 8 * hp.embedding_size])
            h_star_t = tf.reduce_sum(tf.multiply(fuse_vector, attention_t), axis=0)

            pvocab = ptrg.pvocab(
                hstar_t=h_star_t,
                st=st
            )
            p_overall = tf.concat([pvocab, tf.reshape(at, shape=[-1, hp.max_seq_length])], axis=1)
            word_pre = tf.argmax(p_overall, axis=1)
            answer_pre.append(word_pre)

            pgen = ptrg.pointer(
                hstar_t=tf.expand_dims(h_star_t, axis=0),
                st=tf.expand_dims(st, axis=0),
                xt=tf.expand_dims(xt, axis=0),
            )

            LOSS = ptrg.loss(
                p_overall=p_overall,
                words_indice=tf.expand_dims(word_indice, axis=0),
                pgen=pgen,
                at=at,
                coverage_vector_t=coverage_vector_t
            )

            losses.append(LOSS)

            index += 1

    answer_pre = tf.stack(answer_pre)
    losses = tf.stack(losses)

    LOSS = tf.reduce_sum(losses, axis=0)
    loss = LOSS / hp.max_seq_length
    train_op = tf.train.GradientDescentOptimizer(hp.learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init)
        sess.run(embedding_init, feed_dict={embedding_placeholder: vocab.embd})
        for epoch in range(1):
            for i in range(10):
                passage_index = marco_dev.passage_index[i]
                label = marco_dev.label[i]
                query_index = marco_dev.query_index[i]
                query_index = query_index[np.newaxis, :]
                #answer = marco_dev.answer[i]
                answer_indice = marco_dev.answer_index[i] #(64, 2)
                answer_id = answer_indice[:, 1] #(64,)
                answer_id = answer_id[np.newaxis, :]
                # ------------------- in the later version, the lower codes will be removed ---------------------------
                passage_input = []
                for q in range(len(label)):
                    if label[q] == 1:
                        passage_input = passage_index[q]
                        break
                # ------------------- in the later version, the upper codes will be removed ---------------------------
                if len(passage_input) == 0:
                    continue
                passage_input = np.array(passage_input)
                passage_input = passage_input[np.newaxis, :]
                dict = {
                    context_input_ids: passage_input,
                    qas_input_ids: query_index,
                    answer_input_ids: answer_id,
                    answer_indices: answer_indice
                }
                #sess.run(train_op, feed_dict=dict)
                print(sess.run(answer_pre, feed_dict=dict))

