import tensorflow as tf
from load_marco import Marco_dataset
from load_dict import Load_glove
from hyperparameters import Hyperparameters as hp
from bert import Bert_server
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def convert2indices(answer_index):
    indices = []
    for i in range(hp.batch_size):
        para = []
        for j in range(hp.max_seq_length):
            try:
                index = answer_index[i][j]
                para.append([i, j, index])
            except IndexError:
                para.append([i, j, 0])
        indices.append(para)
    # [batch_size, max_sequence_length, 3]
    # 3: [which_batch, which_words, the_word_index]
    return np.array(indices)


vocab = Load_glove(hp.word)
marco_train = Marco_dataset(hp.marco_dev_path, vocab)
bert_server = Bert_server()

with tf.device('/gpu:1'):
    # placeholder
    H = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.bert_embedding_size])
    s = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.bert_embedding_size])
    answer = tf.placeholder(dtype=tf.int32, shape=[None, hp.max_seq_length, 3])

    # attention vector
    wh = tf.Variable(tf.random_normal(shape=[hp.bert_embedding_size, hp.attention_inter_size]), dtype=tf.float32)
    ws = tf.Variable(tf.random_normal(shape=[hp.bert_embedding_size, hp.attention_inter_size]), dtype=tf.float32)
    batten = tf.Variable(tf.constant(0.1, shape=[1, hp.attention_inter_size]))
    v = tf.Variable(tf.random_normal(shape=[hp.attention_inter_size, 1]))
    hr = tf.tile(tf.expand_dims(H, 2), [1, 1, hp.max_seq_length, 1])
    # hr.shape = [batch_size, max_length, time_step, bert_embedding_size]
    sr = tf.tile(tf.expand_dims(s, 1), [1, hp.max_seq_length, 1, 1])
    # sr.shape = [batch_size, time_step, max_length, bert_embedding_size]
    hr = tf.reshape(hr, [-1, hp.bert_embedding_size])
    sr = tf.reshape(sr, [-1, hp.bert_embedding_size])
    E = tf.matmul(
            tf.nn.tanh(
                tf.add(
                    tf.add(
                        tf.matmul(hr, wh),  # [batch_size x max_length x time_step, bert_embedding_size]
                        tf.matmul(sr, ws)   # [batch_size x time_step x max_length, bert_embedding_size]
                    ),
                batten
                )
            ),
        v) # [batch_size x max_length x time_step, 1]
    E = tf.reshape(E, shape=[-1, hp.max_seq_length, hp.max_seq_length, 1])  # [batch_size, max_length, time_step, 1]
    attention = tf.reduce_sum(tf.nn.softmax(E, axis=1), axis=3)  # [batch_size, max_length, time_step]

    # context vector
    hrr = tf.tile(tf.expand_dims(H, 2), [1, 1, hp.max_seq_length, 1])
    attention_r = tf.tile(tf.expand_dims(attention, 3), [1, 1, 1, hp.bert_embedding_size])
    htstart = tf.reduce_sum(tf.math.multiply(hrr, attention_r), axis=1)

    # prediction
    prediction_pre = tf.concat([s, htstart], axis=2)
    prediction_pre = tf.reshape(prediction_pre, shape=[-1, 2 * hp.bert_embedding_size])
    prediction_w1 = tf.Variable(tf.random_normal(
        shape=[2 * hp.bert_embedding_size, hp.prediction_inter_size]),
        dtype=tf.float32
    )
    prediction_b1 = tf.Variable(tf.constant(0.1, shape=[1, hp.prediction_inter_size]), dtype=tf.float32)
    prediction_w2 = tf.Variable(tf.random_normal(
        shape=[hp.prediction_inter_size, hp.vocab_size]),
        dtype=tf.float32
    )
    prediction_b2 = tf.Variable(tf.constant(0.1, shape=[1, hp.vocab_size]), dtype=tf.float32)
    p_vocab = tf.add(tf.matmul(prediction_pre, prediction_w1), prediction_b1)
    p_vocab = tf.nn.tanh(p_vocab)
    p_vocab = tf.add(tf.matmul(p_vocab, prediction_w2), prediction_b2)
    p_vocab = tf.reshape(p_vocab, shape=[-1, hp.max_seq_length, hp.vocab_size])
    p_vocab = tf.nn.tanh(p_vocab)
    p_vocab = tf.nn.softmax(p_vocab, axis=2)

    #loss
    prob = tf.gather_nd(p_vocab, answer)
    loss = tf.reduce_sum(0 - tf.math.log(tf.clip_by_value(prob, 1e-8, 1.0)), axis=1) / hp.max_seq_length
    loss = tf.reduce_mean(loss, axis=0)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=hp.learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for _ in range(hp.epoch):
            start_index = 0
            for i in range(int(marco_train.total / hp.batch_size)):
                H_text = marco_train.paragraph[start_index:start_index + hp.batch_size]
                s_text = marco_train.answer[start_index:start_index + hp.batch_size]
                H_input = bert_server.convert2vector(H_text)
                s_input = bert_server.convert2vector(s_text)
                answer_index = marco_train.answer_index[start_index:start_index + hp.batch_size]
                start_index += hp.batch_size
                answer_index = convert2indices(answer_index)
                feedDict = {
                    H: H_input,
                    s: s_input,
                    answer: answer_index
                }
                sess.run(train_op, feed_dict=feedDict)
                print(sess.run(loss, feed_dict=feedDict))
