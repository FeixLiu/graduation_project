import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from hyperparameters import Hyperparameters as hp
import tensorflow as tf
from BiDAF import BiDAF
from classification_vector import classification
from bert import bert_server
from BiLSTM import BiLSTM
import numpy as np
from extract_valid_para import extract_valid

with tf.device('/gpu:1'):
    passage = tf.placeholder(shape=[None, hp.max_seq_length, hp.bert_embedding_size], dtype=tf.float32)
    query = tf.placeholder(shape=[None, hp.max_seq_length, hp.bert_embedding_size], dtype=tf.float32)
    label = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    keep_prob = tf.placeholder(dtype=tf.float32)

    bert = bert_server()

    fuse_vector = BiDAF(
        refc=passage,
        refq=query,
        cLength=hp.max_seq_length,
        qLength=hp.max_seq_length,
        hidden_units=hp.bert_embedding_size
    ).fuse_vector

    features = BiLSTM(
        inputs=fuse_vector,
        time_steps=hp.max_seq_length,
        hidden_units=4 * hp.bert_embedding_size,
        batch_size=hp.max_para,
    ).result

    classification_vector = classification(
        inputs=features,
        embedding_size=8 * hp.bert_embedding_size,
        max_seq_length=hp.max_seq_length,
        bert_embedding_size=hp.bert_embedding_size,
        keep_prob=keep_prob
    ).class_vector

    with tf.name_scope('class_loss'):
        class_loss = tf.reduce_sum(
            tf.reduce_sum(
                tf.nn.weighted_cross_entropy_with_logits(
                    targets=label,
                    logits=classification_vector,
                    pos_weight=hp.pos_weight),
                axis=0),
            axis=0
        )
        class_loss_summary = tf.summary.scalar("class_loss", class_loss)

    with tf.name_scope('class_acc'):
        prediction = tf.cast(tf.greater(tf.nn.sigmoid(classification_vector), 0.5), tf.float32)
        class_acc = tf.reduce_sum(
            tf.divide(
                tf.subtract(
                    10.,
                    tf.reduce_sum(
                        tf.mod(
                            tf.add(
                                label,
                                prediction),
                            2),
                        axis=0)),
                10),
            axis=0
        )
        class_acc_summary = tf.summary.scalar('class acc', class_acc)

    class_merged = tf.summary.merge([class_loss_summary, class_acc_summary])

    ''' # get the valid paragraph, useless for the classify
    valid_para = extract_valid(
        fuse_vector=fuse_vector,
        classification_vector=tf.nn.sigmoid(classification_vector),
        max_seq=hp.max_seq_length,
        embd_size=4 * hp.bert_embedding_size,
        pos_para=hp.pos_para
    ).valid_para
    '''

    train_op = tf.train.GradientDescentOptimizer(learning_rate=hp.learning_rate).minimize(class_loss)

    init = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('bidaf_classify/log', sess.graph)
        saver = tf.train.Saver(max_to_keep=hp.max_to_keep)
        counter = 0
        index = 0
        passages = np.load('../../data/marco_embd/marco_train_passage_' + str(index) + '.npy')
        querys = np.load('../../data/marco_embd/marco_train_query_' + str(index) + '.npy')
        labels = np.load('../../data/marco_embd/marco_train_label_' + str(index) + '.npy')
        for epoch in range(5):
            for i in range(passages.shape[0]):
                passage_input = passages[i]
                query_input = np.tile(querys[i], [hp.max_para, 1, 1])
                label_input = labels[i]
                dict = {
                    passage: passage_input,
                    query: query_input,
                    label: label_input,
                    keep_prob: hp.keep_prob
                }
                sess.run(train_op, feed_dict=dict)
                if i % hp.loss_acc_iter == 0:
                    if sess.run(class_loss, feed_dict=dict) != 0:
                        writer.add_summary(sess.run(class_merged, feed_dict=dict), counter)
                        counter += 1
                saver.save(sess, 'bidaf_classify/model/my_model', global_step=i)
