import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from hyperparameters import Hyperparameters as hp
import tensorflow as tf
from load_marco import load_marco
from BiDAF import BiDAF
from classification_vector import classification
from BiLSTM_cudnn import BiLSTM
from extract_valid_para import extract_valid
from load_dict import load_dict
import numpy as np

vocab = load_dict(hp.word, hp.embedding_size)
marco_train = load_marco(
    vocab=vocab,
    path=hp.marco_train_path,
    max_seq_length=hp.max_seq_length,
    max_para=hp.max_para
)

'''
marco_dev = load_marco(
    vocab=vocab,
    path=hp.marco_dev_path,
    max_seq_length=hp.max_seq_length,
    max_para=hp.max_para
)
'''

with tf.device('/gpu:1'):
    with tf.variable_scope('embedding'):
        embedding_weight = tf.Variable(tf.constant(0.0, shape=[hp.vocab_size, hp.embedding_size]),
                                       trainable=False,
                                       name='embedding_weight')
        embedding_placeholder = tf.placeholder(tf.float32, [hp.vocab_size, hp.embedding_size])
        embedding_init = embedding_weight.assign(embedding_placeholder)
        keep_prob = tf.placeholder(tf.float32)

        context_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, hp.max_seq_length])
        qas_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, hp.max_seq_length])
        label = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        context_embedding = tf.nn.embedding_lookup(embedding_weight, context_input_ids)
        qas_embedding = tf.nn.embedding_lookup(embedding_weight, qas_input_ids)

    with tf.variable_scope('context_lstm', reuse=tf.AUTO_REUSE):
        context_lstm = BiLSTM(
            inputs=context_embedding,
            hidden_units=hp.embedding_size,
            dropout=hp.keep_prob,
            name='context_lstm'
        ).result

    with tf.variable_scope('qas_lstm', reuse=tf.AUTO_REUSE):
        qas_lstm = BiLSTM(
            inputs=qas_embedding,
            hidden_units=hp.embedding_size,
            dropout=hp.keep_prob,
            name='qas_lstm'
        ).result

    with tf.variable_scope('bidaf', reuse=tf.AUTO_REUSE):
        fuse_vector = BiDAF(
            refc=context_lstm,
            refq=qas_lstm,
            cLength=hp.max_seq_length,
            qLength=hp.max_seq_length,
            hidden_units=hp.embedding_size,
            name='bidaf'
        ).fuse_vector

    with tf.variable_scope('features', reuse=tf.AUTO_REUSE):
        features = BiLSTM(
            inputs=fuse_vector,
            hidden_units=8 * hp.embedding_size,
            dropout=hp.keep_prob,
            name='features'
        ).result

    with tf.variable_scope('classification', reuse=tf.AUTO_REUSE):
        classification_vector = classification(
            inputs=features,
            embedding_size=16 * hp.embedding_size,
            max_seq_length=hp.max_seq_length,
            bert_embedding_size=hp.embedding_size,
            keep_prob=keep_prob,
            name='classification'
        ).class_vector

    with tf.name_scope('class_loss'):
        class_loss = tf.reduce_sum(
            tf.reduce_sum(
                tf.nn.weighted_cross_entropy_with_logits(
                    targets=label,
                    logits=classification_vector,
                    pos_weight=hp.pos_weight),
                axis=0),
            axis=0,
            name='class_loss'
        )
        class_loss_summary = tf.summary.scalar("class_loss_summary", class_loss)

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
            axis=0,
            name='class_acc'
        )
        class_acc_summary = tf.summary.scalar('class_acc_summary', class_acc)

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
        #saver = tf.train.Saver()
        #saver.restore(sess, tf.train.latest_checkpoint('./bidaf_classify/model/'))
        writer = tf.summary.FileWriter('bidaf_classify/log', sess.graph)
        saver = tf.train.Saver(max_to_keep=hp.max_to_keep)
        sess.run(embedding_init, feed_dict={embedding_placeholder: vocab.embd})
        counter = 0
        for epoch in range(hp.epoch):
            for i in range(marco_train.total):
                context_id = marco_train.passage_index[i]
                qas_id = np.tile(marco_train.query_index[i][np.newaxis,:], [hp.max_para, 1])
                labels = marco_train.label[i]
                dict = {
                    context_input_ids: context_id,
                    qas_input_ids: qas_id,
                    label: labels,
                    keep_prob: hp.keep_prob
                }
                sess.run(train_op, feed_dict=dict)
                if i % hp.loss_acc_iter == 0:
                    writer.add_summary(sess.run(class_merged, feed_dict=dict), counter)
                    counter += 1
            if epoch % hp.save_model_epoch == 0:
                saver.save(sess, 'bidaf_classify/model/my_model', global_step=epoch)
