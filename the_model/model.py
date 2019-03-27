import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from hyperparameters import Hyperparameters as hp
import tensorflow as tf
from bert import bert_server
from load_marco_simplify import load_marco
from BiDAF import BiDAF
from classification_vector import classification
from BiLSTM import BiLSTM
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


with tf.device('/gpu:1'):
    bert = bert_server()

    keep_prob = tf.placeholder(tf.float32)

    context_input = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.embedding_size])
    qas_input = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.embedding_size])
    label = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    answer_input = tf.placeholder(dtype=tf.float32, shape=[hp.max_seq_length, hp.embedding_size])
    answer_index = tf.placeholder(dtype=tf.int32, shape=[hp.max_seq_length, 2])

    with tf.variable_scope('bidaf', reuse=tf.AUTO_REUSE):
        fuse_vector = BiDAF(
            refc=context_input,
            refq=qas_input,
            cLength=hp.max_seq_length,
            qLength=hp.max_seq_length,
            hidden_units=hp.embedding_size,
            name='bidaf'
        ).fuse_vector

    with tf.variable_scope('features', reuse=tf.AUTO_REUSE):
        features = BiLSTM(
            inputs=fuse_vector,
            time_steps=hp.max_seq_length,
            hidden_units=8 * hp.embedding_size,
            batch_size=hp.max_para,
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
            axis=0
        )
        class_loss_summary = tf.summary.scalar("class_loss", class_loss)

    with tf.name_scope('extract_valid'):
        valid_para = extract_valid(
            fuse_vector=fuse_vector,
            classification_vector=tf.nn.sigmoid(classification_vector),
            max_seq=hp.max_seq_length,
            embd_size=4 * hp.embedding_size,
            pos_para=hp.pos_para,
        ).valid_para

    init = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('bidaf_ptr/log', sess.graph)
        saver = tf.train.Saver(max_to_keep=hp.max_to_keep)
        counter = 0
        for epoch in range(hp.epoch):
            for i in range(marco_train.total):
                counter += 1
                passage = marco_train.passage[i]
                label = marco_train.label[i]
                query = marco_train.question[i]
                answer = marco_train.answer[i]
                answer_indice = marco_train.answer_index[i]
                labels = marco_train.label[i]
                passage_embd = bert.convert2vector(passage)
                query_embd = bert.convert2vector([query])
                answer_embd = bert.convert2vector([answer])[0]
                print(passage)
                #sess.run(train_op, feed_dict=dict)
                #writer.add_summary(sess.run(class_loss_summary, feed_dict=dict), counter)



