import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from hyperparameters import Hyperparameters as hp
import tensorflow as tf
from load_marco_simplify import load_marco
import numpy as np
from BiDAF import BiDAF
from classification_vector import classification
from BiLSTM import BiLSTM
from extract_valid_para import extract_valid
from load_dict import load_dict
from ptr_generator import PTR_Gnerator

vocab = load_dict(hp.word, hp.embedding_size)
marco_train = load_marco(
    vocab=vocab,
    path=hp.marco_train_path,
    max_seq_length=hp.max_seq_length,
    max_para=hp.max_para
)

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
        ans_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, hp.max_seq_length])
        context_embedding = tf.nn.embedding_lookup(embedding_weight, context_input_ids)
        qas_embedding = tf.nn.embedding_lookup(embedding_weight, qas_input_ids)
        ans_embedding = tf.nn.embedding_lookup(embedding_weight, ans_input_ids)

        label_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        answer_index = tf.placeholder(dtype=tf.int32, shape=[hp.max_seq_length, 2])


    with tf.variable_scope('context_lstm', reuse=tf.AUTO_REUSE):
        context_lstm = BiLSTM(
            inputs=context_embedding,
            time_steps=hp.max_seq_length,
            hidden_units=hp.embedding_size,
            batch_size=hp.max_para,
            name='context_lstm'
        ).result

    with tf.variable_scope('qas_lstm', reuse=tf.AUTO_REUSE):
        qas_lstm = BiLSTM(
            inputs=qas_embedding,
            time_steps=hp.max_seq_length,
            hidden_units=hp.embedding_size,
            batch_size=1,
            name='qas_lstm'
        ).result

    with tf.variable_scope('ans_lstm', reuse=tf.AUTO_REUSE):
        ans_lstm = BiLSTM(
            inputs=ans_embedding,
            time_steps=hp.max_seq_length,
            hidden_units=hp.embedding_size,
            batch_size=1,
            name='ans_lstm'
        ).result[0]

    with tf.variable_scope('bidaf', reuse=tf.AUTO_REUSE):
        fuse_vector = BiDAF(
            refc=context_lstm,
            refq=tf.tile(qas_lstm, [hp.max_para, 1, 1]),
            cLength=hp.max_seq_length,
            qLength=hp.max_seq_length,
            hidden_units=2 * hp.embedding_size,
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
                    targets=label_input,
                    logits=classification_vector,
                    pos_weight=hp.pos_weight),
                axis=0),
            axis=0
        )
        class_loss = tf.math.multiply(0.1, class_loss)
        class_loss_summary = tf.summary.scalar("class_loss", class_loss)

    with tf.variable_scope('extract_valid', reuse=tf.AUTO_REUSE):
        valid_para = extract_valid(
            fuse_vector=fuse_vector,
            classification_vector=tf.nn.sigmoid(classification_vector),
            max_seq=hp.max_seq_length,
            embd_size=8 * hp.embedding_size,
            pos_para=hp.pos_para,
        ).valid_para

    with tf.variable_scope('ptr_generator', reuse=tf.AUTO_REUSE):
        ptrg = PTR_Gnerator(
            fuse_vector=valid_para,
            decoder_state=ans_lstm,
            vocab_size=hp.vocab_size,
            attention_inter_size=hp.attention_inter_size,
            fuse_vector_embedding_size=8 * hp.embedding_size,
            context_seq_length=2 * hp.max_seq_length,
            ans_seq_length=hp.max_seq_length,
            decoder_embedding_size=2 * hp.embedding_size,
            ans_ids=answer_index,
            epsilon=hp.epsilon,
            name='ptr_generator'
        )

    with tf.name_scope('pre_loss'):
        pre_loss = ptrg.loss / hp.max_seq_length
        pre_loss_summary = tf.summary.scalar("pre_loss", pre_loss)

    with tf.name_scope('total_loss'):
        total_loss = tf.add(class_loss, pre_loss)
        total_loss_summary = tf.summary.scalar("total_loss", total_loss)

    loss_merged = tf.summary.merge([class_loss_summary, pre_loss_summary, total_loss_summary])

    train_op = tf.train.GradientDescentOptimizer(learning_rate=hp.learning_rate).minimize(total_loss)

    init = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init)
        sess.run(embedding_init, feed_dict={embedding_placeholder: vocab.embd})
        writer = tf.summary.FileWriter('bidaf_class_ptr/log', sess.graph)
        saver = tf.train.Saver(max_to_keep=hp.max_to_keep)
        counter = 0
        for epoch in range(hp.epoch):
            for i in range(marco_train.total):
                counter += 1
                passage_ids = marco_train.passage_index[i]
                label = marco_train.label[i]
                query_ids = marco_train.query_index[i][np.newaxis, :]
                answer_ids = marco_train.answer_index[i][np.newaxis, :]
                answer_indice = marco_train.answer_indice[i]
                labels = marco_train.label[i]
                dict = {
                    keep_prob: hp.keep_prob,
                    label_input: labels,
                    ans_input_ids: answer_ids,
                    context_input_ids: passage_ids,
                    qas_input_ids: query_ids,
                    answer_index: answer_indice
                }
                sess.run(train_op, feed_dict=dict)
                sess.run(train_op, feed_dict=dict)
                if i % hp.loss_acc_iter == 0:
                    writer.add_summary(sess.run(loss_merged, feed_dict=dict), counter)
                    counter += 1
            saver.save(sess, 'bidaf_class_ptr/model/my_model', global_step=epoch)
