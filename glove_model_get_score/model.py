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
from classification_f1 import classification_f1
from bleu import get_blue_1

vocab = load_dict(hp.word, hp.embedding_size)
marco_train = load_marco(
    vocab=vocab,
    path=hp.marco_train_path,
    max_seq_length=hp.max_seq_length,
    max_para=hp.max_para,
    vocab_size=hp.vocab_size
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
            time_steps=hp.max_seq_length,
            hidden_units=8 * hp.embedding_size,
            batch_size=hp.max_para,
            inputs=fuse_vector,
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

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(embedding_init, feed_dict={embedding_placeholder: vocab.embd})
        saver = tf.train.Saver()
        saver.restore(sess, './bidaf_class_ptr/model/my_model-18')
        cf = classification_f1()
        gb1 = get_blue_1()
        for i in range(marco_train.total):
            passage_ids = marco_train.passage_index[i]
            label = marco_train.label[i]
            query_ids = marco_train.query_index[i][np.newaxis, :]
            answer_ids = marco_train.answer_index[i][np.newaxis, :]
            answer_indice = marco_train.answer_indice[i]
            labels = marco_train.label[i]
            para_word = marco_train.para_word[i]
            dict = {
                keep_prob: hp.keep_prob,
                label_input: labels,
                ans_input_ids: answer_ids,
                context_input_ids: passage_ids,
                qas_input_ids: query_ids,
                answer_index: answer_indice
            }
            #print(sess.run(classification_vector, feed_dict=dict))
            #print(labels)
            prediction = sess.run(ptrg.prediction, feed_dict=dict)
            classify = sess.run(classification_vector, feed_dict=dict)
            labels_target = np.reshape(labels, (10))
            labels_target = labels_target.astype(np.int)
            classify = np.reshape(classify, (10))
            classify_temp = 1 / (1 + np.exp(-classify))
            classify = []
            for i in classify_temp:
                if i > 0.5:
                    classify.append(1)
                else:
                    classify.append(0)
            classify = np.array(classify)
            cf.update_acc_recall(labels_target, classify)
            rst = []
            for i in range(len(prediction)):
                if np.sum(prediction[i:]) == 0:
                    break
                if prediction[i] <= hp.vocab_size:
                    rst.append(vocab.index2vocab[prediction[i]])
                else:
                    rst.append(para_word['index2word'][prediction[i] - hp.vocab_size])
            target_temp = []
            target_id = answer_ids[0]
            for i in range(1, len(target_id), 1):
                if np.sum(target_id[i:]) == 0:
                    break
                if target_id[i] <= hp.vocab_size:
                    target_temp.append(vocab.index2vocab[target_id[i]])
                else:
                    target_temp.append(para_word['index2word'][target_id[i] - hp.vocab_size])
            target = [target_temp]
            gb1.update_bleu(target, rst)
        print(cf.total_para, cf.total_right_para, cf.total_pos,cf.pos_right)
        print(gb1.score, gb1.totalsc)
