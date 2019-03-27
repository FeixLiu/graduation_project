"""
this version each time only handle with only one paragraph
when concat with the classification, the sequence length of each paragraph should multiple by two
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from hyperparameters import Hyperparameters as hp
import tensorflow as tf
from load_marco_simplify import load_marco
from load_dict import load_dict
from BiDAF import BiDAF
from ptr_generator_no_coverage import PTR_Gnerator
from bert import bert_server
from BiLSTM_cudnn import BiLSTM
import numpy as np


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
    bert = bert_server()

    keep_prob = tf.placeholder(tf.float32)

    context_input = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.embedding_size])
    qas_input = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.embedding_size])
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
        fuse_vector = tf.reshape(fuse_vector, shape=[hp.max_seq_length, 4 * hp.embedding_size])

    with tf.variable_scope('ptr_generator', reuse=tf.AUTO_REUSE):
        ptrg = PTR_Gnerator(
            fuse_vector=fuse_vector,
            decoder_state=answer_input,
            vocab_size=hp.vocab_size,
            attention_inter_size=hp.attention_inter_size,
            fuse_vector_embedding_size=4 * hp.embedding_size,
            context_seq_length=hp.max_seq_length,
            ans_seq_length=hp.max_seq_length,
            decoder_embedding_size=hp.embedding_size,
            ans_ids=answer_index,
            name='ptr_generator'
        )

    with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
        loss = ptrg.loss
        class_loss_summary = tf.summary.scalar("loss_summary", loss)

    train_op = tf.train.GradientDescentOptimizer(learning_rate=hp.learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('bidaf_ptr/log', sess.graph)
        saver = tf.train.Saver(max_to_keep=hp.max_to_keep)
        counter = 0
        for epoch in range(500000):
            for i in range(1):
                counter += 1
                passage = marco_dev.passage[i]
                label = marco_dev.label[i]
                query = marco_dev.question[i]
                #query_index = query_index[np.newaxis, :]
                answer = marco_dev.answer[i]
                answer_indice = marco_dev.answer_index[i] #(64, 2)
                #answer_id = answer_indice[:, 1] #(64,)
                #answer_id = answer_id[np.newaxis, :]
                # ------------------- in the later version, the lower codes will be removed ---------------------------
                passage_input = ''
                for q in range(len(label)):
                    if label[q] == 1:
                        passage_input = passage[q]
                        break
                # ------------------- in the later version, the upper codes will be removed ---------------------------
                if len(passage_input) == 0:
                    continue
                passage_input = bert.convert2vector([passage_input])
                query_input = bert.convert2vector([query])
                answer = bert.convert2vector([answer])[0]
                dict = {
                    context_input: passage_input,
                    qas_input: query_input,
                    answer_input: answer,
                    answer_index: answer_indice,
                }
                #sess.run(train_op, feed_dict=dict)
                #writer.add_summary(sess.run(class_loss_summary, feed_dict=dict), counter)
                print(sess.run(ptrg._pgen, feed_dict=dict))

