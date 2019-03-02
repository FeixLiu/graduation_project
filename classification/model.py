from hyperparameters import Hyperparameters as hp
import tensorflow as tf
from BiDAF import BiDAF
from marco_dataset import Marco_dataset as md
from bert import Bert_server as bs
from LinearReLU import *
from BiLSTM import BiLSTM
from utils import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.device('/gpu:1'):
    with tf.variable_scope('bert_service', reuse=tf.AUTO_REUSE):
        bert = bs()


    marco_train = md(path=hp.marco_train_path)
    marco_dev = md(path=hp.marco_dev_path)

    para_input = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.bert_embedding_size])
    qas_input = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.bert_embedding_size])
    target = tf.placeholder(dtype=tf.float32, shape=[None, hp.classes])
    keep_prob = tf.placeholder(tf.float32)

    with tf.variable_scope('biAttention', reuse=tf.AUTO_REUSE):
        fuse_vector = BiDAF(
            refc=para_input,
            refq=qas_input,
            cLength=hp.max_seq_length,
            qLength=hp.max_seq_length,
            hidden_units=hp.bert_embedding_size
        ).fuse_vector

    with tf.variable_scope('attentionBiGRU', reuse=tf.AUTO_REUSE):
        relu_fuse_vector = LinearRelu3d(
            inputs=fuse_vector,
            input_length=hp.max_seq_length,
            inputs_size=4 * hp.bert_embedding_size,
            outputs_size=hp.bidaf_lstm_hidden_units,
            keepProb=None
        ).relu
        bidaf_lstm = BiLSTM(
            inputs=relu_fuse_vector,
            input_size=4 * hp.bert_embedding_size,
            time_steps=hp.max_seq_length,
            hidden_units=hp.bidaf_lstm_hidden_units,
            batch_size=hp.batch_size,
            project=False
        )
        bidaf_outputs = bidaf_lstm.outputs
        bidaf_states = bidaf_lstm.states
        bidafBiLSTM = tf.concat([bidaf_outputs[0], bidaf_outputs[1]], axis=2)

    with tf.variable_scope('selfAttention', reuse=tf.AUTO_REUSE):
        self_fuse_vector = BiDAF(
            refc=bidafBiLSTM,
            refq=bidafBiLSTM,
            cLength=hp.max_seq_length,
            qLength=hp.max_seq_length,
            hidden_units=2 * hp.bidaf_lstm_hidden_units
        ).fuse_vector
        relu_self_fuse_vector = LinearRelu3d(
            inputs=self_fuse_vector,
            input_length=hp.max_seq_length,
            inputs_size=8 * hp.bidaf_lstm_hidden_units,
            outputs_size=hp.bidaf_lstm_hidden_units
        ).relu

    with tf.variable_scope('prediction', reuse=tf.AUTO_REUSE):
        attention_sum = tf.add(relu_fuse_vector, relu_self_fuse_vector)
        sum_embedding = tf.reduce_sum(attention_sum, axis=2)
        prediction = LinearRelu2d(
            inputs=sum_embedding,
            inputs_size=hp.max_seq_length,
            outputs_size=hp.classes,
            keepProb=keep_prob
        ).relu

    with tf.variable_scope('lossAndTrainOfClassification'):
        #class balanced cross-entropy
        loss = class_balanced_cross_entropy(labels=target, logtis=prediction, beta=hp.class_balance).loss
        train_op = tf.train.AdamOptimizer().minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
        sess.run(init)
