import tensorflow as tf
from load_dict import load_dict
from load_marco import load_marco
from extract_valid_para import extract_valid
from classification_vector import classification
from hyperparameters import Hyperparameters as hp
from BiDAF import BiDAF
from BiLSTM import BiLSTM
from BiLSTM_time_step import BiLSTM_TS

'''
vocab = load_dict(hp.word, hp.embedding_size)
marco_train = load_marco(
    vocab=vocab,
    path=hp.marco_train_path,
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

        context_input_ids = tf.placeholder(dtype=tf.int32, shape=[10, hp.max_seq_length])
        qas_input_ids = tf.placeholder(dtype=tf.int32, shape=[1, hp.max_seq_length])
        ans_input_ids = tf.placeholder(dtype=tf.int32, shape=[1, hp.max_seq_length])
        label = tf.placeholder(shape=[10, 1], dtype=tf.float32)
        context_embedding = tf.nn.embedding_lookup(embedding_weight, context_input_ids)
        qas_embedding = tf.nn.embedding_lookup(embedding_weight, qas_input_ids)
        ans_word_embedding = tf.nn.embedding_lookup(embedding_weight, ans_input_ids)

        with tf.variable_scope('context_lstm', reuse=tf.AUTO_REUSE):
            context_lstm = BiLSTM(
                inputs=context_embedding,
                hidden_units=hp.embedding_size,
                dropout=hp.keep_prob,
                name='context_lstm'
            ).result

        with tf.variable_scope('qas_lstm', reuse=tf.AUTO_REUSE):
            qas_lstm = BiLSTM(
                inputs=tf.tile(qas_embedding, [hp.max_para, 1, 1]),
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
                axis=0
            )
            class_loss_summary = tf.summary.scalar("class_loss", class_loss)

        with tf.variable_scope('extract_valid', reuse=tf.AUTO_REUSE):
            valid_para = extract_valid(
                fuse_vector=fuse_vector,
                classification_vector=tf.nn.sigmoid(classification_vector),
                max_seq=hp.max_seq_length,
                embd_size=8 * hp.embedding_size,
                pos_para=hp.pos_para,
            ).valid_para

        with tf.variable_scope('ans_time_step_lstm', reuse=tf.AUTO_REUSE):
            ans_bilstm = BiLSTM_TS(
                hidden_units=hp.embedding_size,
                dropout=hp.keep_prob,
                name='ans_time_step_lstm'
            )

        with tf.variable_scope('ptr_generator', reuse=tf.AUTO_REUSE):
            for i in range(hp.max_seq_length):
                xt = ans_word_embedding[:,i]
                time_ans = tf.expand_dims(xt, axis=0)
                st = ans_bilstm.get_embd(time_ans)
                print(xt)
