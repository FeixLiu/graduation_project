import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from hyperparameters import Hyperparameters as hp
import tensorflow as tf
from load_marco import load_marco
from load_dict import load_dict
from BiDAF import BiDAF
from classification_vector import classification
from bert import bert_server
from extract_valid_para import extract_valid

vocab = load_dict(path=hp.word)
marco_dev = load_marco(
    vocab=vocab,
    path=hp.marco_dev_path,
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

    classification_vector = classification(
        fuse_vector=fuse_vector,
        embedding_size=4 * hp.bert_embedding_size,
        max_seq_length=hp.max_seq_length,
        bert_embedding_size=hp.bert_embedding_size,
        keep_prob=keep_prob
    ).class_vector

    loss_class = tf.reduce_sum(
        tf.reduce_sum(
            tf.nn.weighted_cross_entropy_with_logits(
                targets=label,
                logits=classification_vector,
                pos_weight=hp.pos_weight),
            axis=0
        ),
        axis=0
    )

    ''' # get the valid paragraph, useless for the classify
    valid_para = extract_valid(
        fuse_vector=fuse_vector,
        classification_vector=classification_vector,
        max_seq=hp.max_seq_length,
        embd_size=4 * hp.bert_embedding_size,
        pos_para=hp.pos_para
    ).valid_para
    '''

    train_op = tf.train.GradientDescentOptimizer(learning_rate=hp.learning_rate).minimize(loss_class)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for _ in range(hp.epoch):
            for i in range(marco_dev.total):
                passage_input = marco_dev.passage[i]
                query_input = marco_dev.question[i]
                passage_input = bert.convert2vector(passage_input)
                query_input = bert.convert2vector([query_input for _ in range(hp.max_para)])
                label_input = marco_dev.label[i]
                dict = {
                    passage: passage_input,
                    query: query_input,
                    label: label_input,
                    keep_prob: hp.keep_prob
                }
                sess.run(train_op, feed_dict=dict)
                print(sess.run(loss_class, feed_dict=dict))
