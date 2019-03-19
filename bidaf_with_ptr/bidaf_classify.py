import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from hyperparameters import Hyperparameters as hp
import tensorflow as tf
from load_marco import load_marco
from load_dict import load_dict
from BiDAF import BiDAF
from classification_vector import classification
from bert import bert_server
from BiLSTM import BiLSTM
from extract_valid_para import extract_valid

vocab = load_dict(path=hp.word)
marco_train = load_marco(
    vocab=vocab,
    path=hp.marco_dev_path,
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

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('bidaf_classify/log', sess.graph)
        saver = tf.train.Saver(max_to_keep=hp.max_to_keep)
        counter = 0
        for epoch in range(hp.epoch):
            for i in range(marco_train.total):
                passage_input = marco_train.passage[i]
                query_input = marco_train.question[i]
                passage_input = bert.convert2vector(passage_input)
                query_input = bert.convert2vector([query_input for _ in range(hp.max_para)])
                label_input = marco_train.label[i]
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
            if epoch % hp.save_model_epoch == 0:
                saver.save(sess, 'bidaf_classify/model/my_model', global_step=epoch)
