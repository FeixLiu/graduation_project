import tensorflow as tf
from load_dict import load_dict
from load_marco import load_marco
from extract_valid_para import extract_valid
from classification_vector import classification
from hyperparameters import Hyperparameters as hp
from BiDAF import BiDAF
from BiLSTM import BiLSTM
from bert import bert_server
from ptr_generator import PTR_Gnerator_test

vocab = load_dict(hp.word, hp.embedding_size)
marco_train = load_marco(
    vocab=vocab,
    path=hp.marco_train_path,
    max_seq_length=hp.max_seq_length,
    max_para=hp.max_para
)

with tf.device('/gpu:1'):
    with tf.variable_scope('placeholder'):
        bert = bert_server()

        context_input = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.embedding_size])
        qas_input = tf.placeholder(dtype=tf.float32, shape=[None, hp.max_seq_length, hp.embedding_size])
        label_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        answer_index = tf.placeholder(dtype=tf.int32, shape=[hp.max_seq_length, 2])
        keep_prob = tf.placeholder(tf.float32)

    with tf.variable_scope('bidaf', reuse=tf.AUTO_REUSE):
        fuse_vector = BiDAF(
            refc=context_input,
            refq=tf.tile(qas_input, [hp.max_para, 1, 1]),
            cLength=hp.max_seq_length,
            qLength=hp.max_seq_length,
            hidden_units=hp.embedding_size,
            name='bidaf'
        ).fuse_vector

    with tf.variable_scope('features', reuse=tf.AUTO_REUSE):
        features = BiLSTM(
            inputs=fuse_vector,
            time_steps=hp.max_seq_length,
            hidden_units=4 * hp.embedding_size,
            batch_size=hp.max_para,
            name='features'
        ).result

    with tf.variable_scope('classification', reuse=tf.AUTO_REUSE):
        classification_vector = classification(
            inputs=features,
            embedding_size=8 * hp.embedding_size,
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
        class_loss_summary = tf.summary.scalar("class_loss", class_loss)

    with tf.variable_scope('extract_valid', reuse=tf.AUTO_REUSE):
        valid_para = extract_valid(
            fuse_vector=fuse_vector,
            classification_vector=tf.nn.sigmoid(classification_vector),
            max_seq=hp.max_seq_length,
            embd_size=4 * hp.embedding_size,
            pos_para=hp.pos_para,
        ).valid_para

    with tf.variable_scope('ptr_generator', reuse=tf.AUTO_REUSE):
        ptrg = PTR_Gnerator_test(
            fuse_vector=valid_para,
            vocab_size=hp.vocab_size,
            attention_inter_size=hp.attention_inter_size,
            fuse_vector_embedding_size=4 * hp.embedding_size,
            bert=bert,
            context_seq_length=2 * hp.max_seq_length,
            ans_seq_length=hp.max_seq_length,
            decoder_embedding_size=hp.embedding_size,
            ans_ids=answer_index,
            epsilon=hp.epsilon,
            vocab=vocab,
            name='ptr_generator'
        )

    init = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, './bidaf_class_ptr/model')
        for epoch in range(hp.epoch):
            for i in range(10):
                passage = marco_train.passage[i]
                label = marco_train.label[i]
                query = marco_train.question[i]
                answer = marco_train.answer[i]
                answer_indice = marco_train.answer_index[i]
                labels = marco_train.label[i]
                passage_embd = bert.convert2vector(passage)
                query_embd = bert.convert2vector([query])
                answer_embd = bert.convert2vector([answer])[0]
                dict = {
                    keep_prob: hp.keep_prob,
                    context_input: passage_embd,
                    qas_input: query_embd,
                    label_input: labels,
                    answer_index: answer_indice
                }
                print(sess.run(ptrg.prediction, feed_dict=dict))
