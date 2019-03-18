"""
this version each time only handle with only one paragraph
when concat with the classification, the sequence length of each paragraph should multiple by two
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from hyperparameters import Hyperparameters as hp
import tensorflow as tf
from load_marco import load_marco
from load_dict import load_dict
from BiDAF import BiDAF
from bert import bert_server
from ptr_generator import PTR_Gnerator

'''
vocab = load_dict(path=hp.word)
marco_dev = load_marco(
    vocab=vocab,
    path=hp.marco_dev_path,
    max_seq_length=hp.max_seq_length,
    max_para=hp.max_para
)
'''

'''
marco_train = load_marco(
    vocab=vocab, 
    path=hp.marco_train_path, 
    max_seq_length=hp.max_seq_length, 
    max_para=hp.max_para
)
'''

with tf.device('/cpu:0'):
    #bert = bert_server()

    # batch size is one
    # in later version the max_seq_length of the passage will be doubled
    passage = tf.placeholder(shape=[None, hp.max_seq_length, hp.bert_embedding_size], dtype=tf.float32)
    query = tf.placeholder(shape=[None, hp.max_seq_length, hp.bert_embedding_size], dtype=tf.float32)
    answer_embd = tf.placeholder(dtype=tf.float32, shape=[hp.max_seq_length, hp.bert_embedding_size])
    answer_word_embd = tf.placeholder(dtype=tf.float32, shape=[hp.max_seq_length, hp.bert_embedding_size])
    answer_indices = tf.placeholder(dtype=tf.int32, shape=[hp.max_seq_length, 2])

    fuse_vector = BiDAF(
        refc=passage,
        refq=query,
        cLength=hp.max_seq_length,
        qLength=hp.max_seq_length,
        hidden_units=hp.bert_embedding_size
    ).fuse_vector
    fuse_vector = tf.reshape(fuse_vector, shape=[hp.max_seq_length, 4 * hp.bert_embedding_size])

    ptrg = PTR_Gnerator(
        bert_embedding_size=hp.bert_embedding_size,
        max_seq_length=hp.max_seq_length,
        ptr_conv_beta=hp.ptr_conv_beta
    )

    coverage_vector_t = tf.Variable(tf.zeros(shape=[hp.max_seq_length, 1]))

    # all variables for attention
    Wh = tf.Variable(tf.random_normal(shape=[4 * hp.bert_embedding_size, hp.attention_inter_size]), dtype=tf.float32)
    Ws = tf.Variable(tf.random_normal(shape=[hp.bert_embedding_size, hp.attention_inter_size]), dtype=tf.float32)
    wc = tf.Variable(tf.random_normal(shape=[1, hp.attention_inter_size]), dtype=tf.float32)
    batten = tf.Variable(tf.constant(0.1, shape=[1, hp.attention_inter_size]), dtype=tf.float32)
    V = tf.Variable(tf.random_normal(shape=[hp.attention_inter_size, 1]), dtype=tf.float32)

    # all variables for pvocab
    pvocab_w = tf.Variable(tf.random_normal(shape=[5 * hp.bert_embedding_size, hp.vocab_size]), dtype=tf.float32)
    pvocab_b = tf.Variable(tf.constant(0.1, shape=[1, hp.vocab_size]), dtype=tf.float32)

    # all variables for pointer
    wh = tf.Variable(tf.random_normal(shape=[4 * hp.bert_embedding_size, 1]), dtype=tf.float32)
    ws = tf.Variable(tf.random_normal(shape=[hp.bert_embedding_size, 1]), dtype=tf.float32)
    wx = tf.Variable(tf.random_normal(shape=[hp.bert_embedding_size, 1]), dtype=tf.float32)
    bptr = tf.Variable(tf.constant(0.1, shape=[1, 1]), dtype=tf.float32)

    # variable for calculate loss
    vocab_size = tf.Variable(tf.constant(hp.vocab_size, shape=[1]), trainable=False)

    LOSS = 0

    with tf.variable_scope('ptr_generator', reuse=tf.AUTO_REUSE):
        for index in range(hp.max_seq_length):
            indices = [index]
            st = tf.gather_nd(answer_embd, indices)
            set = tf.expand_dims(st, 0)
            xt = tf.gather_nd(answer_word_embd, indices)
            word_indice = tf.gather_nd(answer_indices, indices)
            at = ptrg.attention(
                Wh=Wh,
                H=fuse_vector,
                Ws=Ws,
                st=set,
                wc=wc,
                coverage=coverage_vector_t,
                batten=batten,
                v=V
            )

            coverage_vector_t = tf.add(coverage_vector_t, at)
            attention_t = tf.tile(at, [1, 4 * hp.bert_embedding_size])
            h_star_t = tf.reduce_sum(tf.multiply(fuse_vector, attention_t), axis=0)

            pvocab = ptrg.pvocab(
                hstar_t=h_star_t,
                st=st,
                w=pvocab_w,
                b=pvocab_b
            )
            p_overall = tf.concat([pvocab, tf.reshape(at, shape=[-1, hp.max_seq_length])], axis=1)

            pgen = ptrg.pointer(
                wh=wh,
                hstar_t=tf.expand_dims(h_star_t, axis=0),
                ws=ws,
                st=tf.expand_dims(st, axis=0),
                wx=wx,
                xt=tf.expand_dims(xt, axis=0),
                bptr=bptr
            )

            LOSS += ptrg.loss(
                p_overall=p_overall,
                words_indice=tf.expand_dims(word_indice, axis=0),
                vocab_size=vocab_size,
                pgen=pgen,
                at=at,
                coverage_vector_t=coverage_vector_t
            )

    LOSS = tf.reduce_sum(LOSS, axis=0)
    loss = LOSS / hp.max_seq_length
    train_op = tf.train.GradientDescentOptimizer(hp.learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    '''
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(hp.epoch):
            for i in range(marco_dev.total):
                passage_marco = marco_dev.passage[i]
                label = marco_dev.label[i]
                query_marco = marco_dev.question[i]
                answer = marco_dev.answer[i]
                answer_word = marco_dev.answer_word[i]
                answer_index = marco_dev.answer_index[i]
                # ------------------- in the later version, the lower codes will be removed ---------------------------
                passage_input = ['']
                for q in range(len(label)):
                    if label[q] == 1:
                        passage_input = [passage_marco[q]]
                # ------------------- in the later version, the upper codes will be removed ---------------------------
                if passage_input[0] == '':
                    continue
                passage_input = bert.convert2vector(passage_input)
                query_input = bert.convert2vector([query_marco])
                answer_input = bert.convert2vector([answer])[0]
                answer_word_input = bert.convert2vector(answer_word)[0]
                dict = {
                    passage: passage_input,
                    query: query_input,
                    answer_embd: answer_input,
                    answer_word_embd: answer_word_input,
                    answer_indices: answer_index
                }
                sess.run(train_op, feed_dict=dict)
                print(sess.run(st, feed_dict=dict))
            '''

