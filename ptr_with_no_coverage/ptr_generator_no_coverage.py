import tensorflow as tf
import numpy as np


class PTR_Gnerator():
    def __init__(self, fuse_vector, decoder_state, vocab_size, attention_inter_size, fuse_vector_embedding_size,
                 context_seq_length, ans_seq_length, decoder_embedding_size, word_embd, word_embd_size, name):
        self._fuse_vector = fuse_vector
        self._decoder_state = decoder_state
        self._vocab_size = vocab_size
        self._attention_inter_size = attention_inter_size
        self._context_seq_length = context_seq_length
        self._ans_seq_length = ans_seq_length
        self._fuse_vector_embedding_size = fuse_vector_embedding_size
        self._decoder_embedding_size = decoder_embedding_size
        self._word_embd = word_embd
        self._word_embd_size = word_embd_size
        self._name = name
        self._attention = self._get_attention()
        self._pvocab = self._get_pvocab()
        self._pgen = self._get_pgen()

    def _get_attention(self):
        Wh = tf.Variable(tf.random_normal(shape=[self._fuse_vector_embedding_size, self._attention_inter_size]),
                         dtype=tf.float32,
                         name=self._name + '_Wh')
        Ws = tf.Variable(tf.random_normal(shape=[self._decoder_embedding_size, self._attention_inter_size]),
                         dtype=tf.float32,
                         name=self._name + '_Ws')
        batten = tf.Variable(tf.constant(0.1, shape=[1, self._attention_inter_size]),
                             dtype=tf.float32,
                         name=self._name + '_batten')
        v = tf.Variable(tf.random_normal(shape=[self._attention_inter_size, 1]),
                        dtype=tf.float32,
                         name=self._name + '_v')
        whh = tf.matmul(self._fuse_vector, Wh)
        H = tf.tile(tf.expand_dims(whh, axis=1), [1, self._ans_seq_length, 1])
        wss = tf.matmul(self._decoder_state, Ws)
        S = tf.tile(tf.expand_dims(wss, axis=0), [self._context_seq_length, 1, 1])
        H = tf.reshape(H, shape=[-1, self._attention_inter_size])
        S = tf.reshape(S, shape=[-1, self._attention_inter_size])
        E = tf.matmul(tf.add(tf.add(H, S), batten), v)
        E = tf.reshape(E, shape=[self._context_seq_length, self._ans_seq_length])
        at = tf.nn.softmax(E, axis=0)
        return at

    def _get_pvocab(self):
        hi = tf.tile(tf.expand_dims(self._fuse_vector, axis=1), [1, self._ans_seq_length, 1])
        at = tf.tile(tf.expand_dims(self._attention, axis=2), [1, 1, self._fuse_vector_embedding_size])
        h_star = tf.reduce_sum(tf.math.multiply(hi, at), axis=0)
        p_pre = tf.concat([h_star, self._decoder_state], axis=1)
        b = tf.Variable(tf.random_normal(shape=[1, self._vocab_size]),
                        dtype=tf.float32,
                        name=self._name + '_B')
        V = tf.Variable(tf.random_normal(shape=[self._fuse_vector_embedding_size + self._decoder_embedding_size,
                                                self._vocab_size]),
                        dtype=tf.float32,
                        name=self._name + '_V')
        p_vocab = tf.add(tf.matmul(p_pre, V), b)
        return p_vocab

    def _get_pgen(self):
        return []


fuse_vector = tf.placeholder(shape=[128, 2400], dtype=tf.float32)
decoder_state = tf.placeholder(shape=[64, 600], dtype=tf.float32)
word_embd = tf.placeholder(shape=[64, 300], dtype=tf.float32)
ptrg = PTR_Gnerator(
    fuse_vector=fuse_vector,
    decoder_state=decoder_state,
    vocab_size=512,
    attention_inter_size=256,
    fuse_vector_embedding_size=2400,
    context_seq_length=128,
    ans_seq_length=64,
    decoder_embedding_size=600,
    word_embd=word_embd,
    word_embd_size=300,
    name='ptr_generator'
)
fv = np.random.randn(128, 2400)
ds = np.random.randn(64, 600)
we = np.random.randn(64, 300)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    dict = {
        fuse_vector: fv,
        decoder_state: ds,
        word_embd: we
    }

    #print(sess.run(ptrg, feed_dict=dict))


