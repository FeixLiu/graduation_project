import tensorflow as tf
import numpy as np


class PTR_Gnerator():
    def __init__(self, fuse_vector, decoder_state, vocab_size, attention_inter_size, fuse_vector_embedding_size,
                 context_seq_length, ans_seq_length, answer_length, decoder_embedding_size, word_embd,
                 word_embd_size, ans_ids, name):
        self._fuse_vector = fuse_vector
        self._decoder_state = decoder_state
        self._vocab_size = vocab_size
        self._attention_inter_size = attention_inter_size
        self._context_seq_length = context_seq_length
        self._ans_seq_length = ans_seq_length
        self._fuse_vector_embedding_size = fuse_vector_embedding_size
        self._answer_length = answer_length
        self._decoder_embedding_size = decoder_embedding_size
        self._word_embd = word_embd
        self._word_embd_size = word_embd_size
        self._ans_ids = ans_ids
        self._ans_index = tf.expand_dims(ans_ids[:, 1], axis=1)
        self._name = name
        self._attention = self._get_attention()
        self._h_star = self._get_h_star()
        self._pvocab = self._get_pvocab()
        self._p_overall = tf.concat([self._pvocab, tf.transpose(self._attention)], axis=1)
        self._pgen = self._get_pgen()
        self.prediction = self._get_pre()
        self.loss = self._get_loss()

    def _get_attention(self):
        Wh = tf.Variable(tf.random_normal(shape=[self._fuse_vector_embedding_size, self._attention_inter_size]),
                         dtype=tf.float32,
                         name=self._name + '_Wh_attention')
        Ws = tf.Variable(tf.random_normal(shape=[self._decoder_embedding_size, self._attention_inter_size]),
                         dtype=tf.float32,
                         name=self._name + '_Ws_attention')
        batten = tf.Variable(tf.constant(0.1, shape=[1, self._attention_inter_size]),
                             dtype=tf.float32,
                         name=self._name + '_batten_attention')
        v = tf.Variable(tf.random_normal(shape=[self._attention_inter_size, 1]),
                        dtype=tf.float32,
                         name=self._name + '_v_attention')
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

    def _get_h_star(self):
        hi = tf.tile(tf.expand_dims(self._fuse_vector, axis=1), [1, self._ans_seq_length, 1])
        at = tf.tile(tf.expand_dims(self._attention, axis=2), [1, 1, self._fuse_vector_embedding_size])
        h_star = tf.reduce_sum(tf.math.multiply(hi, at), axis=0)
        return h_star

    def _get_pvocab(self):
        p_pre = tf.concat([self._h_star, self._decoder_state], axis=1)
        b = tf.Variable(tf.random_normal(shape=[1, self._vocab_size]),
                        dtype=tf.float32,
                        name=self._name + '_B_pvocab')
        V = tf.Variable(tf.random_normal(shape=[self._fuse_vector_embedding_size + self._decoder_embedding_size,
                                                self._vocab_size]),
                        dtype=tf.float32,
                        name=self._name + '_V_pvocab')
        p_vocab = tf.nn.tanh(tf.add(tf.matmul(p_pre, V), b))
        p_vocab = tf.nn.softmax(p_vocab, axis=1)
        return p_vocab

    def _get_pgen(self):
        wh = tf.Variable(tf.random_normal(shape=[self._fuse_vector_embedding_size, 1]),
                         dtype=tf.float32,
                         name=self._name + '_wh_pgen')
        ws = tf.Variable(tf.random_normal(shape=[self._decoder_embedding_size, 1]),
                         dtype=tf.float32,
                         name=self._name + '_ws_pgen')
        wx = tf.Variable(tf.random_normal(shape=[self._word_embd_size, 1]),
                         dtype=tf.float32,
                         name=self._name + '_wx_pgen')
        bptr = tf.Variable(tf.constant(0.1, shape=[1, 1]))
        whh = tf.matmul(self._h_star, wh)
        wss = tf.matmul(self._decoder_state, ws)
        wxx = tf.matmul(self._word_embd, wx)
        pgen = tf.add(tf.add(tf.add(whh, wss), wxx), bptr)
        pgen = tf.nn.sigmoid(pgen)
        return pgen

    def _get_loss(self):
        answer_prob = tf.expand_dims(tf.gather_nd(self._p_overall, self._ans_ids), axis=1)
        vocab_dim = tf.Variable(tf.constant(self._vocab_size, shape=[self._ans_seq_length, 1]),
                                dtype=tf.int32,
                                trainable=False,
                                name=self._name + '_vocab_dim')
        no_pgen = tf.greater(self._ans_index, vocab_dim)
        no_pgen = tf.cast(no_pgen, tf.float32)
        yes_pgen = tf.less_equal(self._ans_index, vocab_dim)
        yes_pgen = tf.cast(yes_pgen, tf.float32)
        p_w_t = tf.math.add(
            tf.math.multiply(
                tf.math.multiply(
                    answer_prob,
                    no_pgen
                ),
                (1. - self._pgen)
            ),
            tf.math.multiply(
                tf.math.multiply(
                    answer_prob,
                    yes_pgen
                ),
                self._pgen
            )
        )
        p_w_t = tf.reduce_sum(p_w_t, axis=1)
        #p_w_t = tf.slice(p_w_t, tf.constant(0, shape=[1], dtype=tf.int32), self._answer_length)
        loss_prob_t = tf.reduce_sum(0. - tf.math.log(tf.clip_by_value(p_w_t, 1e-8, 1.0)), axis=0)
        return loss_prob_t

    def _get_pre(self):
        pgenpv = tf.math.multiply(self._pgen, self._pvocab)
        pgenat = tf.math.multiply(tf.subtract(1., - self._pgen), self._attention)
        pgenpoverall = tf.concat([pgenpv, tf.transpose(pgenat)], axis=1)
        prediction = tf.argmax(pgenpoverall, axis=1)
        return prediction
