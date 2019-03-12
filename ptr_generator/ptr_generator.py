import tensorflow as tf
from hyperparameters import Hyperparameters as hp


class PTR_Gnerator():
    def attention(self, Wh, H, Ws, st, wc, coverage, batten, v):
        H = tf.reshape(H, shape=[-1, 4 * hp.bert_embedding_size])
        st = tf.reshape(st, shape=[-1, hp.bert_embedding_size])
        coverage = tf.reshape(coverage, shape=[-1, 1])
        et = tf.matmul(
            tf.nn.tanh(
                tf.add(
                    tf.add(
                        tf.add(
                            tf.matmul(
                                H,
                                Wh
                            ),
                            tf.matmul(
                                st,
                                Ws
                            )
                        ),
                        tf.matmul(
                            coverage,
                            wc
                        )
                    ),
                    batten
                )
            ),
            v
        )
        et = tf.reshape(et, [-1, hp.max_seq_length])
        at = tf.nn.softmax(et, axis=1)
        return at

    def pointer(self, wh, hstar_t, ws, st, wx, xt, bptr):
        pgen = tf.nn.sigmoid(
            tf.add(
                tf.add(
                    tf.add(
                        tf.matmul(
                            hstar_t,
                            wh
                        ),
                        tf.matmul(
                            st,
                            ws
                        )
                    ),
                    tf.matmul(
                        xt,
                        wx
                    )
                ),
                bptr
            )
        )
        return pgen

    def pvocab(self, st, hstar_t, w, b):
        pvocab_pre = tf.concat([hstar_t, st], axis=1)
        pvocab = tf.nn.softmax(
            tf.nn.tanh(
                tf.add(
                    tf.matmul(
                        pvocab_pre,
                        w
                    ),
                    b
                )
            )
        )
        return pvocab

    def loss(self, p_overall, words_indice, vocab_size, pgen, at, coverage_vector_t):
        answer_prob = tf.expand_dims(tf.gather_nd(p_overall, words_indice), axis=1)
        no_pgen = tf.greater(
            tf.expand_dims(
                tf.gather_nd(
                    words_indice,
                    [[0, 1], [1, 1], [2, 1], [3, 1]]
                ),
                axis=1
            ),
            vocab_size
        )
        no_pgen = tf.cast(no_pgen, tf.float32)
        yes_pgen = tf.less_equal(
            tf.expand_dims(
                tf.gather_nd(
                    words_indice,
                    [[0, 1], [1, 1], [2, 1], [3, 1]]
                ),
                axis=1
            ),
            vocab_size
        )
        yes_pgen = tf.cast(yes_pgen, tf.float32)
        p_w_t = tf.math.add(
            tf.math.multiply(
                tf.math.multiply(
                    answer_prob,
                    no_pgen
                ),
                (1. - pgen)
            ),
            tf.math.multiply(
                tf.math.multiply(
                    answer_prob,
                    yes_pgen
                ),
                pgen
            )
        )
        loss_prob_t = tf.reduce_sum(0 - tf.math.log(tf.clip_by_value(p_w_t, 1e-8, 1.0)), axis=0)
        agc = tf.cast(tf.greater(at, coverage_vector_t), tf.float32)
        cga = tf.cast(tf.greater(coverage_vector_t, at), tf.float32)
        covloss_t = tf.reduce_sum(
            tf.add(
                tf.math.multiply(
                    at,
                    agc
                ),
                tf.math.multiply(
                    coverage_vector_t,
                    cga
                )
            ),
            axis=1
        )
        covloss_t = tf.reduce_sum(covloss_t, axis=0)
        loss = tf.math.multiply(hp.alpha, covloss_t) + loss_prob_t
        return loss

