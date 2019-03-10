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
        et = tf.reshape(et, [-1, hp.max_seq_length, 1])
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

