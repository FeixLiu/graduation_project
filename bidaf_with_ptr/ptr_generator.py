import tensorflow as tf


class PTR_Gnerator():
    """
    self._bert_embedding_size (int): the bert embedding size
    self._max_seq_length (int): the max sequence length
    self._ptr_conv_beta (float): the beta weight when add two losses
    """
    def __init__(self, bert_embedding_size, max_seq_length, ptr_conv_beta):
        """
        function: initialize the class
        :param bert_embedding_size (int): the bert embedding size
        :param max_seq_length (int): the max sequence length
        :param ptr_conv_beta (float): the beta weight when add two losses
        """
        self._bert_embedding_size = bert_embedding_size
        self._max_seq_length = max_seq_length
        self._ptr_conv_beta = ptr_conv_beta


    def attention(self, Wh, H, Ws, st, wc, coverage, batten, v):
        """
        function: count the attention vector over the input tensor for the time t
        :param Wh (tensor): the weight tensor for the H
            shape: [4 * bert_embedding_size, attention_inter_size]
        :param H (tensor): the input fuse vector from the BiDAF
            shape: [max_seq_length, 4 * bert_embedding_size] # pos_para * max_seq_length for the later version
        :param Ws (tensor): the weight tensor for the st
            shape: [bert_embedding_size, attention_inter_size]
        :param st (tensor): the last state of the answer embedding at the time t
            shape: [1, bert_embedding_size]
        :param wc (tensor): the weight tensor for the coverage
            shape: [1, attention_inter_size]
        :param coverage (tensor): the coverage vector at the time t
            shape: [max_seq_length, 1]
        :param batten (tensor): the bias vector for the attention
            shape: [1, max_seq_length]
        :param v (tensor): the overall weight vector for the attention
            shape: [attention_inter_size, 1]
        :return at (tensor): the attention vector for the time t
            shape: [max_seq_length, 1]
        """
        H = tf.reshape(H, shape=[-1, 4 * self._bert_embedding_size])
        st = tf.reshape(st, shape=[-1, self._bert_embedding_size])
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
        et = tf.reshape(et, [self._max_seq_length, 1])
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
        loss = tf.math.multiply(self._ptr_conv_beta, covloss_t) + loss_prob_t
        return loss

