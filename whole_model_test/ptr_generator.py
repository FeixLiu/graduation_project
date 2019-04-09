import tensorflow as tf


class PTR_Gnerator_test():
    def __init__(self, fuse_vector, vocab_size, attention_inter_size, fuse_vector_embedding_size, context_seq_length,
                 ans_seq_length, decoder_embedding_size, ans_ids, epsilon, name, bert=None, vocab=None):
        self._fuse_vector = fuse_vector
        self._vocab_size = vocab_size
        self._attention_inter_size = attention_inter_size
        self._context_seq_length = context_seq_length
        self._ans_seq_length = ans_seq_length
        self._fuse_vector_embedding_size = fuse_vector_embedding_size
        self._decoder_embedding_size = decoder_embedding_size
        self._ans_ids = ans_ids
        self._ans_index = tf.expand_dims(ans_ids[:, 1], axis=1)
        self._epsilon = epsilon
        self._name = name
        self._bert = bert
        self._vocab = vocab
        self.prediction = self._predict()

    def _predict(self):
        Wh = tf.Variable(tf.truncated_normal(mean=0., stddev=0.01,
                                            shape=[self._fuse_vector_embedding_size, self._attention_inter_size]),
                         dtype=tf.float32,
                         name=self._name + '_Wh_attention')
        Ws = tf.Variable(tf.truncated_normal(mean=0., stddev=0.01,
                                            shape=[self._decoder_embedding_size, self._attention_inter_size]),
                         dtype=tf.float32,
                         name=self._name + '_Ws_attention')
        batten = tf.Variable(tf.constant(0.1, shape=[1, self._attention_inter_size]),
                             dtype=tf.float32,
                         name=self._name + '_batten_attention')
        v = tf.Variable(tf.truncated_normal(mean=0., stddev=0.01,
                                            shape=[self._attention_inter_size, 1]),
                        dtype=tf.float32,
                         name=self._name + '_v_attention')
        b = tf.Variable(tf.constant(0.1, shape=[1, self._vocab_size]),
                        dtype=tf.float32,
                        name=self._name + '_B_pvocab')
        V = tf.Variable(tf.truncated_normal(mean=0., stddev=0.01,
                                            shape=[self._fuse_vector_embedding_size + self._decoder_embedding_size,
                                                   self._vocab_size]),
                        dtype=tf.float32,
                        name=self._name + '_V_pvocab')
        wh = tf.Variable(tf.truncated_normal(mean=0., stddev=0.01,
                                             shape=[self._fuse_vector_embedding_size, 1]),
                         dtype=tf.float32,
                         name=self._name + '_wh_pgen')
        ws = tf.Variable(tf.truncated_normal(mean=0., stddev=0.01,
                                             shape=[self._decoder_embedding_size, 1]),
                         dtype=tf.float32,
                         name=self._name + '_ws_pgen')
        bptr = tf.Variable(tf.constant(0.1, shape=[1, 1]), name=self._name + '_bptr_pgen')
        vocab_dim = tf.Variable(tf.constant(self._vocab_size, shape=[self._ans_seq_length, 1]),
                                dtype=tf.int32,
                                trainable=False,
                                name=self._name + '_vocab_dim')

        current_ans = '<start>'
        loss = 0
        prediction = []
        for i in range(self._ans_seq_length):
            st = self._bert.convert2vector([current_ans])[0][i]
            st = tf.convert_to_tensor(st)
            st = tf.expand_dims(st, axis=0)

            # get attention
            sat = tf.tile(st, [self._context_seq_length, 1])
            Whh = tf.matmul(self._fuse_vector, Wh)
            Wss = tf.matmul(sat, Ws)
            eit = tf.add(tf.add(Whh, Wss), batten)
            eit = tf.matmul(eit, v)
            e_mean, e_var = tf.nn.moments(eit, axes=[0])
            scale_eit = tf.Variable(tf.ones([1]))
            shift_eit = tf.Variable(tf.zeros([1]))
            eit = tf.nn.batch_normalization(eit, e_mean, e_var, shift_eit, scale_eit, self._epsilon)
            at = tf.nn.softmax(eit, axis=0)

            # get hstart
            h_star_t = tf.expand_dims(tf.reduce_sum(tf.math.multiply(self._fuse_vector, at), axis=0), axis=0)

            # get p_vocab
            p_pre_t = tf.concat([h_star_t, st], axis=1)
            p_vocab = tf.add(tf.matmul(p_pre_t, V), b)
            p_mean, p_var = tf.nn.moments(p_vocab, axes=[1])
            p_mean = tf.expand_dims(p_mean, axis=1)
            p_var = tf.expand_dims(p_var, axis=1)
            scale_pvocav = tf.Variable(tf.ones([1, 1]))
            shift_pvocab = tf.Variable(tf.zeros([1, 1]))
            p_vocab = tf.nn.batch_normalization(p_vocab, p_mean, p_var, shift_pvocab, scale_pvocav, self._epsilon)
            p_vocab = tf.nn.softmax(p_vocab, axis=1)

            # get pgen
            whh = tf.matmul(h_star_t, wh)
            wss = tf.matmul(st, ws)
            pgen = tf.add(tf.add(whh, wss), bptr)
            pgen = tf.nn.sigmoid(pgen)

            '''
            # get loss
            answer_id = tf.gather_nd(self._ans_ids, [[i]])
            answer_index = tf.gather_nd(self._ans_index, [i])
            answer_prob = tf.expand_dims(tf.gather_nd(p_vocab, answer_id), axis=1)
            no_pgen = tf.greater(answer_index, self._vocab_size)
            no_pgen = tf.cast(no_pgen, tf.float32)
            yes_pgen = tf.less_equal(answer_index, self._vocab_size)
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
            p_w_t = tf.reduce_sum(p_w_t, axis=1)
            loss_prob_t = tf.reduce_sum(0. - tf.math.log(tf.clip_by_value(p_w_t, 1e-8, 1.0)), axis=0)
            loss += loss_prob_t
            '''

            # get prediction
            pgenpv = tf.math.multiply(pgen, p_vocab)
            pgenat = tf.math.multiply(tf.subtract(1., pgen), tf.transpose(at))
            pgenpoverall = tf.concat([pgenpv, pgenat], axis=1)
            predictiont = tf.argmax(pgenpoverall, axis=1)
            prediction.append(predictiont)

            # get word
            # haven't extract word from the paragraph
            current_word = self._vocab.index2vocab[predictiont]
            current_ans = current_ans + ' ' + current_word

        return prediction
