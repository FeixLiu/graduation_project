import tensorflow as tf


class BiDAF():
    """
    self._refc (tensor): the context tensor
        shape = [paragraph_numbers, max_seq_length, bert_embedding_size]
    self._refq (tensor): the query tensor
        shape = [paragraph_numbers, max_seq_length, bert_embedding_size]
    self._cLength (int): the length of the refc, equal with max_seq_length
    self._qLength (int): the length of the refq, equal with max_seq_length
    self._hidden_units (int): the hidden units of the embedding layer, equal with bert_embedding_size
    self.fuse_vector (tensor): fuse_vector of the BiDAF
        shape = [paragraph_numbers, max_seq_length, 4 * bert_embedding_length]
    self._sim_Mat (tensor): the similarity matrix between text and query
        shape = [paragraph_numbers, max_seq_length, max_seq_length]
    self._c2q_attention (tensor): text to query attention
        shape = [paragraph_numbers, max_seq_length, bert_embedding_size]
    self._q2c_attention (tensor): text to query attention
        shape = [paragraph_numbers, max_seq_length, bert_embedding_size]
    """
    def __init__(self, refc, refq, cLength, qLength, hidden_units):
        """
        function: initialize the class
        :param refc (tensor): the context tensor
            shape = [paragraph_numbers, max_seq_length, bert_embedding_size]
        :param refq (tensor): the query tensor
            shape = [paragraph_numbers, max_seq_length, bert_embedding_size]
        :param cLength (int): the length of the refc, equal with max_seq_length
        :param qLength (int): the length of the refq, equal with max_seq_length
        :param hidden_units (int): the hidden units of the embedding layer, equal with bert_embedding_size
        """
        self._refc = refc
        self._refq = refq
        self._cLength = cLength
        self._qLength = qLength
        self._hidden_units = hidden_units
        self.fuse_vector = self._biAttention()

    def _biAttention(self):
        """
        function: the process of the BiDAF
        :return fuse_vector (tensor): fuse_vector of the BiDAF
            shape = [paragraph_numbers, max_seq_length, 4 * bert_embedding_length]
        """
        self._sim_Mat = self._simMat()
        self._c2q_attention = self._c2q_attention()
        self._q2c_attention = self._q2c_attention()
        fuse_vector = self._calculateG()
        return fuse_vector

    def _simMat(self):
        """
        function: calculate the similarity matrix between text and query
        :return simMat (tensor): the similarity matrix between text and query
            shape = [paragraph_numbers, max_seq_length, max_seq_length]
        """
        weights_coMat = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[6 * self._hidden_units, 1]))
        cExp = tf.tile(tf.expand_dims(self._refc, 2), [1, 1, self._qLength, 1])
        qExp = tf.tile(tf.expand_dims(self._refq, 1), [1, self._cLength, 1, 1])
        simMat = tf.concat([cExp, qExp, tf.math.multiply(cExp, qExp)], axis=3)
        simMat = tf.reshape(simMat, [-1, 6 * self._hidden_units])
        simMat = tf.matmul(simMat, weights_coMat)
        simMat = tf.reshape(simMat, [-1, self._cLength, self._qLength])
        return simMat

    def _c2q_attention(self):
        """
        function: calculate the attention from the text to the query
        :return c2q_attention (tensor): text to query attention
            shape = [paragraph_numbers, max_seq_length, bert_embedding_size]
        """
        soft_sim = tf.nn.softmax(self._sim_Mat, axis=2)
        attention_weight = tf.tile(tf.reduce_sum(soft_sim, axis=2, keepdims=True), [1, 1, self._qLength])
        c2q_attention = tf.matmul(attention_weight, self._refq)
        return c2q_attention

    def _q2c_attention(self):
        """
        function: calculate the attention from the query to the text
        :return q2c_attention (tensor): text to query attention
            shape = [paragraph_numbers, max_seq_length, bert_embedding_size]
        """
        soft_sim = tf.nn.softmax(tf.reduce_max(self._sim_Mat, axis=2), axis=1)
        attented_context_vector = tf.matmul(tf.expand_dims(soft_sim, 1), self._refc)
        q2c_attention = tf.tile(attented_context_vector, [1, self._cLength, 1])
        return q2c_attention

    def _calculateG(self):
        """
        function: calculate the bi-direction attention flow fuse_vector with the two side attention
        :return fuse_vector (tensor): fuse_vector of the BiDAF
            shape = [paragraph_numbers, max_seq_length, 4 * bert_embedding_length]
        """
        hu = tf.concat([self._refc, self._c2q_attention], axis=2)
        hmu = tf.math.multiply(self._refc, self._c2q_attention)
        hmh = tf.math.multiply(self._refc, self._q2c_attention)
        fuse_vector = tf.concat([hu, hmu, hmh], axis=2)
        return fuse_vector
