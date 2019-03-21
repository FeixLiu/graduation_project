import tensorflow as tf


class extract_valid():
    """
    self._fuse_vector (tensor): the fuse vector after BiDAF
    self._classification_vector (tensor): indicating which paragraph may answer the question
    self._max_seq (int): max sequence length of the paragraph
    self._embd_size (int): the fuse vector size
    self._pos_para (int): how many positive paragraphs each passage have
    self.valid_para (tensor): the valida paragraphs
        shape: [2 * max_seq_length, 4 * bert_embedding_size]
    """
    def __init__(self, fuse_vector, classification_vector, max_seq, embd_size, pos_para):
        """
        function: initialize the class
        :param fuse_vector (tensor): the fuse vector after BiDAF
        :param classification_vector (tensor): indicating which paragraph may answer the question
        :param max_seq (int): max sequence length of the paragraph
        :param embd_size (int): the fuse vector size
        :param pos_para (int): how many positive paragraphs each passage have
        """
        self._fuse_vector = fuse_vector
        self._classification_vector = classification_vector
        self._max_seq = max_seq
        self._embd_size = embd_size
        self._pos_para = pos_para
        self.valid_para = self._get_para()

    def _get_para(self):
        """
        function: get the valid paragraphs and pad if total paragraphs is less than two
        :return pos_para (tensor): the valida paragraphs
            shape: [2 * max_seq_length, 4 * bert_embedding_size]
        """
        pos_index = tf.reshape(tf.reduce_sum(tf.where(tf.greater(self._classification_vector, 0.5)), axis=1), shape=[-1, 1])
        pos_para = tf.gather_nd(self._fuse_vector, pos_index)
        try:
            if pos_para.shape[0].value < self._pos_para:
                pad = tf.constant(0., shape=[self._pos_para, self._max_seq, self._embd_size])
                pos_para = tf.concat([pos_para, pad], axis=0)
        except TypeError:
            pad = tf.constant(0., shape=[self._pos_para, self._max_seq, self._embd_size])
            pos_para = tf.concat([pos_para, pad], axis=0)
        valid_index = tf.constant([[0], [1]])
        pos_para = tf.gather_nd(pos_para, valid_index)
        pos_para = tf.reshape(pos_para, shape=[-1, self._embd_size])
        return pos_para