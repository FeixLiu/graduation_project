import tensorflow as tf


class classification():
    """
    self._fuse_vector (tensor): the output of the BiDAF
        shape: [paragraph_numbers, max_seq_length, max_seq_length]
    self._embedding_size (int): the output size of the BiDAF (4 * bert_embedding_size)
    self._max_seq_length (int): max length of the passage
    self._bert_embedding_size (int): the bert embedding size
    self.class_vector (tensor): the classification vector
        shape: [paragraph_numbers, 1]
    """
    def __init__(self, fuse_vector, embedding_size, max_seq_length, bert_embedding_size):
        """
        function: initialize the class
        :param fuse_vector (tensor): the output of the BiDAF
            shape: [paragraph_numbers, max_seq_length, max_seq_length]
        :param embedding_size (int): the output size of the BiDAF (4 * bert_embedding_size)
        :param max_seq_length (int): max length of the passage
        :param bert_embedding_size (int): the bert embedding size
        """
        self._fuse_vector = fuse_vector
        self._embedding_size = embedding_size
        self._max_seq_length = max_seq_length
        self._bert_embedding_size = bert_embedding_size
        self.class_vector = self._classify()

    def _classify(self):
        """
        function: calculate the classification from the fuse vector
        :return class_vector (tensor): the classification vector
            shape: [paragraph_numbers, 1]
        """
        fuse = tf.reshape(self._fuse_vector, shape=[-1, self._embedding_size])
        classify_weights1 = tf.Variable(tf.random_normal(shape=[self._embedding_size, self._bert_embedding_size]), dtype=tf.float32)
        classify_biases1 = tf.Variable(tf.constant(0.1, shape=[1, self._bert_embedding_size]), dtype=tf.float32)
        classify_weights2 = tf.Variable(tf.random_normal(shape=[self._bert_embedding_size, 1]), dtype=tf.float32)
        classify_biases2 = tf.Variable(tf.constant(0.1, shape=[1, 1]), dtype=tf.float32)
        classify_inter = tf.add(tf.matmul(fuse, classify_weights1), classify_biases1)
        class_vector = tf.add(tf.matmul(tf.nn.relu(classify_inter), classify_weights2), classify_biases2)
        class_vector = tf.reshape(class_vector, shape=[-1, self._max_seq_length, 1])
        class_vector = tf.reduce_sum(class_vector, axis=1)
        return class_vector
