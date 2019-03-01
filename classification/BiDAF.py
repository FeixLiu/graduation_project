from hyperparameters import Hyperparameters as hp
from bert import Bert_server as bs
import tensorflow as tf
import numpy as np


class BiDAF():
    """
    :param refc: the first reference c
    :param refq: the second reference q
    :param cLength: the length of c
    :param qLength: the length of q
    :param hidden_units: the nums of hidden units of the last lstm

    :arg _sim_Mat: the similarity matrix between c and q [batch_size, cLength, qLength]
    :arg _c2q_attention: the c to q attention [batch_size, qLength, bert_embedding_size]
    :arg _q2c_attention: the q to c attention [batch_size, qLength, bert_embedding_size]
    :arg fuse_vector: the output of BiDAF # [batch_size, qLength, 4 * bert_embedding_size]
    """
    def __init__(self, refc, refq, cLength, qLength, hidden_units):
        self._refc = refc
        self._refq = refq
        self._cLength = cLength
        self._qLength = qLength
        self._hidden_units = hidden_units
        self.fuse_vector = self.biAttention()

    def biAttention(self):
        self._sim_Mat = self.simMat()
        self._c2q_attention = self.c2q_attention()
        self._q2c_attention = self.q2c_attention()
        fuse_vector = self.calculateG()
        return fuse_vector

    def simMat(self):
        weights_coMat = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[3 * self._hidden_units, 1]))
        cExp = tf.tile(tf.expand_dims(self._refc, 2), [1, 1, self._qLength, 1])
        qExp = tf.tile(tf.expand_dims(self._refq, 1), [1, self._cLength, 1, 1])
        simMat = tf.concat([cExp, qExp, tf.math.multiply(cExp, qExp)], axis=3)
        simMat = tf.reshape(simMat, [-1, 3 * self._hidden_units])
        simMat = tf.matmul(simMat, weights_coMat)
        simMat = tf.reshape(simMat, [-1, self._cLength, self._qLength])
        return simMat

    def c2q_attention(self):
        soft_sim = tf.nn.softmax(self._sim_Mat, axis=2)
        attention_weight = tf.tile(tf.reduce_sum(soft_sim, axis=2, keepdims=True), [1, 1, self._qLength])
        c2q_attention = tf.matmul(attention_weight, self._refq)
        return c2q_attention

    def q2c_attention(self):
        soft_sim = tf.nn.softmax(tf.reduce_max(self._sim_Mat, axis=2), axis=1)
        attented_context_vector = tf.matmul(tf.expand_dims(soft_sim, 1), self._refc)
        q2c_attention = tf.tile(attented_context_vector, [1, self._cLength, 1])
        return q2c_attention

    def calculateG(self):
        hu = tf.concat([self._refc, self._c2q_attention], axis=2)
        hmu = tf.math.multiply(self._refc, self._c2q_attention)
        hmh = tf.math.multiply(self._refc, self._q2c_attention)
        fuse_vector = tf.concat([hu, hmu, hmh], axis=2)
        return fuse_vector
