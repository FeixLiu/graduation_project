import tensorflow as tf
import numpy as np

a = tf.placeholder(shape=[8, 3072], dtype=tf.float32)
b = tf.placeholder(shape=[8, 2], dtype=tf.int32)

W = tf.Variable(tf.truncated_normal(stddev=0.1, mean=0., shape=[3072, 10], dtype=tf.float32))
B = tf.Variable(tf.constant(0.1, shape=[1, 10]))

q = tf.add(tf.matmul(a, W), B)
p = tf.nn.softmax(q, axis=1)
prediction = tf.argmax(p, axis=1)

prob = tf.gather_nd(p, b)
loss = 0. - tf.math.log(prob)
loss = tf.reduce_sum(loss, axis=0)

train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    a_in = np.random.rand(8, 3072)
    b_in = np.array([[0, 2], [1, 5], [2, 1], [3, 6], [4, 7], [5, 8], [6, 0], [7, 9]])
    for j in range(100000):
        dict = {
            a: a_in,
            b: b_in
        }
        sess.run(train_op, feed_dict=dict)
        print(sess.run(B, feed_dict=dict))
