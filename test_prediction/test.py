import tensorflow as tf
import numpy as np

a = tf.placeholder(shape=[8, 3072], dtype=tf.float32)
b = tf.placeholder(shape=[8, 2], dtype=tf.int32)

W = tf.Variable(tf.truncated_normal(stddev=0.1, mean=0., shape=[3072, 30000], dtype=tf.float32))
B = tf.Variable(tf.constant(0.1, shape=[1, 30000]))

losses = []
pred = []

for i in range(8):
    index = [i]
    input = tf.expand_dims(a[i], axis=0)
    output = tf.expand_dims(b[i], axis=0)
    p = tf.add(tf.matmul(input, W), B)
    p = tf.nn.softmax(p)
    prediction = tf.argmax(p, axis=1)
    pred.append(prediction)
    prob = tf.gather_nd(p, output)
    loss = 0. - tf.math.log(prob)
    losses.append(loss)

losses = tf.stack(losses)
losses = tf.reduce_sum(losses, axis=0)
pred = tf.stack(pred)

train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(losses)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    a_in = np.random.rand(8, 3072)
    b_in = np.array([[0, 3412], [0, 5413], [0, 131], [0, 643], [0, 13417], [0, 8431], [0, 310], [0, 9]])
    for j in range(100000):
        dict = {
            a: a_in,
            b: b_in
        }
        sess.run(train_op, feed_dict=dict)
        print(sess.run(pred, feed_dict=dict))
