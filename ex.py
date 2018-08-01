import numpy as np
import tensorflow as tf

action = np.array([[0, 1], [1, 0]])
#action = np.array([[[1,1,1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1,1]]])





distribution = np.array([[[1,1,1,1,1,1,1,1,1,1], [2,2,2,2,2,2,2,2,2,2]],[[3,3,3,3,3,3,3,3,3,3],[4,4,4,4,4,4,4,4,4,4]]])

a = tf.placeholder(tf.float32, [None, 2])
b = tf.placeholder(tf.float32, [None, 2, 10])
c = tf.expand_dims(a, -1)
d = b * c
e = tf.reduce_sum(d, axis=1)
#a = tf.placeholder(tf.float32, [None, 2])
#b = tf.shape_n(a)



sess = tf.Session()
print(sess.run(a, feed_dict={a: action}))
#print(sess.run(b, feed_dict={b: distribution}))
#print(sess.run(c, feed_dict={a: action, b: distribution}))

print(sess.run(d, feed_dict={a: action, b: distribution}))
print(sess.run(e, feed_dict={a: action, b: distribution}))