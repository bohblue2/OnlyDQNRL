import tensorflow as tf
import gym
import matplotlib
import numpy as np
import random
from collections import deque
import math
matplotlib.use('TkAgg')

class C51:
    def __init__(self, sess):
        self.sess = sess
        self.input_size = 4
        self.action_size = 2
        self.v_max = 10
        self.v_min = -10
        self.category = 51
        self.minibatch_size = 64

        self.delta_z = (self.v_max - self.v_min) / float(self.category - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.category)]

        self.X = tf.placeholder(tf.float32, [None, self.input_size])
        self.Y = tf.placeholder(tf.float32, [None, self.action_size, self.category])

        self.target_network, self.target_params = self._build_network('target')
        self.main_network, self.main_params = self._build_network('main')

        self.assign_ops = []
        for v_old, v in zip(self.target_params, self.main_params):
            self.assign_ops.append(tf.assign(v_old, v))

    def _build_network(self, name):
        with tf.variable_scope(name):
            layer_1 = tf.layers.dense(inputs=self.X, units=64, activation=tf.nn.selu, trainable=True)
            layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.nn.selu, trainable=True)
            layer_3 = tf.layers.dense(inputs=layer_2, units=self.action_size*self.category, activation=tf.nn.selu, trainable=True)
            reshape = tf.reshape(layer_3, [-1, self.action_size, self.category])
            output = tf.nn.softmax(reshape)
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return output, params

    def choose_action(self, s):
        dist = self.sess.run(self.main_network, feed_dict={self.X: [s]})
        dist = dist[0]
        Q_s_a = [np.dot(dist[i], self.z)for i in range(self.action_size)]
        action = np.argmax(Q_s_a)
        return action


memory_size = 500000
memory = deque(maxlen=memory_size)

sess = tf.Session()
c51 = C51(sess)
sess.run(tf.global_variables_initializer())