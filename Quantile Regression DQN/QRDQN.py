import tensorflow as tf
import gym
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import math


class QRDQN:
    def __init__(self, sess):
        self.sess = sess
        self.input_size = 4
        self.action_size = 2
        self.category = 50
        self.delta_tau = 1/self.category
        self.minibatch_size = 32
        self.gamma = 0.99

        self.X = tf.placeholder(tf.float32, [None, 4])
        self.action = tf.placeholder(tf.float32, [None, 2])

        self.target_network, self.target_params = self._build_network('target')
        self.main_network, self.main_params = self._build_network('main')

        self.theta_s_a = self.main_network
        expand_dim_action = tf.expand_dims(self.action, -1)
        theta_s_a = tf.reduce_sum(self.main_network * expand_dim_action, axis=1)

        self.assign_ops = []
        for v_old, v in zip(self.target_params, self.main_params):
            self.assign_ops.append(tf.assign(v_old, v))

    def train(self, memory, global_step):
        self.minibatch_size = global_step
        minibatch = random.sample(memory, self.minibatch_size)
        state_stack = [mini[0] for mini in minibatch]
        next_state_stack = [mini[1] for mini in minibatch]
        action_stack = [mini[2] for mini in minibatch]
        reward_stack = [mini[3] for mini in minibatch]
        done_stack = [mini[4] for mini in minibatch]

        Q_next_state = self.sess.run(self.target_network, feed_dict={self.X: next_state_stack})
        next_action = np.argmax(np.mean(Q_next_state, axis=2), axis=1)
        Q_next_state_next_action = [Q_next_state[i, action, :] for i, action in enumerate(next_action)]
        T_theta = [np.ones(self.category)*reward if done else reward + 0.99 * Q for reward, Q, done in zip(reward_stack, Q_next_state_next_action, done_stack)]
g

    def _build_network(self, name):
        with tf.variable_scope(name):
            layer_1 = tf.layers.dense(inputs=self.X, units=64, activation=tf.nn.tanh, trainable=True)
            layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.nn.tanh, trainable=True)
            layer_3 = tf.layers.dense(inputs=layer_2, units=self.action_size * self.category, activation=None,
                                      trainable=True)

            output = tf.reshape(layer_3, [-1, self.action_size, self.category])
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return output, params

    def choose_action(self, obs):
        Q = self.sess.run(self.main_network, feed_dict={self.X: [obs]})
        Q_s_a = np.mean(Q[0], axis=1)
        action = np.argmax(Q_s_a)
        return action

env = gym.make('CartPole-v0')
sess = tf.Session()
qrdqn = QRDQN(sess)
sess.run(tf.global_variables_initializer())
sess.run(qrdqn.assign_ops)
memory_size = 500000
memory = deque(maxlen=memory_size)

for episode in range(1):
    e = 1. / ((episode / 10) + 1)
    done = False
    state = env.reset()
    global_step = 0
    while not done:
        global_step += 1
        if np.random.rand() < e:
            action = env.action_space.sample()
        else:
            action = qrdqn.choose_action(state)

        next_state, reward, done, _ = env.step(action)

        if done:
            reward = 0
        else:
            reward = 1
        action_one_hot = np.zeros(2)
        action_one_hot[action] = 1
        memory.append([state, next_state, action_one_hot, reward, done])
        state = next_state
        if done:
            qrdqn.train(memory, global_step)