import tensorflow as tf
import gym
import matplotlib
import numpy as np
import random
from collections import deque
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
        self.action = tf.placeholder(tf.float32, [None, self.action_size])

        self.target_network, self.target_params = self._build_network('target')
        self.main_network, self.main_params = self._build_network('main')

        self.assign_ops = []
        for v_old, v in zip(self.target_params, self.main_params):
            self.assign_ops.append(tf.assign(v_old, v))

    def train(self, memory):
        states_stack, next_states_stack, actions_stack, rewards_stack, dones_stack = [], [], [], [], []
        if len(memory) < self.minibatch_size: in_batch_size = len(memory)
        else: in_batch_size = self.minibatch_size
        minibatch = random.sample(memory, in_batch_size)

        m_prob = [np.zeros((in_batch_size, self.category)) for _ in range(self.action_size)]

        for mini in minibatch:
            states_stack.append(mini[0])
            next_states_stack.append(mini[0])
            actions_stack.append(mini[0])
            rewards_stack.append(mini[0])
            dones_stack.append(mini[0])

        Q_batch = np.array(self.sess.run(self.target_network, feed_dict={self.X: states_stack}))



    def _build_network(self, name):
        all_dist_Q = []
        with tf.variable_scope(name):
            for i in range(self.action_size):
                layer_1 = tf.layers.dense(inputs=self.X, units=128, activation=tf.nn.selu, trainable=True)
                layer_2 = tf.layers.dense(inputs=layer_1, units=128, activation=tf.nn.selu, trainable=True)
                output = tf.layers.dense(inputs=layer_2, units=self.category, activation=tf.nn.softmax, trainable=True)
                all_dist_Q.append(output)
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return all_dist_Q, params

    def choose_action(self, s):
        dist = self.sess.run(self.main_network, feed_dict={self.X: [s]})
        action = np.array([np.dot(i, self.z) for i in dist])
        action = np.argmax(action)
        return action


memory_size = 500000
memory = deque(maxlen=memory_size)

sess = tf.Session()
c51 = C51(sess)
sess.run(tf.global_variables_initializer())
#print(sess.run(c51.target_params))
#print('a')
#sess.run(c51.assign_ops)
#print(sess.run(c51.target_params))
#print('a')
#print(sess.run(c51.main_params))

env = gym.make('CartPole-v0')
for episode in range(1):
    done = False
    state = env.reset()

    while not done:
        action = c51.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        if done: reward = -1
        memory.append([state, next_state, action, reward, done])
        state = next_state
        if done:
            c51.train(memory)