import tensorflow as tf
import gym
import matplotlib
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import math

class C51:
    def __init__(self, sess):
        self.sess = sess
        self.input_size = 4
        self.action_size = 2
        self.v_min = -10
        self.category = 51
        self.v_max = self.v_min + self.category - 1
        self.minibatch_size = 64

        self.delta_z = (self.v_max - self.v_min) / float(self.category - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.category)]

        self.X = tf.placeholder(tf.float32, [None, self.input_size])
        self.Y = tf.placeholder(tf.float32, [None, self.category])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])

        self.target_network, self.target_params = self._build_network('target')
        self.main_network, self.main_params = self._build_network('main')

        expand_dim_action = tf.expand_dims(self.action, -1)
        Q_s_a = tf.reduce_sum(self.main_network * expand_dim_action, axis=1)
        self.cross_entropy = tf.reduce_mean(-(self.Y * tf.clip_by_value(tf.log(Q_s_a), 1e-10, 1)))
        
        self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.cross_entropy)

        self.assign_ops = []
        for v_old, v in zip(self.target_params, self.main_params):
            self.assign_ops.append(tf.assign(v_old, v))

    def train(self, memory):
        #self.minibatch_size = 5
        minibatch = random.sample(memory, self.minibatch_size)
        state_stack = [mini[0] for mini in minibatch]
        next_state_stack = [mini[1] for mini in minibatch]
        action_stack = [mini[2] for mini in minibatch]
        reward_stack = [mini[3] for mini in minibatch]
        done_stack = [mini[4] for mini in minibatch]

        target_dist_batch = self.sess.run(self.target_network, feed_dict={self.X: next_state_stack})
        sum_target_dist = [np.dot(target_dist_element, self.z) for target_dist_element in target_dist_batch]
        next_action = np.argmax(sum_target_dist, axis=1)
        target_distribution = [x[next_action[i]] for i, x in enumerate(target_dist_batch)]
        
        m_probs = []
        a = [[1,2,3,4],[4,3,2,1]]
        b = [-1, 0, 1, 2]
        Q = np.dot(a, b)

        a = [[1, 2],[3, 2],[33, 1]]
        print(np.argmax(a, axis=1))



        

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
env = gym.make('CartPole-v0')
c51 = C51(sess)
sess.run(tf.global_variables_initializer())
sess.run(c51.assign_ops)

for episode in range(1000):
    e = 1. / ((episode / 10) + 1)
    done = False
    state = env.reset()
    global_step = 0
    while not done:
        global_step += 1
        if np.random.rand() < e:
            action = env.action_space.sample()
        else:
            action = c51.choose_action(state)

        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -1
        else:
            reward = 1
        action_one_hot = np.zeros(2)
        action_one_hot[action] = 1
        memory.append([state, next_state, action_one_hot, reward, done])
        if done:
            #c51.train(memory)
            if len(memory) > 100:
                c51.train(memory)   
                sess.run(c51.assign_ops)
            print(episode, global_step)