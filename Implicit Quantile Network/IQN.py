import tensorflow as tf
import numpy as np
import math
from collections import deque
import gym
import random

class IQN:
    def __init__(self, sess):
        self.action_size = 2
        self.batch_size = 32
        self.num_quantiles = 32
        self.quantile_embedding_dim = 64
        self.state_size = 4
        self.sess = sess
        self.gamma = 0.99

        self.state = tf.placeholder(tf.float32, [None, self.state_size])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.Y = tf.placeholder(tf.float32, [None, self.num_quantiles])

        self.main_network, self.main_params, self.main_quantiles = self._build_net('main')
        self.target_network, self.target_params, self.target_quantiles = self._build_net('target')

        expand_dim_action = tf.expand_dims(self.action, -1)
        theta_s_a = tf.reduce_sum(self.main_network * expand_dim_action, axis=1)
        quant = tf.reshape(self.main_quantiles, [self.batch_size, self.num_quantiles])
        sorted_quantiles = tf.contrib.framework.sort(quant, direction='ASCENDING')
        ze = tf.zeros([self.batch_size, 1])
        shift_sorted_quantiles = tf.concat([ze, sorted_quantiles], 1)[:,:self.num_quantiles]
        tau = sorted_quantiles - shift_sorted_quantiles
        inv_tau = 1.0 - tau

        Huber_loss = tf.losses.huber_loss(self.Y, theta_s_a, reduction=tf.losses.Reduction.NONE)

        error_loss = self.Y - theta_s_a
        Loss = tf.where(tf.less(error_loss, 0.0), inv_tau * Huber_loss, tau * Huber_loss)
        Loss = tf.reduce_mean(tf.reduce_sum(Loss, axis=1))

        self.train_op = tf.train.AdamOptimizer(1e-5).minimize(Loss)


        self.assign_ops = []
        for v_old, v in zip(self.target_params, self.main_params):
            self.assign_ops.append(tf.assign(v_old, v))

    def train(self, memory):
        minibatch = random.sample(memory, self.batch_size)
        state_stack = [mini[0] for mini in minibatch]
        next_state_stack = [mini[1] for mini in minibatch]
        action_stack = [mini[2] for mini in minibatch]
        reward_stack = [mini[3] for mini in minibatch]
        done_stack = [mini[4] for mini in minibatch]

        done_stack = [int(i) for i in done_stack]

        Q_next_state = self.sess.run(self.target_network, feed_dict={self.state: next_state_stack})
        next_action = np.argmax(np.mean(Q_next_state, axis=2), axis=1)
        Q_next_state_next_action = [Q_next_state[i, action, :] for i, action in enumerate(next_action)]
        T_theta = [reward + (1-done)*self.gamma*Q for reward, Q, done in zip(reward_stack, Q_next_state_next_action, done_stack)]
        for i in T_theta:
            print(i)
        self.sess.run(self.train_op, feed_dict={self.state: state_stack, self.action: action_stack, self.Y: T_theta})

    def _build_net(self, name):
        with tf.variable_scope(name):
            state_net = tf.layers.dense(inputs=self.state, units=64, activation=tf.nn.relu)
            state_net = tf.layers.dense(inputs=state_net, units=64, activation=tf.nn.relu)
            state_net_size = state_net.get_shape().as_list()[-1]
            state_net_tiled = tf.tile(state_net, [self.num_quantiles, 1])

            quantile_shape = [self.num_quantiles * self.batch_size, 1]
            quantiles = tf.random_uniform(quantile_shape, minval=0, maxval=1, dtype=tf.float32) #tau
            quantile_net = tf.tile(quantiles, [1, self.quantile_embedding_dim])
            pi = tf.constant(math.pi)
            quantile_net = tf.cast(tf.range(1, self.quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
            quantile_net = tf.cos(quantile_net)
            quantile_net = tf.layers.dense(inputs=quantile_net, units=state_net_size, activation=tf.nn.relu)

            net = tf.multiply(state_net_tiled, quantile_net)
            net = tf.layers.dense(inputs=net, units=512, activation=tf.nn.relu)
            net = tf.layers.dense(inputs=net, units=128, activation=tf.nn.relu)
            quantile_values = tf.layers.dense(inputs=net, units=self.action_size, activation=None)
            quantile_values = tf.reshape(quantile_values, [self.batch_size, self.action_size, self.num_quantiles])

            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        return quantile_values, params, quantiles

    def choose_action(self, obs):
        obs = np.tile(obs, [self.batch_size, 1])
        Q_s_a = self.sess.run(self.main_network, feed_dict={self.state: obs})
        Q_s_a = np.sum(Q_s_a[0], axis=1)
        action = np.argmax(Q_s_a)
        return action



env = gym.make('CartPole-v0')
sess = tf.Session()
qrdqn = IQN(sess)
sess.run(tf.global_variables_initializer())
sess.run(qrdqn.assign_ops)
memory_size = 500000
memory = deque(maxlen=memory_size)

#r = tf.placeholder(tf.float32)  ########
#rr = tf.summary.scalar('reward', r)
#merged = tf.summary.merge_all()  ########
#writer = tf.summary.FileWriter('/Users/chageumgang/Desktop/OnlyDQNRL/Implicit Quantile Network/board/IQN', sess.graph)  ########

for episode in range(10000):
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
            reward = -1
        else:
            reward = 0
        action_one_hot = np.zeros(2)
        action_one_hot[action] = 1
        memory.append([state, next_state, action_one_hot, reward, done])
        state = next_state
        if done:
            if len(memory) > 1000:
                qrdqn.train(memory)
                sess.run(qrdqn.assign_ops)
            #summary = sess.run(merged, feed_dict={r: global_step})
            #writer.add_summary(summary, episode)
            print(episode, global_step)