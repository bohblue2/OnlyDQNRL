import numpy as np
import tensorflow as tf
import gym

def pg(input_layer):
    L1 = tf.layers.dense(input_layer, 20, activation=tf.nn.relu)
    L2 = tf.layers.dense(L1, 20, activation=tf.nn.relu)
    action_pred = tf.layers.dense(L2, 2, activation=tf.nn.softmax)

    return action_pred

def train(action_pred, Y, advantages):
    log_lik = -Y * tf.log(action_pred)
    log_lik_adv = log_lik * advantages
    loss = tf.reduce_mean(tf.reduce_sum(log_lik_adv, axis=1))
    train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    return train

def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r

