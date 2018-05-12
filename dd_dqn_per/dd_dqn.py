import numpy as np
import tensorflow as tf
from typing import List


class DQN:

    def __init__(self, session, input_size, output_size: int, name, mode):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self.mode = mode

        self._build_network()

    def _build_network(self, l_rate=0.01):
        with tf.variable_scope(self.net_name):
            
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")

            net = tf.layers.dense(self._X, 64, activation=tf.nn.tanh)
            net = tf.layers.dense(net, 64, activation=tf.nn.tanh)

            ad_net = tf.layers.dense(net, 64, activation=tf.nn.tanh)
            ad_net = tf.layers.dense(ad_net, 64, activation=None)
            self.advantage = tf.layers.dense(ad_net, self.output_size, activation=None)
            
            v_net = tf.layers.dense(net, 64, activation=tf.nn.tanh)
            v_net = tf.layers.dense(v_net, 64, activation=None)
            self.value = tf.layers.dense(v_net, 1, activation=None)

            if self.mode == 'default':
                self._Qpred = tf.add(self.advantage, self.value)
            elif self.mode == 'mean':
                self._Qpred = tf.add(self.advantage, tf.reduce_mean(self.value))
            elif self.mode == 'max':
                self._Qpred = tf.add(self.advantage, tf.reduce_max(self.value))

            self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
            self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)

            optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
            self._train = optimizer.minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [-1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        feed = {
            self._X: x_stack,
            self._Y: y_stack
        }
        return self.session.run([self._loss, self._train], feed)

def replay_train(mainDQN, targetDQN, train_batch, batch_size):
    DISCOUNT_RATE = 0.99
    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch])
    rewards = np.array([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch])

    X = states

    Q_value_next_state = mainDQN.predict(next_states)
    evaluated_action = np.argmax(Q_value_next_state, axis=1)
    ev_action = []
    for i in range(batch_size):
        ev_action.append(targetDQN.predict(next_states)[i][evaluated_action[i]])
    Q_target = rewards + DISCOUNT_RATE * np.array(ev_action) * ~done

    y = mainDQN.predict(states)

    y[np.arange(len(X)), actions] = Q_target

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(X, y)


def get_copy_var_ops(*, dest_scope_name, src_scope_name):
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder