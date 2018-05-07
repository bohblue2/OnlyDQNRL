import numpy as np
import tensorflow as tf
import gym
from pg import pg, train, discount_rewards
X = tf.placeholder(tf.float32, [None, 4], name="input_x")
Y = tf.placeholder(tf.float32, [None, 2], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

action_pred = pg(X)
train = train(action_pred, Y, advantages)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
env = gym.make('CartPole-v0')

spend_time = tf.placeholder(tf.float32)
rr = tf.summary.scalar('reward', spend_time)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./board/pg_sparse_reward', sess.graph)

for i in range(500):
    xs = np.empty(shape=[0, 4])
    ys = np.empty(shape=[0, 2])
    rewards = np.empty(shape=[0, 1])
    score = 0
    state = env.reset()
    while True:
        x = np.reshape(state, [1, 4])
        action_prob = sess.run(action_pred, feed_dict={X: x})
        action = np.random.choice(np.arange(2), p=action_prob[0])
        xs = np.vstack([xs, x])
        y = np.zeros(2)
        y[action] = 1
        ys = np.vstack([ys, y])
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            reward = -20
        else:
            reward = 0
        rewards = np.vstack([rewards, reward])
        
        if done:
            # Determine standardized rewards
            discounted_rewards = discount_rewards(rewards, 0.99)
            # Normalization
            discounted_rewards = (discounted_rewards - discounted_rewards.mean())/(discounted_rewards.std() + 1e-7)
            _ = sess.run(train, feed_dict={X: xs,Y: ys,advantages: discounted_rewards})
            summary = sess.run(merged, feed_dict={spend_time: score})
            writer.add_summary(summary, i)
            print(i, score)
            break