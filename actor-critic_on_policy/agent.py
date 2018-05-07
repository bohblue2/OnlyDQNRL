import numpy as np
import tensorflow as tf
import gym
from A2C import Actor, Critic

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 500
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')

sess = tf.Session()

actor = Actor(sess, n_features=4, n_actions=2, lr=0.001)
critic = Critic(sess, n_features=4, lr=0.001)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

spend_time = tf.placeholder(tf.float32)
rr = tf.summary.scalar('reward', spend_time)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./board/a2c_onpolicy_500', sess.graph)

for i_episode in range(MAX_EPISODE):
    state = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()
        try:
            action = actor.choose_action(state)
        except:
            action = 0
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -20
        else:
            reward = 0

        track_r.append(reward)

        td_error = critic.learn(state, reward, next_state)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(state, action, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        state = next_state
        t += 1

        if done:
            summary = sess.run(merged, feed_dict={spend_time: t})
            writer.add_summary(summary, i_episode)
            print("episode:", i_episode, "  reward:", t)
            break
