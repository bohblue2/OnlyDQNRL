import numpy as np
import tensorflow as tf
import gym
from A2C import Actor, Critic

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
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

for i_episode in range(MAX_EPISODE):
    state = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        action = actor.choose_action(state)

        next_state, reward, done, _ = env.step(action)
        if done: reward = -20

        track_r.append(reward)

        td_error = critic.learn(state, reward, next_state)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(state, action, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        state = next_state
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break
