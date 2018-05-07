import numpy as np
import tensorflow as tf
from collections import deque
import gym
from a2c import Actor, Critic, discount_rewards

OUTPUT_GRAPH = False
MAX_EPISODE = 500
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')

sess = tf.Session()

action_number = 2
feature_number = 4

actor = Actor(sess, n_features=feature_number, n_actions=action_number, lr=0.001)
critic = Critic(sess, n_features=feature_number, lr=0.001)     # we need a good teacher, so the teacher should learn faster than the actor
spend_time = tf.placeholder(tf.float32)
rr = tf.summary.scalar('reward', spend_time)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./board/a2c_offpolicy_500', sess.graph)
sess.run(tf.global_variables_initializer())

for i_episode in range(MAX_EPISODE):
    state = env.reset()
    t = 0
    track_r = []
    states = np.empty(shape=[0,feature_number])
    next_states = np.empty(shape=[0,feature_number])
    rewards = np.empty(shape=[0,1])
    actions = np.empty(shape=[0,action_number])
    while True:

        predict_actor = actor.choose_action(state)

        next_state, reward, done, _ = env.step(predict_actor)

        if done:
            reward = -20
        else:
            reward = 0
        states = np.vstack([states, state])
        next_states = np.vstack([next_states, next_state])
        rewards = np.vstack([rewards, reward])
        action = np.zeros(action_number)
        action[predict_actor] = 1
        actions = np.vstack([actions, action])
        state = next_state
        t += 1
        
        if done:
            discounted_rewards = discount_rewards(rewards)
            td_error = critic.learn(states, discounted_rewards, next_states)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(states, actions, td_error)     # true_gradient = grad[logPi(s,a) * td_error]
            summary = sess.run(merged, feed_dict={spend_time: t})
            writer.add_summary(summary, i_episode)
            print("episode:", i_episode, "  reward:", t)
            break