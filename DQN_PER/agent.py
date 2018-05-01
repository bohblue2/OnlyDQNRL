import numpy as np
import tensorflow as tf
import random
from collections import deque
from dqn import DQN, get_copy_var_ops, replay_train
from per import per_sample

import gym

env = gym.make('CartPole-v0')

# Constants defining our neural network
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n


REPLAY_MEMORY = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5
MAX_EPISODES = 1000

r = tf.placeholder(tf.float32)  ########
rr = tf.summary.scalar('reward', r)
merged = tf.summary.merge_all()  ########

def main():
    replay_buffer = deque(maxlen=REPLAY_MEMORY)

    with tf.Session() as sess:
        mainDQN = DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        sess.run(tf.global_variables_initializer())

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

        #writer = tf.summary.FileWriter('./board/dqn', sess.graph)  ########
        writer = tf.summary.FileWriter('./board/dqn_per', sess.graph)  ########

        for episode in range(MAX_EPISODES):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            state = env.reset()

            while not done:
                if np.random.rand() < e:
                    action = env.action_space.sample()
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)

                if done:  # Penalty
                    reward = -1

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))
                if done:
                    if len(replay_buffer) > BATCH_SIZE:
                        #minibatch = random.sample(replay_buffer, BATCH_SIZE)                   # only_DQN
                        minibatch = per_sample(mainDQN, targetDQN, replay_buffer, BATCH_SIZE)   # per_DQN
                        #print(minibatch)
                        loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                    if step_count % TARGET_UPDATE_FREQUENCY == 0:
                        sess.run(copy_ops)
                    summary = sess.run(merged, feed_dict={r: step_count})
                    writer.add_summary(summary, episode)
                state = next_state
                step_count += 1

            print("Episode: {}  steps: {}".format(episode, step_count))


if __name__ == "__main__":
    main()