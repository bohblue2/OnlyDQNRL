import numpy as np
import tensorflow as tf
import random
from collections import deque
from dueling_dqn import DQN, get_copy_var_ops, replay_train

import gym

env = gym.make('CartPole-v0')

# Constants defining our neural network
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n


REPLAY_MEMORY = 50000
BATCH_SIZE = 1024
TARGET_UPDATE_FREQUENCY = 5
MAX_EPISODES = 300

def main():
    replay_buffer = deque(maxlen=REPLAY_MEMORY)

    last_100_game_reward = deque(maxlen=100)

    with tf.Session() as sess:
        mainDQN = DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main", mode="mean")
        targetDQN = DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target", mode="mean")
        sess.run(tf.global_variables_initializer())

        spend_time = tf.placeholder(tf.float32)
        rr = tf.summary.scalar('reward', spend_time)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./board/duel_dqn', sess.graph)

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

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
                        minibatch = random.sample(replay_buffer, BATCH_SIZE)
                        loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                        sess.run(copy_ops)
                    #if step_count % TARGET_UPDATE_FREQUENCY == 0:
                    #    sess.run(copy_ops)
                    summary = sess.run(merged, feed_dict={spend_time: step_count})
                    writer.add_summary(summary, episode)

                state = next_state
                step_count += 1

            print("Episode: {}  steps: {}".format(episode, step_count))


if __name__ == "__main__":
    main()