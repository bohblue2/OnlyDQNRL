import numpy as np

def per_sample(mainDQN, targetDQN, replay_buffer, BATCH_SIZE):
    DISCOUNT_RATE = 0.99
    states = np.vstack([x[0] for x in replay_buffer])
    actions = np.array([x[1] for x in replay_buffer])
    rewards = np.array([x[2] for x in replay_buffer])
    next_states = np.vstack([x[3] for x in replay_buffer])
    done = np.array([x[4] for x in replay_buffer])

    X = states
    Q_target = rewards + DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1) * ~done

    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target

    T_s_a = Q_target
    Q_s_a = mainDQN.predict(X)
    Q_s_a = Q_s_a[np.arange(len(X)),actions]
    error = []
    for i in range(len(X)):
        error.append(abs(2*(T_s_a[i]-Q_s_a[i])))
    error = np.array(error)
    small_p = (error+1e-7)**0.99
    sum_small_p = np.sum(small_p)

    large_p = small_p / sum_small_p

    selected_index = np.random.choice(np.arange(len(X)), BATCH_SIZE, p=large_p)
    sampled_list = []
    for i in range(len(selected_index)):
        sampled_list.append(replay_buffer[selected_index[i]])
    return sampled_list