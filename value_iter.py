import numpy as np
from policy_iter import policy_improvement


def value_iteration(env, V, policy, threshhold=0.0001,gamma = 0.99):

    # V : n_state
    # policy : n_state, n_action
    n_actions = len(env.actions)

    while True:
        delta = 0
        for state in range(V.shape[0]):
            if state in env.terminal_states:
                continue

            value_list = np.zeros((n_actions))
            for action in range(n_actions):
                next_state, reward, done = env.step(state,action)

                if done:
                    V[next_state]=0

                value_list[action] = reward + gamma * V[next_state]
    
            delta = max(delta, abs(max(value_list) - V[state]))
            V[state] = max(value_list)
        
        if delta < threshhold:
            break
    
    final_policy = policy_improvement(env, V, policy, gamma = gamma)
    env.render(V)

    
    return V, final_policy