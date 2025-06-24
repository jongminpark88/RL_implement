import numpy as np

def policy_evaluation(env, V, policy, threshhold=0.0001,gamma = 0.99):

    # V : n_state
    # policy : n_state, n_action

    while True:
        delta = 0
        for state_index, state_value in enumerate(V):
            if state_index in env.terminal_states:
                continue

            value_sum = 0
            for action_index, action_prob in enumerate(policy[state_index]):
                next_state_index, reward, done = env.step(state_index,action_index)

                if done:
                    V[next_state_index]=0

                value_sum += action_prob * (reward + gamma * V[next_state_index])
            delta = max(value_sum - V[state_index],delta) 
            V[state_index] = value_sum
        
        if delta < threshhold:
            break
    
    return V


def policy_improvement(env, V, policy, gamma=0.99):
    n_actions = len(env.actions)
    n_states = V.shape[0]
    new_policy = np.zeros((n_states, n_actions))

    for state in range(n_states):
        action_values = np.zeros(n_actions)

        for action in range(n_actions):
            next_state, reward, done = env.step(state, action)
            action_values[action] = reward + gamma * V[next_state]

        best_action = np.argmax(action_values)
        new_policy[state, best_action] = 1

    return new_policy


def policy_iteration(env, V, policy, threshhold=0.0001,gamma = 0.99):
    policy_stable = False
    i=1
    while not(policy_stable):
        print(i)
        V = policy_evaluation(env, V, policy, threshhold=threshhold,gamma = gamma)
        env.render(V)
        new_policy = policy_improvement(env, V, policy, gamma = gamma)
        policy_stable = np.array_equal(new_policy, policy)
        policy = new_policy
        #env.render_policy(policy)
        i+=1

    return V, policy


        
            
            

        
                




