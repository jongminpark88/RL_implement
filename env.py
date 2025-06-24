import numpy as np

class Bandit():
    def __init__(self, num):
        self.machine = np.random.rand(num)
    
    def get_machine_prob(self,machine_num):
        return self.machine[machine_num]

    def get_total_prob(self):
        return self.machine

    def action(self,machine_num):
        prob = self.machine[machine_num]
        coin = int(np.random.rand() < prob) #prob가 0.4 == 0.4의 확률로 1을 얻는다. >> 0.4의 확률로 참
        return coin


class GridworldEnv:
    def __init__(self):
        self.grid_size = 4
        self.n_states = 16
        self.terminal_states = [0, 15]
        self.state = None
        
        self.actions = ['up', 'down', 'left', 'right']
        self.action_map = {
            'up': -self.grid_size,
            'down': self.grid_size,
            'left': -1,
            'right': 1
        }

    def reset(self, state=1):
        assert 0 <= state < self.n_states
        self.state = state
        return self.state

    def step(self, action):
        if self.state in self.terminal_states:
            return self.state, 0, True

        row, col = divmod(self.state, self.grid_size)
        next_state = self.state

        if action == 'up' and row > 0:
            next_state -= self.grid_size
        elif action == 'down' and row < self.grid_size - 1:
            next_state += self.grid_size
        elif action == 'left' and col > 0:
            next_state -= 1
        elif action == 'right' and col < self.grid_size - 1:
            next_state += 1
        # else: hit wall → stay in same state

        reward = -1
        done = next_state in self.terminal_states

        self.state = next_state
        return next_state, reward, done

    def get_all_states(self):
        return [s for s in range(self.n_states) if s not in self.terminal_states]

    def get_possible_actions(self, state):
        return self.actions  # All actions always available

    def render(self, values=None):
        for i in range(self.grid_size):
            row = ""
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                if idx in self.terminal_states:
                    row += " T  "
                elif values is not None:
                    row += f"{values[idx]:5.1f}"
                else:
                    row += f"{idx:3d} "
            print(row)


        


