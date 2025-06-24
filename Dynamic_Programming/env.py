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
        self.n_states = self.grid_size ** 2          # 16
        self.terminal_states = [0, 15]
        self.state = None

        # 0: up, 1: down, 2: left, 3: right
        self.actions = [0, 1, 2, 3]
        self.action_map = {
            0: -self.grid_size,   # up
            1:  self.grid_size,   # down
            2: -1,                # left
            3:  1                 # right
        }
        self.action_symbols = ['↑', '↓', '←', '→']

    def reset(self, state: int = 1):
        """환경 초기화 후 현재 상태 반환"""
        assert 0 <= state < self.n_states
        self.state = state
        return self.state

    def step(self, state: int, action: int):
        """
        외부에서 넘겨준 현재 상태 `state`와 `action`을 받아
        다음 상태, 보상, 종료 여부를 반환.
        Args:
            state  (int): 현재 상태 인덱스
            action (int): {0:up, 1:down, 2:left, 3:right}
        Returns:
            next_state (int), reward (float), done (bool)
        """
        # 터미널에서 행동하면 그대로 종료
        if state in self.terminal_states:
            return state, 0.0, True

        row, col = divmod(state, self.grid_size)
        move = self.action_map.get(action, 0)
        next_state = state + move

        # 경계(벽) 체크: 벽이면 이동 무효
        if action == 0 and row == 0:                          # 위쪽 벽
            next_state = state
        elif action == 1 and row == self.grid_size - 1:       # 아래쪽 벽
            next_state = state
        elif action == 2 and col == 0:                        # 왼쪽 벽
            next_state = state
        elif action == 3 and col == self.grid_size - 1:       # 오른쪽 벽
            next_state = state

        reward = -1.0
        done = next_state in self.terminal_states

        # 내부 상태도 업데이트(선택 사항)
        self.state = next_state
        return next_state, reward, done

    def get_all_states(self):
        return [s for s in range(self.n_states) if s not in self.terminal_states]

    def get_possible_actions(self, state: int):
        return self.actions  # 네 방향 모두 시도 가능

    def render(self, values=None):
        for i in range(self.grid_size):
            row_str = ""
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                if idx in self.terminal_states:
                    row_str += "  T  "
                elif values is not None:
                    row_str += f"{values[idx]:5.1f}"
                else:
                    row_str += f"{idx:3d} "
            print(row_str)

    def render_policy(self, policy):
        """
        policy: np.array of shape [n_states, n_actions] — 확률분포 or one-hot
        """
        for i in range(self.grid_size):
            row_str = ""
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                if idx in self.terminal_states:
                    row_str += "  T  "
                else:
                    # 정책이 결정적(one-hot)일 경우
                    action_index = int(np.argmax(policy[idx]))
                    row_str += f"  {self.action_symbols[action_index]}  "
            print(row_str)



        


