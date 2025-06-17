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

