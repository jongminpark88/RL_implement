import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
from collections import deque
import random
import copy



class V_net(nn.Module):
    def __init__(self, state_dim = 11):
        super(V_net, self).__init__()
        self.mlp_V = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
            )
        
    def forward(self,x):
        x = self.mlp_V(x)
        return x


class Q_net(nn.Module):
    def __init__(self, state_dim = 11, action_size = 3):
        super(Q_net, self).__init__()
        self.mlp_Q = nn.Sequential(
            nn.Linear(state_dim + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
            )
        
    def forward(self,state, action):
        x = torch.cat([state, action], dim=1)
        x = self.mlp_Q(x)
        return x


class Policy_net(nn.Module):
    def __init__(self,state_dim = 11, gaussian_parameter = 6):
        super(Policy_net, self).__init__()
        self.mlp_policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, gaussian_parameter),
            #nn.Softmax(dim=1)
            )
        
    def forward(self,x):
        x = self.mlp_policy(x)
        return x

#=========================
#=========================

class Replay_buffer:
    def __init__(self,buffer_size,batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen = self.buffer_size)

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.append([state, action, reward, next_state, done])
    
    def __len__(self):
        return len(self.replay_buffer)
    
    def batch_sampling(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        return batch # 이중 리스트 <- 넘파이로하면, 배열 길이가 달라서 object로 됨

#=========================
#=========================

class SAC:
    def __init__(self):
        self.learning_rate = 3 * 1e-4
        self.gamma = 0.99
        self.buffer_size = int(1e6) 
        self.batch_size = 256
        self.smoothing_coefficient = 0.005
        self.action_size = 3
        self.state_dim = 11
        self.target_interval = 1
        self.step_num = 500000
        self.reward_scale = 5

        self.buffer = Replay_buffer(self.buffer_size, self.batch_size)

        self.V_net = V_net(self.state_dim)
        self.V_target_net = V_net(self.state_dim)
        self.Q_net_1 = Q_net(self.state_dim)
        self.Q_net_2 = Q_net(self.state_dim)
        self.policy_net = Policy_net(self.state_dim)

        self.optimizer_V = optim.Adam(self.V_net.parameters(), lr=self.learning_rate)
        self.optimizer_Q_1 = optim.Adam(self.Q_net_1.parameters(), lr=self.learning_rate)
        self.optimizer_Q_2 = optim.Adam(self.Q_net_2.parameters(), lr=self.learning_rate)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)


    def get_action(self, state):
        # state : batch, state_dim
        # self.policy_net(state) : batch, 6

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

        if state.ndim == 1:
            state = state.unsqueeze(0)

        state_action = self.policy_net(state)


        mean = state_action[:, :3]
        log_std = state_action[:, 3:]
        log_std  = torch.clamp(log_std, -20, 2) 
        std = torch.exp(log_std).clamp(min=1e-6)

        # mean : batch, 3
        # std : batch, 3

        dist = D.Normal(mean, std) 
        z = dist.rsample()                           # (B, 3), grad-tracking
        action = torch.tanh(z)                       # (B, 3)  ⇒  [-1, 1]

        log_prob = dist.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)   # - Σ log(1-tanh² z)
        log_prob = log_prob.sum(dim=-1, keepdim=True)     # (B, 1)

        return action, log_prob
    
    
    def memory_add(self, state, action, reward, next_state, done):
        self.buffer.update(state, action, reward, next_state, done)
    

    def target_V_update(self):
        for target_param, param in zip(self.V_target_net.parameters(), self.V_net.parameters()):
            target_param.data.copy_(
                self.smoothing_coefficient * param.data +
                (1.0 - self.smoothing_coefficient) * target_param.data
            )


    def env_step(self, env, state):
        action_tensor, _ = self.get_action(state) # action shape : 1, 3
        # next_state, reward, done, info = env.step(action)
        # print(state.shape)
        # state = state.squeeze(0)
        action = action_tensor.detach().cpu().numpy().squeeze(0)
        next_state, reward, terminated, trunc, _ = env.step(action)
        done = terminated or trunc

        reward = reward / self.reward_scale

        self.memory_add(state, action, reward, next_state, done)
        #print(reward)

        return next_state, reward, done

    def gradient_update(self):

        # 현재 버퍼가 배치사이즈보다 작음 
        if len(self.buffer) < 256:
            return 
        else:
            batch = self.buffer.batch_sampling()
            # batch shape : batch_size, 5
            # [[state, action, reward, next_state, done].....]
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

            state_batch = torch.from_numpy(np.stack(state_batch).astype(np.float32))            # (B, state_dim)
            action_batch = torch.from_numpy(np.stack(action_batch).astype(np.float32))           # (B, act_dim)
            reward_batch = torch.from_numpy(np.asarray(reward_batch, dtype=np.float32).reshape(-1, 1))  # (B, 1)
            next_state_batch = torch.from_numpy(np.stack(next_state_batch).astype(np.float32))      # (B, state_dim)
            done_batch = torch.from_numpy(np.asarray(done_batch, np.float32).reshape(-1,1))


        # 초기화
        '''
        self.optimizer_V.zero_grad()
        self.optimizer_Q_1.zero_grad()
        self.optimizer_Q_2.zero_grad()
        self.optimizer_policy.zero_grad()
        '''

        #==============
        # loss 설정 : Q
        #==============
        with torch.no_grad():
            next_v_value = self.V_target_net(next_state_batch)
            Q_target = reward_batch + (1 - done_batch) * self.gamma * next_v_value

        Q_1 = self.Q_net_1(state_batch, action_batch)
        Q_2 = self.Q_net_2(state_batch, action_batch)

        loss_Q_1 = 0.5 * (Q_1 - Q_target)**2
        loss_Q_2 = 0.5 * (Q_2 - Q_target)**2

        loss_Q_1 = torch.mean(loss_Q_1, dim = 0)
        loss_Q_2 = torch.mean(loss_Q_2, dim = 0)

        self.optimizer_Q_1.zero_grad()
        loss_Q_1.backward()
        self.optimizer_Q_1.step()

        self.optimizer_Q_2.zero_grad()
        loss_Q_2.backward()
        self.optimizer_Q_2.step()


        #==============
        # loss 설정 : V
        #==============

        # Soft value function 계산용 현재 policy 기반 action 추출
        current_policy_action, current_action_prob = self.get_action(state_batch)
        # current_policy_action : batch, 3
        # current_action_prob : batch, 1
        current_policy_Q_1 = self.Q_net_1(state_batch, current_policy_action)
        current_policy_Q_2 = self.Q_net_2(state_batch, current_policy_action)
        current_policy_Q = torch.min(current_policy_Q_1, current_policy_Q_2)

        # current_policy_Q : batch, 1
        with torch.no_grad():
            V_target = current_policy_Q - current_action_prob 
        # V_target : batch, 1

        V = self.V_net(state_batch)
        # V : batch, 1
        loss_V = 0.5 * (V - V_target)**2
        loss_V = torch.mean(loss_V, dim = 0)

        self.optimizer_V.zero_grad()
        loss_V.backward()
        self.optimizer_V.step()


        #==============
        # loss 설정 : policy
        #==============
        loss_policy = current_action_prob - current_policy_Q
        loss_policy = torch.mean(loss_policy, dim = 0)

        self.optimizer_policy.zero_grad()
        loss_policy.backward()
        self.optimizer_policy.step()



    
    def main_iteration(self,env,state):
        # done 될때까지 반복
            # env step
            # gradient step 한번
            # if 간격 확인
                # target_v 업데이트
        
        step = 1
        episode_num = 1
        episode_reward = 0
        final_list = []

        while step < self.step_num :
            next_state, reward, done = self.env_step(env, state)
            self.gradient_update()

            if step % self.target_interval == 0:
                self.target_V_update()
            
            episode_reward += reward

            if done:
                final_list.append([episode_num,episode_reward,step])
                episode_num += 1
                episode_reward = 0
                state,_ = env.reset(seed=8)
            
            else:
                state = next_state

            print(step)
            step += 1
             

        return final_list



        
    





        


