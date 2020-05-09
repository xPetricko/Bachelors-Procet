import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from nets import Net


class Agent():
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 200

    def __init__(self, alpha, gamma, img_stack, nn_type):
        self.alpha = alpha
        self.gamma = gamma
        self.img_stack = img_stack

        # self.transition = np.dtype([('s', np.float64, (self.img_stack, 96, 96)), ('a', np.float64), ('d', np.float64),
        #                             ('r', np.float64), ('s_n', np.float64, (self.img_stack, 96, 96))])
        self.transition = []
        self.training_step = 0
        self.device = T.device("cpu" if T.cuda.is_available() else "cpu")
        self.net = Net(alpha=self.alpha, gamma=self.gamma,
                       img_stack=self.img_stack).float().to(self.device)
        self.buffer = np.empty(self.buffer_capacity, dtype=self.transition)
        self.counter = 0

    def select_action(self, state):
        state = T.from_numpy(state).float().to(self.device).unsqueeze(0)
        actions,v = self.net(state)

        actions = F.softmax(actions, dim=0)
        probs = T.distributions.Categorical(actions)

        action = probs.sample()
        a_logp = probs.log_prob(action)

        

        return action.cpu().numpy(), a_logp,v

    def save_param(self, name):
        T.save(self.net.state_dict(), 'data/param/'+name+'params.pkl')

    def store(self, transition):
        self.transition.append(transition)

    def reset_memory(self):
        self.transition = []

    def load_param(name):
        self.net.load_state_dict(T.load('data/param/"'+name+'.pkl'))

    def update(self):

        
        act = [sars[0] for sars in self.transition]
        rew = [sars[1] for sars in self.transition]
        logprobs = [sars[2] for sars in self.transition]
        state_val = [sars[3] for sars in self.transition]

         # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in rew[::-1]:
            dis_reward = reward + self.gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
        rewards = T.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, value, reward in zip(logprobs, state_val , rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)   

        print(loss)
        self.net.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.net.optimizer.step()
