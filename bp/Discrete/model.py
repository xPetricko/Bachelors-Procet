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
    buffer_capacity, batch_size = 2000,200

    def __init__(self, alpha, gamma, img_stack, nn_type):
        self.alpha = alpha
        self.gamma = gamma
        self.img_stack = img_stack

        self.transition = np.dtype([('s', np.float64, (self.img_stack, 96, 96)), ('a', np.float64), ('d', np.float64),
                                    ('r', np.float64), ('s_n', np.float64, (self.img_stack, 96, 96))])
        self.training_step = 0
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.net = Net(alpha=self.alpha, gamma=self.gamma,
                        img_stack=self.img_stack).double().to(self.device)
        self.buffer = np.empty(self.buffer_capacity, dtype=self.transition)
        self.counter = 0

    def select_action(self, state):
        state = T.from_numpy(state).double().to(self.device).unsqueeze(0)
        actions = self.net(state)[0]
        
        actions = F.softmax(actions,dim=0)
        probs = T.distributions.Categorical(actions)

        action = probs.sample()

        return action.cpu().numpy()

    def save_param(self, name):
        T.save(self.net.state_dict(), 'data/param/'+name+'params.pkl')

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1

    def reset_memory(self):
        self.transition = np.dtype([('s', np.float64, (self.img_stack, 96, 96)), ('a', np.float64), ('d', np.float64),
                                    ('r', np.float64), ('s_n', np.float64, (self.img_stack, 96, 96))])
        self.counter = 0
        

    def load_param(name):
        self.net.load_state_dict(T.load('data/param/"'+name+'.pkl'))

    def update(self):
        
        asd = 0
        for x in self.buffer:
            if x==None:
                print("None")
            asd+=1

        print(asd)
        s = T.tensor(self.buffer['s'], dtype=T.double).to(self.device)
        a = T.tensor(self.buffer['a'], dtype=T.double).to(
            self.device).view(-1, 1)
        r = T.tensor(self.buffer['r'], dtype=T.double).to(
            self.device)
        s_n = T.tensor(self.buffer['s_n'], dtype=T.double).to(self.device)

        d = T.tensor(self.buffer['d'], dtype=T.double).to(
            self.device).view(-1, 1)

        d_r = [T.sum(T.FloatTensor([self.gamma**i for i in range(r[j:].size(0))])\
             * r[j:]) for j in range(r.size(0))]  

        target_v = r.view(-1, 1) + T.FloatTensor(d_r).view(-1, 1).to(self.device)

        logits, v = self.net(s)
        dists = F.softmax(logits, dim=1)
        probs = T.distributions.Categorical(dists)

        value_loss = F.mse_loss(v, target_v.detach())

        entropy = []
        for dist in dists:
            entropy.append(-T.sum(dist.mean() * T.log(dist)))
        entropy = T.stack(entropy).sum()

        advantage = target_v - v
        policy_loss = -probs.log_prob(a.view(a.size(0))).view(-1, 1) * advantage.detach()
        policy_loss = policy_loss.mean()
        
        loss = policy_loss + value_loss - 0.001 * entropy 

        self.net.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.net.optimizer.step()
