import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from nets import Net, NetMP


class Agent():
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 200, 128

    def __init__(self, alpha, gamma, img_stack, nn_type):
        self.alpha = alpha
        self.gamma = gamma
        self.img_stack = img_stack

        self.transition = np.dtype([('s', np.float64, (self.img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                                    ('r', np.float64), ('s_n', np.float64, (self.img_stack, 96, 96))])
        self.training_step = 0
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        if nn_type == 0:
            self.net = Net(alpha=self.alpha, gamma=self.gamma,
                           img_stack=self.img_stack).double().to(self.device)
        else:
            self.net = NetMP(alpha=self.alpha, gamma=self.gamma,
                             img_stack=self.img_stack).double().to(self.device)

        self.buffer = np.empty(self.buffer_capacity, dtype=self.transition)
        self.counter = 0

    def select_action(self, state):
        state = T.from_numpy(state).double().to(self.device).unsqueeze(0)
        with T.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)
        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def save_param(self, name):
        T.save(self.net.state_dict(), 'data/param/'+name+'params.pkl')

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def load_param(name):
        self.net.load_state_dict(torch.load('data/param/"'+name+'.pkl'))

    def update(self):
        self.training_step += 1

        s = T.tensor(self.buffer['s'], dtype=T.double).to(self.device)
        a = T.tensor(self.buffer['a'], dtype=T.double).to(self.device)
        r = T.tensor(self.buffer['r'], dtype=T.double).to(self.device).view(-1, 1)
        s_n = T.tensor(self.buffer['s_n'], dtype=T.double).to(self.device)

        old_a_logp = T.tensor(self.buffer['a_logp'], dtype=T.double).to(
            self.device).view(-1, 1)

        with T.no_grad():
            target_v = r + self.gamma * self.net(s_n)[1]
            adv = target_v - self.net(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8) #optional

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                print(index)
                print(dist.log_prob(a[index]))
                print(dist.log_prob(a[index]).sum(dim=1, keepdim=True))
                print(old_a_logp[index])
                input()
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = T.exp(a_logp - old_a_logp[index])
             
                surr1 = ratio * adv[index]
                surr2 = T.clamp(ratio, 1.0 - self.clip_param,
                                1.0 + self.clip_param) * adv[index]
                action_loss = -T.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(
                    self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.net.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm) #optional
                self.net.optimizer.step()

    def update_vanila(self):
        self.training_step += 1

        s = T.tensor(self.buffer['s'], dtype=T.double).to(self.device)
        a = T.tensor(self.buffer['a'], dtype=T.double).to(self.device)
        r = T.tensor(self.buffer['r'], dtype=T.double).to(
            self.device).view(-1, 1)
        s_n = T.tensor(self.buffer['s_n'], dtype=T.double).to(self.device)

        for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

            (alpha, beta), critic_value = self.net(s[index])
            _, critic_value_ = self.net(s_n[index])
            a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)

            delta = r[index] + self.gamma*critic_value_ - critic_value

            actor_loss = self.log_probs * delta
            critic_loss = delta**2

            loss = actor_loss + critic_loss

            self.net.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm) #optional
            self.net.optimizer.step()
