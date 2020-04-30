import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class ActorCriticNet(nn.Module):
    def __init__(self,gamma, img_stack,alpha=1e-3):
        super(ActorCriticNet, self).__init__()
        self.img_stack = img_stack
        self.gamma = gamma


        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(self.img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v


class Agent():
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self,alpha, gamma, img_stack):
        self.alpha = alpha
        self.gamma = gamma
        self.img_stack = img_stack

        self.transition = np.dtype([('s', np.float64, (self.img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (self.img_stack, 96, 96))])


        
        self.training_step = 0
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.actor_critic = ActorCriticNet(alpha=self.alpha, gamma=self.gamma, img_stack=self.img_stack).double().to(self.device)
        self.buffer = np.empty(self.buffer_capacity, dtype=self.transition)
        self.counter = 0



    def select_action(self, state):
        state = T.from_numpy(state).double().to(self.device).unsqueeze(0)
        with T.no_grad():
            alpha, beta = self.actor_critic(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def save_param(self, name):
        T.save(self.actor_critic.state_dict(), 'data/param/'+name+'params.pkl')
    
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
        s_ = T.tensor(self.buffer['s_'], dtype=T.double).to(self.device)

        old_a_logp = T.tensor(self.buffer['a_logp'], dtype=T.double).to(self.device).view(-1, 1)

        with T.no_grad():
            target_v = r + self.gamma * self.actor_critic(s_)[1]
            adv = target_v - self.actor_critic(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8) #optional

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                alpha, beta = self.actor_critic(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = T.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = T.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -T.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.actor_critic(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.actor_critic.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm) #optional
                self.actor_critic.optimizer.step()
