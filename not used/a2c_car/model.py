import math

import numpy as np
from torch.autograd import Variable

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



    # def save_model(self,PATH):
    #     T.save(self, PATH)

    # def load_model(PATH):
    #     model = T.load(PATH)
    #     model.eval()

    #     return model


# class ActorCritic(nn.Module):
#     def __init__(self,num_inputs,num_actions,alpha )
#         super(ActorCriticRacing, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))


class ActorCritic(nn.Module):

    def __init__(self, alpha, n_inputs, n_actions):
        super(ActorCritic, self).__init__()
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.alpha =alpha
        
        

        self.conv1 = nn.Conv2d(self.n_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        input_dims = self.calc_input_dims()
        
        self.critic_linear = nn.Linear(input_dims, 1)
        self.actor_linear = nn.Linear(input_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.train()


    def calc_input_dims(self):

        data = T.zeros((1,1,32,32))
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)

        return int(np.prod(data.size()))

    def forward(self, observation):
        observation = T.from_numpy(np.reshape(observation,(1,1,32,32)))
        state = T.tensor(observation).to(self.device)

        x = F.elu(self.conv1(state))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
  
        x = x.view(-1, 32 * 2 * 2)
        return self.critic_linear(x), self.actor_linear(x)


class NewAgent(object):
    def __init__(self, n_inputs, n_actions,alpha=0.0001 ,gamma=0.99):
        self.gamma = gamma
        self.actor_critic = ActorCritic(alpha = alpha,n_inputs = n_inputs, n_actions = n_actions)
        self.log_probs = None

    def choose_action(self,observation):
        policy, _ = self.actor_critic.forward(observation)
        policy = F.softmax(policy)
        action_probs = T.distributions.Categorical(policy)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)

        return action.item()

    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()

        _, critic_value = self.actor_critic.forward(state)
        _, critic_value_ = self.actor_critic.forward(state_)

        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)
        delta = reward + self.gamma*critic_value_*(1-int(done)) -critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta ** 2

        (actor_loss + critic_loss).mean().backward()
        self.actor_critic.optimizer.step()
