import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class GenericNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(GenericNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.lr = lr
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu:0")
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Agent(object):
    def __init__(
        self, alpha, beta, input_dims, gamma=0.99, l1_size=256, l2_size=256, n_actions=2
    ):
        self.gamma = gamma
        self.log_probs = None
        self.actor = GenericNetwork(alpha, input_dims, l1_size, l2_size, n_actions)
        self.critic = GenericNetwork(beta, input_dims, l1_size, l2_size, n_actions=1)

    def choose_action(self, observation):
        probabilities = F.softmax(self.actor.forward(observation))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)

        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critict_value = self.critic.forward(state)
        critic_value_ = self.critic.forward(new_state)

        delta = (reward + self.gamma * critict_value * (1 - int(done)) - critict_value)

        actor_loss = -(self.log_probs * delta)
        critic_loss = delta ** 2

        (actor_loss + critic_loss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()

