import numpy as np
import gym

import matplotlib.pyplot as plt

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.input_dims = input_dims
        self.lr = lr
        self.n_actions=n_actions
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims,self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self.to(self.device)


    def forward(self,observation):
        state = T.tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Agent(object):
    def __init__(self, lr, input_dims, gamma=0.99, n_actions = 4, l1_size=256, l2_size = 256):

        self.gamma=gamma
        self.reward_memory=[]
        self.action_memory=[]
        self.policy = PolicyNetwork(lr,input_dims,l1_size,l2_size,n_actions)

    def choose_action(self, observation):
        probabilities = F.softmax(self.policy.forward(observation))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()
        G = np.zeros_like(self.reward_memory, dtype = np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k]*discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G-mean)/std

        G = T.tensor(G, dtype=T.float).to(self.policy.device)

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g*logprob

        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []


env = gym.make('LunarLander-v2')
agent = Agent(lr=0.001, input_dims=[8], gamma = 0.99, n_actions=4,l1_size=128,l2_size=128)

score_history = []
score = 0
n_episodes = 2500

for i in range(n_episodes):
    print('episode: ',i,' score: ', score)
    done =False
    score = 0
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.store_rewards(reward)
        observation = observation_
        score += reward
    score_history.append(score)
    agent.learn()


plt.plot(score_history)
plt.show()

