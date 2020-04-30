import math

import numpy as np
from torch.autograd import Variable

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




def normalized_columns_initializer(weights, std=1.0):
    out = T.randn(weights.size())
    out *= std / T.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.swqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]

        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)









class ActorCritticNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims,fc2_dims, n_actions):
        super(ActorCritticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.pi = nn.Linear(self.fc2_dims, n_actions)
        self.v = nn.Linear(self.fc2_dims,1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)

        return pi,v

    def save_model(self,PATH):
        T.save(self, PATH)

    def load_model(PATH):
        model = T.load(PATH)
        model.eval()

        return model


class ActorCritic(nn.Module):
    def __init__(self,num_inputs,num_actions,alpha )
        super(ActorCriticRacing, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))


class _ActorCritic(nn.Module):

    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm_size = 64

        # self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
        self.lstm = nn.LSTMCell(32 * 2 * 2, self.lstm_size)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(self.lstm_size, 1)
        self.actor_linear = nn.Linear(self.lstm_size, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, ):
        inputs, (hx, cx) = inputs
        # print (inputs.size())
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        # print (x.size())
        # x = x.view(-1, 32 * 3 * 3)
        x = x.view(-1, 32 * 2 * 2)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)



class NewAgent(object):
    def __init__(self,alpha, input_dims, gamma=0.99, layer1_size=256,
                layer2_size=256, n_actions=2):
        self.gamma = gamma
        self.actor_critic = ActorCritticNetwork(alpha,input_dims,layer1_size,layer2_size,n_actions = n_actions)
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

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()
