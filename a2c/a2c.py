import torch
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical

from nets import Net
import numpy as np

class A2CAgent():

    def __init__(self, env, gamma, lr):
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

        self.action_dim = 5
        
        self.gamma = gamma
        self.lr = lr
        
        self.model = Net(gamma, 4, lr).float().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        logits, _ = self.model(state)
        dist = F.softmax(logits, dim=0)
        probs = Categorical(dist)

        return probs.sample().cpu().detach().item()
    
    def compute_loss(self, trajectory):
        states = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
        actions = torch.LongTensor([sars[1] for sars in trajectory]).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
        next_states = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
        dones = torch.FloatTensor([sars[4] for sars in trajectory]).view(-1, 1).to(self.device)
        
        # compute discounted rewards
        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[j:].size(0))])\
             * rewards[j:]) for j in range(rewards.size(0))] 
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)
        
        logits, values = self.model.forward(states)
        dists = F.softmax(logits, dim=1)
        probs = Categorical(dists)
        
        # compute value loss
        value_loss = F.mse_loss(values, value_targets.detach())
        
        
        # compute entropy bonus
        entropy = []
        for dist in dists:
            entropy.append(-torch.sum(dist.mean() * torch.log(dist)))
        entropy = torch.stack(entropy).sum()
        
        # compute policy loss
        advantage = value_targets - values
        policy_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantage.detach()
        policy_loss = policy_loss.mean()
        
        total_loss = policy_loss + value_loss - 0.001 * entropy 
        return total_loss
        
    def update(self, trajectory):
        loss = self.compute_loss(trajectory)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()