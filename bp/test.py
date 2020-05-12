import argparse

import numpy as np

import gym
import torch
import torch.nn as nn


from model import Agent
from env import Env


parser = argparse.ArgumentParser(
    description='Bachelors project Andrej Petricko PPO Reinforcement Learning')
parser.add_argument('--alpha', type=float, default=1e-3,
                    metavar='G', help='discount factor (default: 0.001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8,
                    metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4,
                    metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0,
                    metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--nn-type', type=int, default=0, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)


if __name__ == "__main__":
    agent = Agent(alpha=args.alpha, gamma=args.gamma,
                  img_stack=args.img_stack, nn_type=args.nn_type)
    env = Env(seed=args.seed, action_repeat=args.action_repeat,
              img_stack=args.img_stack)

    agent.load_param(
        name='ppo_net_params')
    test_records = []
    running_score = 0
    state = env.reset()
    for i_ep in range(100):
        score = 0
        state = env.reset()

        for t in range(100000):
            action, _ = agent.select_action(state)
            print(action)
            state_, reward, done, die = env.step(
                action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            score += reward
            state = state_
            if done or die:
                break

        test_records.append(score)
        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
    print(np.mean(test))
