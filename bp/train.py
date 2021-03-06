import argparse

import numpy as np

import torch


from env import Env
from model import *


import matplotlib.pyplot as plt


def save_args(TRAIN_NO, args):
    args_name = ["alpha", "ganna", "action_repeat",
                 "img_stack", "seed", "render", "log_iterval"]
    with open("data/args/"+str(TRAIN_NO)+'args.txt', 'w') as f:
        f.write(str(args))


parser = argparse.ArgumentParser(
    description='Bachelors project Andrej Petricko PPO Reinforcement Learning')
parser.add_argument('--alpha', type=float, default=1e-3,
                    metavar='A', help='learning rate(default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8,
                    metavar='', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4,
                    metavar='', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0,
                    metavar='', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--nn-type', type=int, default=0, metavar='',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


NUM_EPISODES = 100000
MAX_STEPS = 2000

TRAIN_NO = 14


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


if __name__ == "__main__":
    save_args(TRAIN_NO, args)

    agent = Agent(alpha=args.alpha, gamma=args.gamma,
                  img_stack=args.img_stack, nn_type=args.nn_type)
    env = Env(seed=args.seed, action_repeat=args.action_repeat,
              img_stack=args.img_stack)
    agent.load_param(name="13_score_460.17260382805284episode_2522_progres_save_params")
    running_score = 0
    agent.load_param("ppo_net_params")
    state = env.reset()
    max_running_score = 100
    score_history = []
    running_score_history = []
    for episode in range(NUM_EPISODES):
        score = 0
        state = env.reset()

        for t in range(MAX_STEPS):
            action, a_logp = agent.select_action(state)
            new_state, reward, done, die = env.step(
                action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            if agent.store((state, action, a_logp, reward, new_state)):
                print('updating')
                agent.update()
            score += reward
            state = new_state
            if done or die:
                break

        running_score = running_score * 0.99 + score * 0.01
        running_score_history.append(running_score)
        score_history.append(score)

        if episode % args.log_interval == 0:
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(
                episode, score, running_score))
            name = str(TRAIN_NO) + "_automatic_save_params"
            agent.save_param(name=name)
            with open("data/score_history/"+str(TRAIN_NO)+'_score_history.txt', 'w') as f:
                f.seek(0)
                f.truncate
                f.writelines("%f\n" % score for score in score_history)
            with open("data/score_history/"+str(TRAIN_NO)+'_runnig_history.txt', 'w') as f:
                f.seek(0)
                f.truncate
                f.writelines("%f\n" % score for score in running_score_history)
        if running_score > max_running_score:
            max_running_score += 10
            name = str(TRAIN_NO)+"_"+"score_"+str(running_score) + \
                "episode_"+str(episode)+"_progres_save_params"
            agent.save_param(name=name)

        if running_score > env.reward_threshold+50:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(
                running_score, score))
            with open("data/score_history/"+str(TRAIN_NO)+'_score_history.txt', 'w') as f:
                f.seek(0)
                f.truncate
                f.writelines("%f\n" % score for score in score_history)
            with open("data/score_history/"+str(TRAIN_NO)+'_runnig_history.txt', 'w') as f:
                f.seek(0)
                f.truncate
                f.writelines("%f\n" % score for score in running_score_history)
            break
