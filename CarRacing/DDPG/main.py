import numpy as np

import gym
from gym.spaces import Box, Discrete

import env as E
from my_func import *

from model import Agent

import matplotlib.pyplot as plt


env = E.CarRacing()
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=1, tau=0.001, observation_shape=(1,32,32), env=env,
                batch_size=64, layer1_size=400, layer2_size=300, n_actions=5)

agent.load_models()
np.random.seed(0)

score_history = []
for i in range(750):
    done = False
    score = 0
    obs = env.reset()
    obs = state_preproces(obs)
    obs = np.reshape(obs, (1,1,32,32))

    for j in range(1000):
        if done:
            break
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step_discrete(act)
        new_state = state_preproces(new_state)
        new_state = np.reshape(new_state, (1,1,32,32))
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score+= reward
        obs = new_state

    score_history.append(score)
    print('episode ', i, ' score %.2f' % score," steps: ",j, '100 game average %.2f' %np.mean(score_history[-100:]))
    plt.plot(score_history)
    plt.xlabel("episode")
    plt.ylabel("score")

    if i % 25 ==0:
        agent.save_models()
        plt.savefig('ddpg.png')



    
































# env = E.CarRacing()

# agent = Agent(lr=0.001, input_dims=1, gamma = 0.99, n_actions=5,l1_size=128,l2_size=128)
# score_history = []
# score = 0
# n_episodes = 250



# for i in range(n_episodes):
#     print('episode: ',i,' score: ', score)
#     done =False
#     score = 0
#     observation = env.reset()
#     observation = state_preproces(observation)
#     observation = np.reshape(observation,(1,1,32,32))
#     for _ in range(750):
#         if done:
#             break
#         action = agent.choose_action(observation)
#         observation_, reward, done, info = env.step_discrete(action)
#         observation_= state_preproces(observation_)
#         observation_ = np.reshape(observation_,(1,1,32,32))
#         agent.store_rewards(reward)
#         observation = observation_
#         score += reward
#     score_history.append(score)
#     agent.learn()

# plt.plot(score_history)
# plt.show()

    



