

from a2c import NewAgent
import env as E
from my_func import *
from model import *


import matplotlib.pyplot as plt
import gym


OBSERVATION_SIZE = 32
NUM_EPISODES = 1000

if __name__ == '__main__':
    E.STATE_W = E.STATE_H = OBSERVATION_SIZE

    agent = None #TODO!!!
    env = E.CarRacing()

    score_history = []

    for episode in range(NUM_EPISODES):
        #Start new episode and reset episode variables
        done = False
        state = env.reset()
        state = state_preproces(state)
        score = 0
        
        t_start = time.time()

        while not done:
            action = None
            
            state_, reward, done, info = env.step_discrete(action)
            state_ = state_preproces(state_)

            agent.learn(state,reward,state_,done)
            state = state_
            score += reward

        t_end = time.time
        score_history.append(score)
        completed = ((i/num_episodes)*100)

        print('ep. %4d' % i,' score %.2f' % score, 'compl: %2.1f' % completed, 'per estimated time: %1dh %2dm %2d'% (estimated_time(completed,NUM_EPISODES))'s' )




