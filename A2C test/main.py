import gym
from a2c import NewAgent
import matplotlib.pyplot as plt

import time

def estimated_time(completed,num_episodes)



if __name__ == '__main__':

    agent = NewAgent(alpha=0.0001, input_dims=[8], gamma=0.99, n_actions=4, layer1_size=2048, layer2_size=512)
    env = gym.make('LunarLander-v2')
    score_history = []
    num_episodes = 1000

    for i in range(num_episodes):
        done = False
        observation = env.reset()
        score = 0
        start = time.time()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.learn(observation,reward,observation_,done)
            observation = observation_
            score += reward
            
        end = time.time()
        score_history.append(score)
      
        completed = ((i/num_episodes)*100)
        estimated = ((100 - completed )/100)*num_episodes*(end-start)
        hour = estimated // 3600
        min = (estimated % 3600) // 60
        sek = (estimated % 3600) % 60
        
        print('ep. %4d' % i,' score %.2f' % score, 'compl: %2.1f' % completed, 'per estimated time: %1d'% hour,'h %2d' %min, 'm %2d'% sek, 's' )

    plt.plot(score_history)
    plt.show()


