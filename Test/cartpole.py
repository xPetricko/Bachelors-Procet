import numpy as np
import gym
from actor_critic_discrete import Agent
import matplotlib.pyplot as plt

# from utils import plotLearning
from gym import wrappers

if __name__ == "__main__":
    agent = Agent(
        alpha=0.00001,
        beta=0.0005,
        input_dims=[4],
        gamma=0.99,
        n_actions=2,
        l1_size=256,
        l2_size=256
    )
    env = gym.make("CartPole-v1")
    score_history = []
    n_episodes = 2500

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.learn(observation, reward, observation_, done)
            observation = observation_
        print("episode ", i, " score: %.3f" % score)
        if not (i % 10 ):
            score_history.append(score)
        if not (i % 500):
            plt.plot(score_history)
            plt.show()

