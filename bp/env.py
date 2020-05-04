import numpy as np
import gym

from ed_env import CarRacing
from my_func import *

class Env():
    def __init__(self,seed,action_repeat, img_stack):
        self.seed = seed
        self.action_repeat = action_repeat
        self.img_stack = img_stack
        
        self.env = CarRacing()#gym.make('CarRacing-v0')
        self.env.seed(self.seed)
        self.reward_threshold = 900#self.env.spec.reward_threshold


    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = rgb_to_gray(img_rgb)
        self.stack = [img_gray] * self.img_stack  # four frames for decision
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(self.action_repeat):
            img_rgb, reward, die, _ = self.env.step(action)
            done = die
            # don't penalize "done state"
#            if die:
#                reward += 100
            # green penalty
#            if np.mean(img_rgb[:, :, 1]) > 185.0:
#                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
#            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break

        img_gray = rgb_to_gray(img_rgb)# self.rgb2gray(img_rgb)

        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack
        return np.array(self.stack), total_reward, done, die

    def render(self, *arg):
        self.env.render(*arg)


    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

