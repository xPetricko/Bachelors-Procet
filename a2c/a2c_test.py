import gym

from a2c import A2CAgent
from env import Env



env = Env(seed=0, action_repeat=1,
              img_stack=4)


MAX_EPISODE = 2000
MAX_STEPS = 100

lr = 1e-4
gamma = 0.99

agent = A2CAgent(env,gamma, lr)

def run():
    for episode in range(MAX_EPISODE):
        state = env.reset()
        transition = [] # [[s, a, r, s', done], [], ...]
        episode_reward = 0
        for steps in range(MAX_STEPS):
            action = agent.get_action(state)
            next_state, reward, done,die = env.step(action)
            transition.append([state, action, reward, next_state, done])
            episode_reward += reward
            if done:
                break
                
            state = next_state


        if episode % 10 == 0:
            print("Episode " + str(episode) + ": " + str(episode_reward))
        agent.update(transition)

run()