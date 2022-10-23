import gym
import gym_cityflow
import numpy as np
import matplotlib.pyplot as plt

EPISODES = 100

if __name__ == "__main__":
    env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')

    rewards = []
    for _ in range(EPISODES):
        obs = env.reset()
        done = False
        total_reward = 0
        turns = 0
        while not done:
            action = np.random.randint(0, 9)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            turns += 1
        rewards.append(total_reward)

    plt.plot(rewards)
    plt.savefig("plot.png")
