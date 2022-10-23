import gym
import gym_cityflow
import numpy as np
import math
import matplotlib.pyplot as plt

def action_prob(Q, state, epsilon, n = 9):
  probs = np.ones(n, dtype = float)
  probs = (probs * epsilon) / n
  best_action = Q[state].argmax()
  probs[best_action] += 1.0 - epsilon
  return probs

EPISODES = 100
epsilon = 0.1
epsilon_decay = 0.9
minimum_epsilon = 0.1
gamma = 0.05
alpha = 0.1

rewards = []
Q = {i: np.zeros(9) for i in range(8)}
env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')

for _ in range(EPISODES):
  total_reward = 0
  turn = 0
  ep = epsilon
  done = False
  state = env.reset().argmax()
  while not done:
    probs = action_prob(Q, state, ep)
    action = np.random.choice(np.arange(len(probs)), p = probs)

    nextState, reward, done, _ = env.step(action)
    turn += 1
    if turn % 10 == 0:
      # ep = max(minimum_epsilon, ep - epsilon_decay)
      ep *= epsilon_decay
    nextState = nextState.argmax()
    total_reward += reward

    diff = reward + gamma * max(Q[nextState]) - Q[state][action]
    Q[state][action] += alpha * diff
    state = nextState

  rewards.append(total_reward)

plt.plot(rewards)
plt.savefig('final_answer.png')
