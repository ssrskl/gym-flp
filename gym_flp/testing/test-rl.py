import gym
import gym_flp
import matplotlib.pyplot as plt
import imageio
import numpy as np
from stable_baselines3 import DQN

# model = DQN.load("dqn-fbs-episode-1000-AEG20")
model = DQN.load("models/dqn-fbs-Convergence-400-AB20-ar3")
best_fitness = float("inf")
best_permutation = None
best_bay = None
astringency_number = 0

env = gym.make("fbs-v0", instance="AB20-ar3", mode="human")
obs = env.reset()
for i in range(20000):
    action, _state = model.predict(obs)
    action = action.item()
    obs, rewards, dones, info = env.step(action)
    if env.getFitness() == env.getMHC():
        astringency_number += 1
    if env.getFitness() < best_fitness:
        best_fitness = env.getFitness()
        best_permutation = env.permutation
        best_bay = env.bay
        print(f"第{i}次迭代，当前最优fitness为: {best_fitness}")
    if dones:
        obs = env.reset()
print(f"收敛次数: {astringency_number}")
layout = layout = (best_permutation, best_bay)
env.reset(layout=layout)
env.render()
env.close()
