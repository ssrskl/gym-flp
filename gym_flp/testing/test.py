import gym
import gym_flp
import matplotlib.pyplot as plt
import imageio
import numpy as np
from stable_baselines3 import DQN

model = DQN.load("dqn-fbs-50000-AEG20")

env = gym.make("fbs-v0", instance="AEG20", mode="human")
permutation = np.array(
    [12, 20, 18, 16, 9, 7, 8, 15, 19, 6, 5, 13, 2, 10, 14, 4, 17, 3, 1, 11]
)
bay = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1])

# permutation = np.array(
#     [7, 8, 9, 2, 4, 5, 6, 13, 10, 14, 15, 1, 3, 19, 18, 12, 20, 11, 17, 16]
# )
# bay = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0])
# # [20  6 13 10  2 14  1  3 15  7 18  8 19  9  4  5 12 11 16 17]
# permutation = np.array(
#     [20, 6, 13, 10, 2, 14, 1, 3, 15, 7, 18, 8, 19, 9, 4, 5, 12, 11, 16, 17]
# )
# # [0. 0. 0. 1. 1. 0. 0. 1. 0. 0. 1. 0. 0. 0. 0. 1. 0. 1. 1. 0.]
# bay = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0])

# 封装layout
layout = (permutation, bay)
obs = env.reset(layout=layout)
env.render()
action = model.predict(obs)
print(f"Action为: {action}")
# print(obs)
# print(env.state)
# print(env.getFitness())
env.step(4)
# env.adjacent_exchange()
env.render()
# env.step(0)
# print(env.permutation)
# print(env.bay)
# env.render()
env.close()
