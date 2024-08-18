import gym
import gym_flp
import matplotlib.pyplot as plt
import imageio
import numpy as np
from stable_baselines3 import DQN

env = gym.make("fbs-v0", instance="O7-maoyan", mode="human")
obs = env.reset()
permutation = env.permutation
bay = env.bay
a = env.a
W = env.W
print(f"排列：{env.permutation}")
print(f"区带:{env.bay}")
print(f"面积:{a}")
print(f"设施长宽比:{env.beta}")
print(f"设施宽度限制:{env.w}")
print(f"设施长度限制:{env.l}")
print(f"厂房W:{env.W}")# y
print(f"厂房L:{env.L}")# x
fac_x, fac_y, fac_b, fac_h = env.getCoordinates_mao(bay, permutation, a, W)


print(f"x坐标：{fac_x}")
print(f"y坐标：{fac_y}")
print(f"宽度：{fac_b}")
print(f"长度：{fac_h}")
env.render()
env.close()
