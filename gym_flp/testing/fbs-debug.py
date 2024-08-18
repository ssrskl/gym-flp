import gym
import gym_flp
import matplotlib.pyplot as plt
import imageio
import numpy as np
from stable_baselines3 import DQN

env = gym.make("fbs-v0", instance="MB12", mode="human")
env.reset()
# 存储每一次的Rewards数组
list_rewards = []

for i in range(100):
    obs, rewards, dones, info = env.step(env.action_space.sample())
    print("第{}步".format(i))
    list_rewards.append(-rewards)
    if dones:
        env.reset()

# 画出Rewards曲线
plt.title("MHC")
plt.plot(list_rewards)
plt.show()
env.render()
env.close()
