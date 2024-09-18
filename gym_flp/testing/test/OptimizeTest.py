# 优化模型测试

import gym
import gym_flp
import matplotlib.pyplot as plt
import imageio
import numpy as np
from stable_baselines3 import DQN
from datetime import datetime


instance = "MB12"
# 加载模型
model = DQN.load("./models/dqn-fbs-Optimize-500-MB12")
# 创建环境
env = gym.make("fbs-v0", instance=instance, mode="human")
obs = env.reset()

max_steps = 100000
current_steps = 0

while current_steps < max_steps:
    current_steps += 1
    action, _states = model.predict(obs)
    action = action.item()
    obs, rewards, dones, info = env.step(action)
    if dones:
        break

print(f"共执行了{current_steps}步, 适应度值为{env.getFitness()}")
