# k分收敛模型生成器

import gym
import gym_flp
import matplotlib.pyplot as plt
import imageio
import numpy as np
from stable_baselines3 import DQN
from datetime import datetime

instance = "O9-maoyan"
max_steps = 1000
current_step = 0
# 加载模型
model = DQN.load("./models/dqn-fbs-Convergence-500-O9-maoyan")
# 创建环境
env = gym.make("fbs-v0", instance=instance, mode="human")
obs = env.reset()
# 运行模型，创建收敛模型
while current_step < max_steps:
    current_step += 1
    action, _states = model.predict(obs)
    action = action.item()
    obs, rewards, dones, info = env.step(action)
    if dones:
        print(f"执行了{current_step}步，生成了收敛模型")
        break
# 打印模型信息
print(
    f"模型排列:{env.permutation}，模型区带:{env.bay}，模型适应度值:{env.getFitness()}"
)
