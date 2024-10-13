import gym
import gym_flp
import matplotlib.pyplot as plt
import imageio
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.ddpg.ddpg import DDPG
from stable_baselines3.ppo.ppo import PPO

# 创建环境和模型
env = gym.make("fbs-v0", instance="MB12", mode="human")
model = DQN("MlpPolicy", env, verbose=1)
# 训练模型
model.learn(total_timesteps=10000)
# 保存模型
model.save("dqn-fbs-immobilization-MB12-10000")
