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
# model = DQN("MlpPolicy", env, verbose=1,tensorboard_log="logs")
model = DQN("MlpPolicy", env, verbose=1)



for episode in range(100):
    obs = env.reset()
    done = False
    total_timesteps = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        action = action.item()
        obs, reward, done, info = env.step(action)
        model.learn(total_timesteps=1, reset_num_timesteps=False)  # 每步更新模型
        total_timesteps += 1
        if total_timesteps >= 10000:
            break
    print("Episode: {}, Total Timesteps: {}".format(episode, total_timesteps))

model.save("dqn-fbs-40000-MB12")
