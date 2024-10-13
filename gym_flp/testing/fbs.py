import gym
import gym_flp
import matplotlib.pyplot as plt
import imageio
import numpy as np
from stable_baselines3 import DQN

env = gym.make("fbs-v0", instance="O7-maoyan", mode="human")
obs = env.reset()

model = DQN.load("dqn-fbs-50000-human-O7-maoyan")
list_rewards = []
best_MHC = float("inf")
best_obs = None
for i in range(1000):
    action,_state = model.predict(obs)
    # action 是一个数组，但是在FBS中使用数字来表示动作
    action = action.item()
    print(action)
    obs = env.state
    obs, rewards, dones, info = env.step(action)
    list_rewards.append(-rewards)
    if (-rewards) < best_MHC:
        best_MHC = -rewards
        best_obs = obs
        env.render()
    if dones:
        env.reset()

plt.plot(list_rewards)
plt.show()
env.render()
print("最好的MHC：", best_MHC)
print("最好的状态：", best_obs)
env.close()
