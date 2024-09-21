# Shuffle 算子测试
import gym
import gym_flp
import numpy as np

instance = "O7-maoyan"
# 加载模型

# 创建环境
env = gym.make("fbs-v0", instance=instance, mode="human")
# O7-maoyan 设置
permutation = np.array([3, 5, 7, 1, 4, 6, 2])
bay = np.array([0, 0, 1, 0, 0, 0, 1])
layout = (permutation, bay)
obs = env.reset(layout=layout)
env.render()
obs, rewards, dones, info = env.step(4)
env.render()
env.close()
