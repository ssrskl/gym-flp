# k分初始生成器的测试
import gym
import gym_flp
import numpy as np

instance = "O7-maoyan"
# 加载模型
# 创建环境
env = gym.make("fbs-v0", instance=instance, mode="human")
permutation = np.array([7, 6, 1, 5, 2, 3, 4])
bay = np.array([0, 0, 0, 0, 1, 0, 1])
permutation = np.array([3, 5, 7, 1, 4, 6, 2])
bay = np.array([0, 0, 1, 0, 0, 0, 1])
obs = env.reset(layout=(permutation, bay))

env.render()
env.close()
