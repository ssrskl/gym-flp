# step的repair测试

import gym
import gym_flp
import numpy as np

instance = "AEG20"
# 加载模型

# 窄模型修复
# 创建环境
env = gym.make("fbs-v0", instance=instance, mode="human")
# AEG20 设置
permutation = np.array(
    [12, 20, 18, 16, 9, 7, 8, 15, 19, 6, 5, 13, 2, 10, 14, 4, 17, 3, 1, 11]
)
bay = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1])
layout = (permutation, bay)
obs = env.reset(layout=layout)
env.render()
obs, rewards, dones, info = env.step(4)
env.render()
env.close()

# # 宽模型修复
# # 创建环境
# env = gym.make("fbs-v0", instance=instance, mode="human")
# # AEG20 设置
# permutation = np.array(
#     [12, 20, 18, 16, 9, 7, 8, 15, 19, 6, 5, 13, 2, 10, 14, 4, 17, 3, 1, 11]
# )
# bay = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
# layout = (permutation, bay)
# obs = env.reset(layout=layout)
# env.render()
# obs, rewards, dones, info = env.step(4)
# env.render()
# env.close()
