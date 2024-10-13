# 局部优化算法测试

import gym
import numpy as np
import gym_flp
import gym_flp.util.FBSUtils as FBSUtils

instance = "Du62"
env = gym.make("fbs-v0", instance=instance, mode="human")
# permutation = np.array([3, 5, 7, 1, 4, 6, 2])
# bay = np.array([0, 0, 1, 0, 0, 0, 1])
# 初始化环境
# env.reset(layout=(permutation, bay))
env.reset()
env.render()

# 执行局部优化算法
FBSUtils.arrangementOptimization(env.permutation, env.bay, instance)

# env.reset(layout=(permutation, bay))
# env.render()

# 关闭环境
env.close()
