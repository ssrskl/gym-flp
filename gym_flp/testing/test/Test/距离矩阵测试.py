# 距离矩阵测试

import numpy as np
import gym
import gym_flp
from gym_flp.util import FBSUtils

instance = "O7-maoyan"
env = gym.make("fbs-v0", instance=instance, mode="human")
permutation = np.array([3, 5, 7, 1, 4, 6, 2])
bay = np.array([0, 0, 1, 0, 0, 0, 1])
env.reset(layout=(permutation, bay))
fac_x = env.fac_x
fac_y = env.fac_y
print(fac_x)
print(fac_y)
print("-" * 100)
print(FBSUtils.getEuclideanDistances(fac_x, fac_y))
print("-" * 100)
print(FBSUtils.getManhattanDistances(fac_x, fac_y))
env.render()
env.close()
