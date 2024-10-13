# 用于测试FBS动作空间
import gym
import numpy as np
import gym_flp
from gym_flp.util import FBSUtils


permutation = np.array([3, 5, 7, 1, 2, 4, 6])
bay = np.array([0, 0, 1, 0, 0, 0, 1])

# 交换同一bay中的两个设施
# permutation, bay = FBSUtils.swap_facility_in_bay(permutation, bay)
# 将bay的值转换
# bay = FBSUtils.bay_convert(bay)
# 交换两个bay
# permutation, bay = FBSUtils.swap_bay(permutation, bay)
env = gym.make("fbs-v0", instance="O7-maoyan", mode="human")
env.reset(layout=(permutation, bay))

# 修复bay
permutation, bay = FBSUtils.bay_repair(
    permutation, bay, env.fac_b, env.fac_h, env.fac_limit_aspect
)
print(permutation, bay)
