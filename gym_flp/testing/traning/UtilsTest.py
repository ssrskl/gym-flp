from gym_flp.util import FBSUtils
import numpy as np
import gym

instance = "O7-maoyan"
env = gym.make("fbs-v0", instance=instance, mode="human")
env.reset()

# 验证交换优化算法
# best_perm = FBSUtils.exchangeOptimization(
#     env.permutation, env.bay, env.area, env.W, env.D, env.F, env.fac_limit_aspect
# )
# layout = (best_perm, env.bay)
# env.reset(layout=layout)
# env.render()


# 验证全排列优化算法

# best_perm = FBSUtils.fullPermutationOptimization(
#     env.permutation, env.bay, env.area, env.W, env.D, env.F, env.fac_limit_aspect
# )

# layout = (best_perm, env.bay)
# env.reset(layout=layout)
# env.render()


# 验证k分初始生成器

permutation = np.array([5, 6, 7, 1, 2, 3, 4])
bay = np.array([0, 0, 0, 0, 1, 0, 1])
env.reset(layout=(permutation, bay))
env.render()
j = 0
for i in range(len(bay)):
    if bay[i] == 1:
        np.random.shuffle(permutation[j:i])
        j = i + 1
env.reset(layout=(permutation, bay))
env.render()
