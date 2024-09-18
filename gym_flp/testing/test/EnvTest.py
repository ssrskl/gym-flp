# 测试环境是否可以保存

import copy
import gym
import gym_flp

instance = "O7-maoyan"
env_1 = gym.make("fbs-v0", instance=instance, mode="human")
env_1.reset()

print(f"env_1转换前的排列: {env_1.permutation}")
print(f"env_1转换前的区代: {env_1.bay}")

env_2 = copy.deepcopy(env_1)  # 使用深拷贝，后续的转换不会影响env_2

print(f"env_2的排列: {env_2.permutation}")
print(f"env_2的区代: {env_2.bay}")

# 转换env_1
env_1.step(env_1.action_space.sample())
print(f"env_1转换后的排列: {env_1.permutation}")
print(f"env_1转换后的区代: {env_1.bay}")

print(f"env_2的排列: {env_2.permutation}")
print(f"env_2的区代: {env_2.bay}")
env_1.render()
env_1.close()
env_2.close()
