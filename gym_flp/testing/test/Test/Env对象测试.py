# Env对象测试

import copy
import gym
import gym_flp

instance = "O7-maoyan"
env_1 = gym.make("fbs-v0", instance=instance, mode="human")
env_2 = gym.make("fbs-v0", instance=instance, mode="human")

env_1.reset()
env_2.reset()

print(f"env_1的排列: {env_1.permutation}")
print(f"env_1的区代: {env_1.bay}")
print(f"env_2的排列: {env_2.permutation}")
print(f"env_2的区代: {env_2.bay}")

env_1.step(1)

print(f"env_1的排列: {env_1.permutation}")
print(f"env_1的区代: {env_1.bay}")
print(f"env_2的排列: {env_2.permutation}")
print(f"env_2的区代: {env_2.bay}")
