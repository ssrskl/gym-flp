# 测试收敛模型的效果
import gym
import gym_flp
import matplotlib.pyplot as plt
import imageio
import numpy as np
from stable_baselines3 import DQN
from datetime import datetime

instance = "AB20-ar3"
# 加载模型
model = DQN.load("./models/dqn-fbs-Convergence-400-AB20-ar3")
# 创建环境
env = gym.make("fbs-v0", instance=instance, mode="human")
obs = env.reset()

# 测试模型
max_steps = 100000
current_steps = 0
start_time = datetime.now()
end_time = datetime.now()
duration = end_time - start_time
duration_list = []
max_episodes = 30  # 最大测试次数
convergence_number = 0  # 收敛次数
fitness_list = []
best_fitness = 10000

for episode in range(max_episodes):
    print(f"第{episode + 1}次测试开始")
    start_time = datetime.now()
    env.reset()  # 重置环境
    while current_steps < max_steps:
        current_steps += 1
        action, _states = model.predict(obs)
        action = action.item()
        obs, rewards, dones, info = env.step(action)
        if dones:
            convergence_number += 1
            break
    fitness_list.append(env.getFitness())
    if env.getFitness() < best_fitness:
        best_fitness = env.getFitness()
    end_time = datetime.now()
    duration = end_time - start_time
    duration_list.append(duration)

env.close()

print(
    f"测试结束，共测试{max_episodes}次，其中{convergence_number}次收敛，平均每次测试时间为{np.mean(duration_list)}，最佳适应度值为{best_fitness}，平均适应度为{np.mean(fitness_list)}"
)
