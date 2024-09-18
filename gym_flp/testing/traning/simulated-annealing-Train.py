# 模拟退火算法
import math
import random
import gym
import gym_flp
import copy
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from gym_flp.util import FBSUtils


class SimulatedAnnealingFBS:
    def __init__(
        self, initial_temperature, cooling_rate, stopping_temperature, env, model
    ):
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.stopping_temperature = stopping_temperature
        self.env = env
        self.best_permutation = env.permutation
        self.best_bay = env.bay
        self.best_fitness = env.Fitness
        self.total_steps = 0
        self.change_count = 0
        self.model = model
        model.learn(total_timesteps=1, reset_num_timesteps=False)

    def anneal(self):
        # 保存环境状态
        fromEnv = copy.deepcopy(self.env)
        action, _states = self.model.predict(fromEnv.state)
        action_number = action.item()
        # print(f"当前执行的变换为：{action}")
        new_obs, reward, done, info = self.env.step(action_number)
        self.model.replay_buffer.add(
            fromEnv.state, new_obs, action, reward, done, [info]
        )
        # 计算新布局的适应度
        new_fitness = self.env.Fitness
        old_fitness = fromEnv.Fitness
        # 决定是否接受新解
        r = random.randint(0, 1)
        scaled_fitness_diff = abs(old_fitness - new_fitness)
        fitness_std = np.std([old_fitness, new_fitness])
        accept_probability = math.exp(-scaled_fitness_diff / (self.temperature))
        # print(f"差值：{scaled_fitness_diff}, 接受概率：{accept_probability}")
        if new_fitness < fromEnv.Fitness or r < accept_probability:
            # 更新环境
            self.change_count += 1
            # 使用局部搜索
            new_permutation = FBSUtils.exchangeOptimization(
                self.env.permutation,
                self.env.bay,
                self.env.area,
                self.env.W,
                self.env.D,
                self.env.F,
                self.env.fac_limit_aspect,
            )
            layout = (new_permutation, self.env.bay)
            self.env.reset(layout=layout)
            new_permutation = FBSUtils.exchangeOptimization(
                self.env.permutation,
                self.env.bay,
                self.env.area,
                self.env.W,
                self.env.D,
                self.env.F,
                self.env.fac_limit_aspect,
            )
            layout = (new_permutation, self.env.bay)
            self.env.reset(layout=layout)
        else:
            self.env = fromEnv  # 恢复原来的排列

        # 记录最佳解
        if self.env.Fitness < self.best_fitness:
            self.best_permutation = self.env.permutation
            self.best_bay = self.env.bay
            self.best_fitness = self.env.Fitness
        # 训练模型
        if self.total_steps % 1000 == 0 and self.total_steps > 0:
            self.train_model()

    def train_model(self):
        # 每次训练模型时取样经验并执行训练
        if self.model.replay_buffer.size() > self.model.batch_size:
            print(f"开始训练模型，步数：{self.total_steps}")
            # 从 replay buffer 中抽取批次进行训练
            self.model.train(batch_size=self.model.batch_size, gradient_steps=1)
            self.model.save(f"./models/dqn-fbs-SA-{self.total_steps}-Du62")

    def run(self):
        while self.temperature > self.stopping_temperature:
            self.total_steps += 1
            self.anneal()
            self.temperature *= self.cooling_rate
            # print(f"当前温度: {self.temperature:.2f}, 最佳适应度值: {self.best_fitness:.2f}")
        # 设置最佳排列并返回结果
        return (
            self.best_permutation,
            self.best_bay,
            self.best_fitness,
            self.change_count,
            self.total_steps,
        )


# 生成初始温度
def generate_initial_temperature(env):
    # 随机生成100个排列，并计算每个排列的适应度与MHC的差值
    fitness_list = []
    for _ in range(100):
        env.reset()
        fitness = env.Fitness
        mhc = env.MHC
        fitness_list.append(fitness - mhc)
    # 计算初始温度
    fitness_std = np.std(fitness_list)
    initial_temperature = -fitness_std / math.log(0.8)
    if initial_temperature == 0:
        initial_temperature = 100
    return initial_temperature


def main() -> None:
    instance = "Du62"
    trainingFrequency = 100
    env = gym.make("fbs-v0", instance=instance, mode="human")
    env.reset()

    model = DQN("MlpPolicy", env, verbose=1)

    # 设置模拟退火参数
    initial_temperature = generate_initial_temperature(env)
    print(f"初始温度：{initial_temperature}")
    cooling_rate = 0.999
    stopping_temperature = 0.1

    sa = SimulatedAnnealingFBS(
        initial_temperature, cooling_rate, stopping_temperature, env, model
    )
    (
        best_permutation,
        best_bay,
        best_fitness,
        change_count,
        total_steps,
    ) = sa.run()

    print(f"初始温度：{initial_temperature}")
    print(f"Best permutation: {best_permutation}")
    print(f"Best fitness: {best_fitness}")
    print(f"总计变换次数：{change_count}")
    print(f"总计迭代次数：{total_steps}")
    env.reset(layout=(best_permutation, best_bay))
    env.render()
    env.close()


if __name__ == "__main__":
    main()
