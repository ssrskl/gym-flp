# 模拟退火算法
import math
import random
import gym
import gym_flp
import copy


class SimulatedAnnealingFBS:
    def __init__(self, initial_temperature, cooling_rate, stopping_temperature, env):
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.stopping_temperature = stopping_temperature
        self.env = env
        self.best_permutation = env.permutation
        self.best_bay = env.bay
        self.best_fitness = env.getFitness()
        self.current_fitness = self.best_fitness
        self.total_steps = 0
        self.change_count = 0

    def anneal(self):
        # 保存环境状态
        fromEnv = copy.deepcopy(self.env)
        # 环境动作
        # action = random.randint(0, 4)
        action = self.env.action_space.sample()
        print(f"当前执行的变换为：{action}")
        self.env.step(action)
        # 计算新布局的适应度
        new_fitness = self.env.getFitness()
        # 决定是否接受新解
        if self.accept_probability(self.current_fitness, new_fitness):
            # 更新环境
            if new_fitness < self.best_fitness:
                self.best_fitness = new_fitness
                self.best_permutation = self.env.permutation
                self.best_bay = self.env.bay
                self.change_count += 1
        else:
            self.env = fromEnv  # 恢复原来的排列
                      
    def accept_probability(self, old_cost, new_cost):
        if new_cost < old_cost:
            return 1.0
        return math.exp((old_cost - new_cost) / self.temperature)

    def run(self):
        while self.temperature > self.stopping_temperature:
            self.total_steps += 1
            self.anneal()
            self.temperature *= self.cooling_rate
            print(
                f"当前温度: {self.temperature:.2f}, 最佳适应度值: {self.best_fitness:.2f}"
            )
        # 设置最佳排列并返回结果
        return (
            self.best_permutation,
            self.best_bay,
            self.best_fitness,
            self.change_count,
            self.total_steps,
        )


def main():
    instance = "AB20-ar3"
    env = gym.make("fbs-v0", instance=instance, mode="human")
    env.reset()

    # 设置模拟退火参数
    initial_temperature = 100.0
    cooling_rate = 0.99
    stopping_temperature = 0.01

    sa = SimulatedAnnealingFBS(
        initial_temperature, cooling_rate, stopping_temperature, env
    )
    (
        best_permutation,
        best_bay,
        best_fitness,
        change_count,
        total_steps,
    ) = sa.run()

    print(f"Best permutation: {best_permutation}")
    print(f"Best fitness: {best_fitness}")
    print(f"总计变换次数：{change_count}")
    print(f"总计迭代次数：{total_steps}")
    env.reset(layout=(best_permutation, best_bay))
    env.render()
    env.close()


if __name__ == "__main__":
    main()
