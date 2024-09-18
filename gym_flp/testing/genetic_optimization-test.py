from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import random
import pandas as pd
import openpyxl
import gym
import gym_flp
import matplotlib.pyplot as plt
import imageio
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.ppo.ppo import PPO
import matplotlib.pyplot as plt

# 设置matplotlib显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class GeneticAlgorithm:

    def __init__(self, population_size, generations, instance, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.instance = instance
        self.mutation_rate = mutation_rate  # 变异概率

        """种群初始化
        """

    def create_individual(self, instance):
        env = gym.make("fbs-v0", instance=instance, mode="human")
        env.reset()
        return env

    def initialize_population(self):
        # 并行化生成种群 2024-08-13
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.create_individual, self.instance)
                for _ in range(self.population_size)
            ]
            population = [future.result() for future in futures]
        return population

    def evaluate_individual(self, individual):
        fitness = individual.getFitness()
        return fitness

    def _select(self, population, fitness_values):  # 锦标赛选择法
        selected = []
        for _ in range(len(population)):
            # 随机选择三个个体进行锦标赛
            tournament = np.random.choice(len(population), 3, replace=False)
            # 找到锦标赛中适应度最高的个体
            winner = tournament[np.argmin(fitness_values[tournament])]
            # 将获胜者添加到选择列表中
            selected.append(population[winner])
        print("Selected individuals:", len(selected))
        return selected

    def _crossover(self, population, crossover_rate):
        # 顺序交叉
        # 随机选择两个父代个体
        parent1, parent2 = random.sample(population, 2)
        # 得到两个父代个体的信息
        parent1_permutation = parent1.permutation
        parent1_bay = parent1.bay
        parent2_permutation = parent2.permutation
        parent2_bay = parent2.bay
        print(
            "父代个体：",
            parent1_permutation,
            parent1_bay,
            parent2_permutation,
            parent2_bay,
        )
        # 随机选择交叉点
        crossover_point1 = random.randint(0, len(parent1_permutation) - 2)
        crossover_point2 = random.randint(
            crossover_point1 + 1, len(parent1_permutation) - 1
        )
        print("交叉点：", crossover_point1, crossover_point2)

        # 初始化子代个体的序列和 bay 数组
        child1_permutation = [-1] * len(parent1_permutation)
        child1_bay = [-1] * len(parent1_bay)
        child2_permutation = [-1] * len(parent2_permutation)
        child2_bay = [-1] * len(parent2_bay)

        # 复制交叉点之间的片段
        child1_permutation[crossover_point1:crossover_point2] = parent1_permutation[
            crossover_point1:crossover_point2
        ]
        child1_bay[crossover_point1:crossover_point2] = parent1_bay[
            crossover_point1:crossover_point2
        ]
        child2_permutation[crossover_point1:crossover_point2] = parent2_permutation[
            crossover_point1:crossover_point2
        ]
        child2_bay[crossover_point1:crossover_point2] = parent2_bay[
            crossover_point1:crossover_point2
        ]

        # 为子代补全剩余部分的序列和 bay 数组
        def fill_remaining_genes(child_perm, child_bay, parent_perm, parent_bay):
            current_pos = crossover_point2
            for i, gene in enumerate(parent_perm):
                if gene not in child_perm:
                    if current_pos >= len(child_perm):
                        current_pos = 0
                    while child_perm[current_pos] != -1:
                        current_pos += 1
                        if current_pos >= len(child_perm):
                            current_pos = 0
                    child_perm[current_pos] = gene
                    child_bay[current_pos] = parent_bay[i]

        fill_remaining_genes(
            child1_permutation, child1_bay, parent2_permutation, parent2_bay
        )
        fill_remaining_genes(
            child2_permutation, child2_bay, parent1_permutation, parent1_bay
        )

        # 处理子代的 bay 数组，假设它是简单复制的
        child1_bay = parent1_bay[:]  # 假设我们不对 bay 做任何变化
        child2_bay = parent2_bay[:]
        # 返回新生成的子代个体
        child1_env = gym.make("fbs-v0", instance=self.instance, mode="human")
        child2_env = gym.make("fbs-v0", instance=self.instance, mode="human")
        child1_env.reset(layout=(child1_permutation, child1_bay))
        child2_env.reset(layout=(child2_permutation, child2_bay))
        # 将子代添加到种群中
        population.append(child1_env)
        population.append(child2_env)
        # 将父代从种群中移除
        if parent1 == parent2:
            population.remove(parent1)
        else:
            population.remove(parent1)
            population.remove(parent2)
        return population

    def _mutate(self, population, mutation_rate):
        for individual in population:
            if random.random() < mutation_rate:
                individual.step(0)
        return population

    def _run(self, experiments_number=1):
        # 实验参数记录
        start_time = datetime.now()  # 实验开始时间
        # 加载环境
        env = gym.make("fbs-v0", instance=self.instance, mode="human")
        model_name = "dqn-fbs-40000-" + self.instance
        # 加载强化训练模型
        model = DQN.load(model_name)
        best_permutation = np.array([])
        best_bay = np.array([])
        best_fitness = float("inf")
        best_individual = env.reset()
        best_fitness_list = []
        # 初始化种群
        population = ga.initialize_population()
        print(f"种群个体数量: {len(population)}")
        for generation in range(ga.generations):
            # 并行计算种群中每个个体的适应度值
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.evaluate_individual, ind) for ind in population
                ]
                fitness_values = [f.result() for f in as_completed(futures)]
                fitness_values = np.array(fitness_values)
            print("best Fitness:", best_fitness)
            print(f"Generation {generation}: Best Fitness = {min(fitness_values)}")
            # population是一个环境集合
            # 选择
            population = self._select(population, fitness_values)
            # 交叉
            offspring = self._crossover(population, crossover_rate=0.8)
            # 变异
            offspring = self._mutate(offspring, mutation_rate=0.1)
            # 更新种群
            population = offspring
            # # 同时遍历种群和适应度值找到最佳个体
            # for i, individual in enumerate(population):
            #     if individual.getFitness() < best_fitness:
            #         best_fitness = individual.getFitness()
            #         best_individual = individual
            #         best_permutation = individual.permutation
            #         best_bay = individual.bay
            #         break
            # # 对最佳个体进行强化学习收敛
            # obs = best_individual.state
            # counter_number = 0
            # print("收敛前：best permutation:", best_permutation)
            # print("收敛前：best bay:", best_bay)
            # print("收敛前：best fitness:", best_fitness)
            # while True:
            #     counter_number += 1
            #     action, _state = model.predict(obs)
            #     action = action.item()
            #     obs, rewards, dones, info = best_individual.step(action)
            #     if dones or counter_number >= 10000:
            #         break
            # print("收敛后：best permutation:", best_individual.permutation)
            # print("收敛后：best bay:", best_individual.bay)
            # print("收敛后：best fitness:", best_individual.getFitness())

        # 打印种群信息
        print(f"种群个体数量：{len(population)}")
        for individual in population:
            print("适应度值:", individual.getFitness())
        # layout = (best_permutation, best_bay)
        # best_individual.reset(layout=layout)
        # best_individual.render()
        # best_individual.close()
        # print("最优排列：", best_permutation)
        # print("最优区带：", best_bay)
        # print("最优MHC：", best_fitness)
        # 记录实验参数
        if False:
            # 记录实验参数
            end_time = datetime.now()  # 实验结束时间
            duration = (end_time - start_time).total_seconds()  # 实验持续时间
            print("实验持续时间：", duration)
            # 保存实验数据
            data = {
                "实验次数": experiments_number,
                "开始时间": start_time,
                "结束时间": end_time,
                "持续时间": duration,
                "种群大小": self.population_size,
                "迭代次数": self.generations,
                "变异率": self.mutation_rate,
                "最优排列": [best_permutation],
                "最优区带": [best_bay],
                "最优MHC": best_fitness,
            }
            df = pd.DataFrame(data)
            print(df)
            file_path = "./实验结果/遗传算法实验数据.xlsx"
            try:
                with pd.ExcelWriter(
                    file_path, engine="openpyxl", mode="a", if_sheet_exists="overlay"
                ) as writer:
                    # 获取现有数据的行数
                    book = writer.book
                    sheet = book.active
                    startrow = sheet.max_row
                    # 将新数据追加到最后一行
                    df.to_excel(writer, index=False, header=False, startrow=startrow)
            except FileNotFoundError:
                # 如果文件不存在，则创建一个新的文件
                with pd.ExcelWriter(file_path, engine="openpyxl", mode="w") as writer:
                    df.to_excel(writer, index=False)
            print(f"实验数据已保存到 {file_path}")


# 循环实验30次
for i in range(1):
    ga = GeneticAlgorithm(
        population_size=5, generations=100, instance="O9-maoyan", mutation_rate=0.1
    )
    ga._run(experiments_number=i)
