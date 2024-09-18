from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import random
from gym_flp.testing.utils.ExperimentDataSaver import ExperimentDataSaver
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
        # population = []
        # for _ in range(self.population_size):
        #     env = gym.make("fbs-v0", instance=self.instance, mode="human")
        #     env.reset()
        #     population.append(env)
        # print("初始化种群:", population)
        # return population

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
            # 随机选择三个进行锦标赛
            tournament = np.random.choice(len(population), 3, replace=False)
            # 找到锦标赛中适应度最高的个体
            winner = tournament[np.argmin(fitness_values[tournament])]
            # 将获胜者添加到选择列表中
            selected.append(population[winner])
        return selected

    def _crossover(self, parent1, parent2):
        # 顺序交叉
        offspring = []
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        offspring.extend([child1, child2])
        return offspring

    def _mutate(self, individual):
        # 随机选择一个基因进行变异
        mutation_point = random.randint(0, len(individual) - 1)
        # 将该基因随机赋值为0或1
        individual[mutation_point] = random.randint(0, 1)
        return individual

    def _run(self, experiments_number=1):
        # 实验参数记录
        start_time = datetime.now()  # 实验开始时间

        # 加载环境
        env = gym.make("fbs-v0", instance=self.instance, mode="human")
        model_name = "dqn-fbs-40000-" + self.instance
        # 加载强化训练模型
        model = DQN.load(model_name)
        # model = PPO.load("ppo-fbs-1000x1000-human-VC10-maoyan")
        best_permutation = np.array([])
        best_bay = np.array([])
        best_fitness = float("inf")
        best_individual = env.reset()
        best_fitness_list = []
        # 初始化种群
        population = ga.initialize_population()
        for generation in range(ga.generations):
            # 并行计算种群中每个个体的适应度值
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.evaluate_individual, ind) for ind in population
                ]
                fitness_values = [f.result() for f in as_completed(futures)]
                fitness_values = np.array(fitness_values)
            # fitness_values = np.array(
            #     [individual.getFitness() for individual in population]
            # )
            print("best Fitness:", best_fitness)
            print(f"Generation {generation}: Best Fitness = {min(fitness_values)}")

            for individual in population:
                obs = individual.state  # 获取当前状态
                action, _state = model.predict(obs)
                action = action.item()
                # 变异操作
                if np.random.random() < self.mutation_rate:
                    action = individual.action_space.sample()
                obs, rewards, dones, info = individual.step(
                    action
                    # individual.action_space.sample()
                )
                # 如果个体的适应度接近最佳适应度，则执行局部搜索优化
                if (
                    best_fitness < individual.getFitness() < 1.5 * best_fitness
                    and individual.getFitness() == individual.getMHC()
                ):
                    # 强化学习局部优化
                    # print("执行强化学习局部搜索")
                    # action, _state = model.predict(individual.state)
                    # action = action.item()
                    # individual.step(action)
                    print("执行全排列局部搜索")
                    individual.local_search()  # 使用局部全排列的方式进行局部优化
                    # individual.adjacent_exchange()  # 使用局部交换的方式进行局部优化
                if individual.getFitness() > 1.5 * best_fitness:
                    individual.adjacent_exchange()  # 使用局部交换的方式进行局部优化

                if individual.getFitness() < best_fitness:
                    best_fitness = individual.getFitness()
                    best_permutation = individual.permutation
                    best_bay = individual.bay
                    best_individual = individual
                    print("当前的最优排列：", best_permutation)
                    print("当前的最优区带：", best_bay)
                    print("当前的最优适应度值：", best_fitness)
                    # best_individual.render()
            # 记录每一代的最佳适应度
            best_fitness_list.append(best_fitness)
            # 选择优化种群过程
            population = self._select(population, fitness_values)
            # population = self._select_roulette(population, fitness_values)
        # plt绘制每一代的最优解
        # plt.plot(best_fitness_list)
        # plt.xlabel("代数")
        # plt.ylabel("最佳适应度")
        # plt.title("适应度变化曲线")
        # plt.show()
        layout = (best_permutation, best_bay)
        best_individual.reset(layout=layout)
        # best_individual.render()
        best_individual.close()
        print("最优排列：", best_permutation)
        print("最优区带：", best_bay)
        print("最优MHC：", best_fitness)
        # 记录实验参数
        end_time = datetime.now()  # 实验结束时间
        duration = (end_time - start_time).total_seconds()  # 实验持续时间
        print("实验持续时间：", duration)
        # 保存实验数据
        saver = ExperimentDataSaver()
        saver.save_data(
            experiments_number=1,
            start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            duration="1 minute",
            population_size=100,
            generations=50,
            mutation_rate=0.01,
            best_permutation=[1, 2, 3, 4, 5],
            best_bay=[1, 0, 0, 1, 1],
            best_fitness=123.45,
        )


# 循环实验30次
for i in range(30):
    ga = GeneticAlgorithm(
        population_size=50, generations=100, instance="O9-maoyan", mutation_rate=0.1
    )
    ga._run(experiments_number=i)
