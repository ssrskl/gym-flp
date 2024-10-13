import random
import gym
import gym_flp
import numpy as np
from stable_baselines3 import DQN


def initialize_population(self):
    population = []
    for _ in range(population_size):
        env = gym.make("fbs-v0", instance=instance, mode="human")
        env.reset()
        population.append(env)
    return population


def evaluate_fitness(individual):
    return individual.getFitness()


def get_fitness(population):
    return np.array([evaluate_fitness(individual) for individual in population])


def individual_info(individual):
    print("Individual:", individual)
    print("适应度值:", evaluate_fitness(individual=individual))
    print(f"排列：{individual.permutation}")
    print(f"区带：{individual.bay}")


def population_info(population):
    for individual in population:
        individual_info(individual)
    print("Population:", len(population))


def selection(population, fitness_values):
    # 实现选择操作，种群大小始终为population_size
    selected = []
    for _ in range(population_size):
        tournament = np.random.choice(len(population), 3, replace=False)
        winner = tournament[np.argmin(fitness_values[tournament])]
        selected.append(population[winner])
    return selected


def crossover(parents, crossover_rate):
    if random.random() < crossover_rate:
        # 实现交叉操作
        parent1, parent2 = random.sample(parents, 2)
        parent1_permutation = parent1.permutation
        parent2_permutation = parent2.permutation
        parent1_bay = parent1.bay
        parent2_bay = parent2.bay
        child1_permutation, child1_bay, child2_permutation, child2_bay = (
            order_crossover(
                parent1_permutation, parent1_bay, parent2_permutation, parent2_bay
            )
        )
        # 生成两个新的个体
        child1 = gym.make("fbs-v0", instance=instance, mode="human")
        child1.reset(layout=(child1_permutation, child1_bay))
        child2 = gym.make("fbs-v0", instance=instance, mode="human")
        child2.reset(layout=(child2_permutation, child2_bay))
        # 添加到种群中
        parents.append(child1)
        parents.append(child2)
        return parents
    else:
        return parents


def order_crossover(parent1, bay1, parent2, bay2):
    size = len(parent1)

    # 随机选择两个交叉点
    start, end = sorted(random.sample(range(size), 2))
    print(f"start:{start}, end:{end}")

    # 初始化子本和子bay为None
    child1 = np.full(size, None)
    child2 = np.full(size, None)
    child_bay1 = np.full(size, None)
    child_bay2 = np.full(size, None)

    # 将父本和父bay在交叉点之间的部分复制到子本中
    child1[start : end + 1] = parent1[start : end + 1]
    child_bay1[start : end + 1] = bay1[start : end + 1]
    child2[start : end + 1] = parent2[start : end + 1]
    child_bay2[start : end + 1] = bay2[start : end + 1]

    # 填充子本的剩余部分
    fill_positions = [i for i in range(size) if i < start or i > end]
    fill_values1 = [item for item in parent2 if item not in child1]
    # 查找bay1中与fill_values1中元素对应的值
    fill_bay_values1 = [bay1[np.where(parent1 == item)[0][0]] for item in fill_values1]
    for i, pos in enumerate(fill_positions):
        child1[pos] = fill_values1[i]
        child_bay1[pos] = fill_bay_values1[i]

    fill_values2 = [item for item in parent1 if item not in child2]
    fill_bay_values2 = [bay2[np.where(parent2 == item)[0][0]] for item in fill_values2]
    for i, pos in enumerate(fill_positions):
        child2[pos] = fill_values2[i]
        child_bay2[pos] = fill_bay_values2[i]

    # return child1, bay1, child2, bay2
    return child1, child_bay1, child2, child_bay2


def mutation(offspring, mutation_rate):
    # 实现变异操作
    return offspring


def reinforce_learning_optimization_train(individuals):
    # 将种群中的个体进行优化
    optimized_individuals = []
    current_individual_number = 0
    for individual in individuals:
        current_individual_number += 1
        obs = individual.state
        for step in range(max_steps_per_episode):
            action, _states = model.predict(obs, deterministic=False)
            action_number = action.item()
            action = action
            new_obs, reward, done, info = individual.step(action_number)
            # 存储transition
            model.replay_buffer.add(obs, new_obs, action, reward, done, [info])
            obs = new_obs
            if done:
                break
        # 每个episode结束后进行训练
        if model.replay_buffer.size() > model.batch_size:
            model.train(gradient_steps=step, batch_size=model.batch_size)
        print(f"当前个体{current_individual_number}/{len(individuals)}")
        optimized_individuals.append(individual)
    optimized_individuals = np.array(optimized_individuals)
    return optimized_individuals


def genetic_algorithm(population_size, num_generations, crossover_rate, mutation_rate):
    # 初始化种群
    population = initialize_population(population_size)
    for generation in range(num_generations):
        # 评估适应度
        fitness_scores = get_fitness(population)
        # 选择
        selected = selection(population, fitness_scores)
        # 交叉
        offspring = crossover(selected, crossover_rate)
        # 变异
        offspring = mutation(offspring, mutation_rate)
        # 强化学习优化
        optimized_offspring = reinforce_learning_optimization_train(offspring)
        # 更新种群
        population = optimized_offspring
        print(f"Generation {generation+1}/{num_generations}")

        # 保存模型
        if generation % 10 == 0 and generation != 0:
            model.save(f"./models/dqn-fbs-episode-{generation}-{instance}")

    # 返回最优解
    fitness_scores = get_fitness(population)
    best_fitenss = np.min(fitness_scores)
    best_individual = population[np.argmin(fitness_scores)]
    print(f"最佳排列：{best_individual.permutation}")
    print(f"最佳区带：{best_individual.bay}")
    best_individual.render()
    print(f"最佳适应度值：{best_fitenss}")


# 主程序
instance = "O9-maoyan"
env = gym.make("fbs-v0", instance=instance, mode="human")
model = DQN("MlpPolicy", env, verbose=1)
# model = DQN.load("dqn-fbs-episode-1000-AEG20")
# 初始化模型
model.learn(total_timesteps=1, reset_num_timesteps=False)
max_steps_per_episode = 10000
# 遗传算法参数
population_size = 50
num_generations = 100
crossover_rate = 0.8
mutation_rate = 0.1

best_solution = genetic_algorithm(
    population_size, num_generations, crossover_rate, mutation_rate
)
