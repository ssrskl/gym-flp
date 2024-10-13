import numpy as np
from gym_flp.envs.FBS import FbsEnv
import random


# 初始化鲸鱼优化算法参数
def initialize_woa_params(env, population_size):
    population = []
    for _ in range(population_size):
        # 使用环境的sampler方法生成初始解
        permutation, bay = env.sampler()
        population.append((permutation, bay))
    return population


# 计算适应度
def calculate_fitness(env, solution):
    env.reset(layout=solution)
    return env.Fitness


# 更新鲸鱼位置
def update_position(env, population, best_solution, a, a2):
    new_population = []
    b = 1  # 螺旋形状的常数
    for i in range(len(population)):
        r1 = random.random()
        r2 = random.random()
        A = 2 * a * r1 - a
        C = 2 * r2
        p = random.random()
        l = random.uniform(-1, 1)  # 在[-1, 1]之间的随机数
        if p < 0.5:
            if abs(A) < 1:
                D = abs(C * np.array(best_solution[0]) - np.array(population[i][0]))
                new_permutation = np.array(best_solution[0]) - A * D
            else:
                random_whale = population[random.randint(0, len(population) - 1)]
                D = abs(C * np.array(random_whale[0]) - np.array(population[i][0]))
                new_permutation = np.array(random_whale[0]) - A * D
        else:
            D = abs(np.array(best_solution[0]) - np.array(population[i][0]))
            new_permutation = D * np.exp(b * l) * np.cos(2 * np.pi * l) + np.array(best_solution[0])
        
        # 确保新解在合法范围内
        new_permutation = np.clip(new_permutation, 0, env.n - 1).astype(int)
        new_bay = population[i][1]  # 保持bay不变
        new_population.append((new_permutation, new_bay))
        
        # 使用env.step()更新环境
        env.reset(layout=(new_permutation, new_bay))
        action = random.choice(list(env.actions.keys()))
        env.step(action)
        
    return new_population


# 主函数：鲸鱼优化算法
def woa(env, num_iterations, population_size):
    population = initialize_woa_params(env, population_size)
    best_solution = None
    best_fitness = float("inf")

    for iteration in range(num_iterations):
        for i in range(population_size):
            fitness = calculate_fitness(env, population[i])
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = population[i]

        a = 2 - iteration * (2 / num_iterations)
        a2 = -1 + iteration * (-1 / num_iterations)
        population = update_position(env, population, best_solution, a, a2)

        print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness:.6f}")

    return best_solution, best_fitness


# 初始化FBS环境
env = FbsEnv(mode="human", instance="O9-maoyan")
env.reset()
# 初始化参数
population_size = 30
num_iterations = 5000
dim = env.n  # 设施数量
lb = 0  # 下界
ub = 1  # 上界

# 执行鲸鱼优化算法
best_solution, best_fitness = woa(env, num_iterations, population_size)

print(f"\nBest Solution: {best_solution}")
print(f"Minimum Fitness: {best_fitness:.6f}")

env.reset(layout=best_solution)
env.render()
env.close()
