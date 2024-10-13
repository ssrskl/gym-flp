# 禁忌搜索优化算法

import random
from gym_flp.envs.FBS import FbsEnv
from gym_flp.util import FBSUtils


# 目标函数：计算设施布局的适应度
def objective_function(env, layout):
    env.reset(layout=layout)
    return env.Fitness


# 获取邻域解
def get_neighbors(env, current_solution, step_size=1, neighborhood_size=5):
    neighbors = []
    for _ in range(neighborhood_size):
        action = random.choice(list(env.actions.keys()))
        env.reset(layout=current_solution)
        env.step(action)
        permutation = env.permutation
        bay = env.bay
        neighbors.append((permutation, bay))
    return neighbors


# 禁忌搜索算法
def tabu_search(env, num_iterations, tabu_list_size, initial_solution, step_size=1):
    current_solution = initial_solution
    best_solution = current_solution
    best_value = objective_function(env, current_solution)

    tabu_list = []

    for iteration in range(num_iterations):
        neighbors = get_neighbors(env, current_solution, step_size)
        candidate_solution = None
        candidate_value = float("inf")

        # 从邻域中选择最优解（不在禁忌表中）
        for neighbor in neighbors:
            neighbor_tuple = tuple(map(tuple, neighbor))  # 将邻域解转换为元组
            if neighbor_tuple not in tabu_list:
                neighbor_value = objective_function(env, neighbor)
                if neighbor_value < candidate_value:
                    candidate_solution = neighbor
                    candidate_value = neighbor_value

        # 更新当前解
        if candidate_solution is not None:
            current_solution = candidate_solution

            # 如果找到更优解，则更新最佳解
            if candidate_value < best_value:
                best_solution = current_solution
                best_value = candidate_value

            # 更新禁忌表
            tabu_list.append(tuple(map(tuple, current_solution)))  # 将当前解转换为元组
            if len(tabu_list) > tabu_list_size:
                tabu_list.pop(0)  # 保持禁忌表大小固定

        print(
            f"Iteration {iteration + 1}: Best Value = {best_value:.6f}, Current Solution = {current_solution}"
        )

    return best_solution, best_value


# 初始化FBS环境
env = FbsEnv(mode="human", instance="O9-maoyan")

# 初始化参数
initial_solution = env.sampler()  # 初始解
num_iterations = 5000  # 迭代次数
tabu_list_size = 50  # 禁忌表大小
step_size = 1  # 邻域步长

# 执行禁忌搜索
best_solution, best_value = tabu_search(
    env, num_iterations, tabu_list_size, initial_solution, step_size
)

print(f"\nBest Solution: {best_solution}")
print(f"Minimum Value: {best_value:.6f}")

env.reset(layout=best_solution)
env.render()
env.close()
