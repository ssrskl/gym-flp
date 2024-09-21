# 禁忌搜索优化算法


import random


# 目标函数 f(x) = x^2
def objective_function(x):
    return x**2


# 获取邻域解（可以简单地通过在当前解附近随机取值生成邻居）
def get_neighbors(current_solution, step_size=1, neighborhood_size=5):
    neighbors = []
    for _ in range(neighborhood_size):
        neighbor = current_solution + random.uniform(-step_size, step_size)
        neighbors.append(neighbor)
    return neighbors


# 禁忌搜索算法
def tabu_search(num_iterations, tabu_list_size, initial_solution, step_size=1):
    current_solution = initial_solution
    best_solution = current_solution
    best_value = objective_function(current_solution)

    tabu_list = []

    for iteration in range(num_iterations):
        neighbors = get_neighbors(current_solution, step_size)
        candidate_solution = None
        candidate_value = float("inf")

        # 从邻域中选择最优解（不在禁忌表中）
        for neighbor in neighbors:
            if neighbor not in tabu_list:
                neighbor_value = objective_function(neighbor)
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
            tabu_list.append(current_solution)
            if len(tabu_list) > tabu_list_size:
                tabu_list.pop(0)  # 保持禁忌表大小固定

        print(
            f"Iteration {iteration + 1}: Best Value = {best_value:.6f}, Current Solution = {current_solution:.6f}"
        )

    return best_solution, best_value


# 初始化参数
initial_solution = random.uniform(-10, 10)  # 初始解
num_iterations = 100  # 迭代次数
tabu_list_size = 10  # 禁忌表大小
step_size = 1  # 邻域步长

# 执行禁忌搜索
best_solution, best_value = tabu_search(
    num_iterations, tabu_list_size, initial_solution, step_size
)

print(f"\nBest Solution: x = {best_solution:.6f}")
print(f"Minimum Value: f(x) = {best_value:.6f}")
