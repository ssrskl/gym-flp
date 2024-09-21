# 禁忌搜索优化算法

import copy
import random
from gym_flp.envs.FBS import FbsEnv
from gym_flp.util import FBSUtils
from stable_baselines3 import DQN


class TabuSearch:
    def __init__(
        self, model, env, num_iterations, tabu_list_size, initial_solution, step_size=1
    ):
        self.model = model
        self.env = env
        self.num_iterations = num_iterations
        self.tabu_list_size = tabu_list_size
        self.initial_solution = initial_solution
        self.step_size = step_size
        self.total_steps = 0  # 添加总步数计数器

    # 目标函数：计算设施布局的适应度
    def objective_function(self, layout):
        self.env.reset(layout=layout)
        return self.env.Fitness

    # 获取邻域解
    def get_neighbors(self, current_solution, step_size=1, neighborhood_size=5):
        neighbors = []
        for _ in range(neighborhood_size):
            fromEnv = copy.deepcopy(self.env)
            action, _ = self.model.predict(self.env.state)
            action_number = action.item()
            self.env.reset(layout=current_solution)
            self.env.step(action_number)
            permutation = self.env.permutation
            bay = self.env.bay
            neighbors.append((permutation, bay))
        return neighbors

    # 禁忌搜索算法
    def tabu_search(self):
        current_solution = self.initial_solution
        best_solution = current_solution
        best_value = self.objective_function(current_solution)

        tabu_list = []

        for iteration in range(self.num_iterations):
            self.total_steps += 1  # 更新总步数
            neighbors = self.get_neighbors(current_solution, self.step_size)
            candidate_solution = None
            candidate_value = float("inf")

            # 从邻域中选择最优解（不在禁忌表中）
            for neighbor in neighbors:
                neighbor_tuple = tuple(map(tuple, neighbor))  # 将邻域解转换为元组
                if neighbor_tuple not in tabu_list:
                    neighbor_value = self.objective_function(neighbor)
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
                tabu_list.append(
                    tuple(map(tuple, current_solution))
                )  # 将当前解转换为元组
                if len(tabu_list) > self.tabu_list_size:
                    tabu_list.pop(0)  # 保持禁忌表大小固定
            print(
                f"Iteration {iteration + 1}: Best Value = {best_value:.6f}, Current Solution = {current_solution}"
            )

        return best_solution, best_value


# 初始化FBS环境和模型
train_num = 5000
instance = "O9-maoyan"
env = FbsEnv(mode="human", instance=instance)
model = DQN.load(f"./models/ts/dqn-fbs-TS-{instance}-{train_num}")

# 初始化参数
initial_solution = env.sampler()  # 初始解
num_iterations = 5000  # 迭代次数
tabu_list_size = 50  # 禁忌表大小
step_size = 1  # 邻域步长

ts = TabuSearch(model, env, num_iterations, tabu_list_size, initial_solution, step_size)
best_solution, best_value = ts.tabu_search()

print(f"\nBest Solution: {best_solution}")
print(f"Minimum Value: {best_value:.6f}")

env.reset(layout=best_solution)
env.render()
env.close()
