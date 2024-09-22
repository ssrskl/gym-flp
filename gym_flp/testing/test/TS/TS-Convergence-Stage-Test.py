# 禁忌搜索收敛模型二阶段优化算法

import copy
from datetime import datetime
import os
import random
from gym_flp.envs.FBS import FbsEnv
from gym_flp.util import FBSUtils
from stable_baselines3 import DQN

from gym_flp.util import TSExperimentDataGenerator


class TabuSearch:
    def __init__(
        self,
        instance,
        model,
        env,
        num_iterations,
        tabu_list_size,
        initial_solution,
        step_size=1,
    ):
        self.instance = instance
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
        fromEnv = copy.deepcopy(self.env)
        for _ in range(neighborhood_size):
            action, _ = self.model.predict(self.env.state)
            action_number = action.item()
            new_obs, reward, done, info = fromEnv.step(action_number)
            self.model.replay_buffer.add(
                self.env.state, new_obs, action, reward, done, [info]
            )
            permutation = fromEnv.permutation
            bay = fromEnv.bay
            neighbors.append((permutation, bay))
            fromEnv = copy.deepcopy(self.env)
        return neighbors

    def get_neighbors_2(self, current_solution, step_size=1, neighborhood_size=5):
        neighbors = []
        for _ in range(neighborhood_size):
            fromEnv = copy.deepcopy(self.env)
            action, _ = self.model.predict(self.env.state)
            action_number = action.item()
            new_obs, reward, done, info = self.env.step(action_number)
            self.model.replay_buffer.add(
                self.env.state, new_obs, action, reward, done, [info]
            )
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

            if iteration % 1000 == 0:
                print(
                    f"Iteration {iteration + 1}: Best Value = {best_value:.6f}, Current Solution = {current_solution}"
                )

        return best_solution, best_value


# 循环测试
for i in range(30):
    print(f"第{i}次循环")
    # 初始化FBS环境和模型
    total_steps = 10000
    instance = "AB20-ar3"
    env = FbsEnv(mode="human", instance=instance)
    model = DQN.load(f"./models/ts/dqn-fbs-TS-{instance}-{total_steps}")
    # 初始化参数
    initial_solution = FBSUtils.binary_solution_generator(
        env.area, env.n, env.fac_limit_aspect, env.L
    )  # 初始解
    num_iterations = 10000  # 迭代次数
    tabu_list_size = 100  # 禁忌表大小
    step_size = 1  # 邻域步长

    ts = TabuSearch(
        instance,
        model,
        env,
        num_iterations,
        tabu_list_size,
        initial_solution,
        step_size,
    )
    best_solution, best_value = ts.tabu_search()

    print(f"\nBest Solution: {best_solution}")
    print(f"Minimum Value: {best_value:.6f}")

    # 释放资源
    model.replay_buffer.clear()
    env.close()

    # env.reset(layout=best_solution)
    # env.render()
    # env.close()

    # 保存实验数据
    base_path = r"E:\projects\pythonprojects\gym-flp\algorithm\src\gym-flp"  # 使用原始字符串避免转义问题
    file_path = os.path.join(
        base_path,
        "ExperimentResult",
        "TS",
        "ts-convergence-stage-{}.xlsx".format(instance),
    )
    TSExperimentDataGenerator(
        experiment_name="TS-Convergence-Stage-Test",
        experiment_id=i + 1,
        start_time=datetime.now(),
        end_time=datetime.now(),
        duration=0,
        best_permutation=best_solution[0],
        best_bay=best_solution[1],
        best_result=best_value,
        tabu_list_size=tabu_list_size,
        num_iterations=num_iterations,
        step_size=step_size,
    ).saveExcel(file_path)
