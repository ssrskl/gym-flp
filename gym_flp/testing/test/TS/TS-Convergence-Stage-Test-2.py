# 禁忌搜索优化算法

import copy
from datetime import datetime
import os
import random
from gym_flp.envs.FBS import FbsEnv
from gym_flp.util import FBSUtils
from gym_flp.util.TSExperimentDataGenerator import TSExperimentDataGenerator
from stable_baselines3 import DQN


# 目标函数：计算设施布局的适应度
def objective_function(env, layout):
    env.reset(layout=layout)
    return env.Fitness


# 获取邻域解2 （领域解来源于当前环境，当前环境在迭代过程中不断更新）
def get_neighbors(env, model, current_solution, step_size=1, neighborhood_size=5):
    neighbors = []
    for _ in range(neighborhood_size):
        # action = random.choice(list(env.actions.keys()))
        action, _ = model.predict(env.state)
        action_number = action.item()
        env.reset(layout=current_solution)
        env.step(action_number)
        permutation = env.permutation
        bay = env.bay
        neighbors.append((permutation, bay))
    return neighbors


# 禁忌搜索算法
def tabu_search(
    env, model, num_iterations, tabu_list_size, initial_solution, step_size=1
):
    current_solution = initial_solution
    best_solution = current_solution
    best_value = objective_function(env, current_solution)

    tabu_list = []

    for iteration in range(num_iterations):
        neighbors = get_neighbors(env, model, current_solution, step_size)
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
        if iteration % 1000 == 0:
            print(
                f"Iteration {iteration + 1}: Best Value = {best_value:.6f}, Current Solution = {current_solution}"
            )

    return best_solution, best_value


instance = "AB20-ar3"
total_steps = 100000
model = DQN.load(f"./models/ts/dqn-fbs-TS-{instance}-{total_steps}")
for i in range(30):
    print(f"第{i+1}次实验")
    # 初始化参数
    env = FbsEnv(mode="human", instance=instance)
    initial_solution = FBSUtils.binary_solution_generator(
        env.area, env.n, env.fac_limit_aspect, env.L
    )  # 初始解
    num_iterations = 100000  # 迭代次数
    tabu_list_size = 200  # 禁忌表大小
    step_size = 1  # 邻域步长

    # 执行禁忌搜索
    best_solution, best_value = tabu_search(
        env, model, num_iterations, tabu_list_size, initial_solution, step_size
    )

    print(f"\nBest Solution: {best_solution}")
    print(f"Minimum Value: {best_value:.6f}")

    # 保存实验数据
    base_path = r"E:\projects\pythonprojects\gym-flp\algorithm\src\gym-flp"  # 使用原始字符串避免转义问题
    file_path = os.path.join(
        base_path,
        "ExperimentResult",
        "TS",
        "ts-{}.xlsx".format(instance),
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
