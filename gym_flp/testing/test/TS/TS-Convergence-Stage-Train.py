# 禁忌搜索收敛模型二阶段优化算法

import copy
import random
from gym_flp.envs.FBS import FbsEnv
from gym_flp.util import FBSUtils
from stable_baselines3 import DQN


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
        self.model.learn(total_timesteps=1, reset_num_timesteps=False)

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
            fromEnv.reset(layout=current_solution)
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
        self.model.learn(total_timesteps=1, reset_num_timesteps=False)
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

            # 定期训练模型
            if self.total_steps % 1000 == 0:
                self.train_model()
                print(
                    f"Iteration {iteration + 1}: Best Value = {best_value:.6f}, Current Solution = {current_solution}"
                )

        return best_solution, best_value

    def train_model(self):
        # 每次训练模型时取样经验并执行训练
        if self.model.replay_buffer.size() > self.model.batch_size:
            print(f"开始训练模型，步数：{self.total_steps}")
            # 从 replay buffer 中抽取批次进行训练
            self.model.train(batch_size=self.model.batch_size, gradient_steps=1)
            self.model.save(
                f"./models/ts/dqn-fbs-TS-{self.instance}-{self.total_steps}"
            )


# 初始化FBS环境和模型
instance = "AB20-ar3"
env = FbsEnv(mode="human", instance=instance)
model = DQN("MlpPolicy", env, verbose=1)

# 初始化参数
initial_solution = FBSUtils.binary_solution_generator(
    env.area, env.n, env.fac_limit_aspect, env.L
)  # 初始解
num_iterations = 10000  # 迭代次数
tabu_list_size = 100  # 禁忌表大小
step_size = 1  # 邻域步长

ts = TabuSearch(
    instance, model, env, num_iterations, tabu_list_size, initial_solution, step_size
)
best_solution, best_value = ts.tabu_search()

print(f"\nBest Solution: {best_solution}")
print(f"Minimum Value: {best_value:.6f}")

env.reset(layout=best_solution)
env.render()
env.close()
