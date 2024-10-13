# 测试CRO算法

import numpy as np


# 定义目标函数
def objective_function(x):
    return x**2


# 初始化参数
reef_size = (10, 10)  # 珊瑚礁的大小 (10x10 的矩阵)
initial_occupation_rate = 0.7  # 初始占用率
num_iterations = 100  # 迭代次数
Fb = 0.8  # 广播繁殖的比例
Fa = 0.2  # 无性繁殖的比例
Fd = 0.1  # 捕食率
Pd = 0.3  # 捕食的概率

# 初始化珊瑚礁
reef = np.full(reef_size, None)  # 珊瑚礁矩阵
num_initial_corals = int(reef_size[0] * reef_size[1] * initial_occupation_rate)
corals = np.random.uniform(-10, 10, num_initial_corals)  # 随机生成初始解
fitness = np.array([objective_function(c) for c in corals])

# 将初始珊瑚放入珊瑚礁
for idx, coral in enumerate(corals):
    i, j = np.unravel_index(idx, reef_size)
    reef[i, j] = coral

# 进化过程
for iteration in range(num_iterations):
    # 1. 性繁殖（广播繁殖和内繁殖）
    larvae = []
    num_broadcast_spawning = int(Fb * len(corals))
    for _ in range(num_broadcast_spawning):
        parent1, parent2 = np.random.choice(corals, 2, replace=False)
        child = (parent1 + parent2) / 2 + np.random.normal(0, 0.1)
        larvae.append(child)

    # 2. 无性繁殖
    num_asexual_spawning = int(Fa * len(corals))
    for _ in range(num_asexual_spawning):
        parent = np.random.choice(corals)
        child = parent + np.random.normal(0, 0.1)
        larvae.append(child)

    # 3. 幼虫沉积
    for larva in larvae:
        i, j = np.random.randint(0, reef_size[0]), np.random.randint(0, reef_size[1])
        if reef[i, j] is None or objective_function(larva) < objective_function(
            reef[i, j]
        ):
            reef[i, j] = larva

    # 4. 捕食（移除适应度最差的部分珊瑚）
    all_corals = [c for c in reef.flatten() if c is not None]
    num_to_predate = int(Fd * len(all_corals))
    if num_to_predate > 0:
        worst_indices = np.argsort([objective_function(c) for c in all_corals])[
            -num_to_predate:
        ]
        for idx in worst_indices:
            i, j = np.unravel_index(idx, reef_size)
            if np.random.rand() < Pd:
                reef[i, j] = None

    # 更新珊瑚列表
    corals = [c for c in reef.flatten() if c is not None]

# 找到适应度最小的珊瑚
best_coral = min(corals, key=objective_function)
best_fitness = objective_function(best_coral)

# 输出结果
print(f"找到的最小值为: {best_fitness:.4f}, 对应的x值为: {best_coral:.4f}")
