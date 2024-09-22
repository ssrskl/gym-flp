# 实验数据保存测试
from datetime import datetime
import os

import numpy as np
from gym_flp.util.TSExperimentDataGenerator import TSExperimentDataGenerator


base_path = r"E:\projects\pythonprojects\gym-flp\algorithm\src\gym-flp"  # 使用原始字符串避免转义问题
file_path = os.path.join(
    base_path, "ExperimentResult", "TS", "ts-convergence-stage-test-testNumber.xlsx"
)  # 使用os.path.join构建路径
print(file_path)
best_permutation = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
best_bay = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
TSExperimentDataGenerator(
    experiment_name="TS-Convergence-Stage-Test",
    experiment_id=1,
    start_time=datetime.now(),
    end_time=datetime.now(),
    duration=0,
    best_permutation=best_permutation,
    best_bay=best_bay,
    best_result=100,
    tabu_list_size=10,
    num_iterations=100,
    step_size=10,
).saveExcel(file_path)
