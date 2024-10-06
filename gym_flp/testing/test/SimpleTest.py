import gym
import gym_flp
import numpy as np

from gym_flp.util import FBSUtils
from gym_flp.util import AuxiliaryUtils


def load_instance_data(instance, case):
    if instance == "O7-maoyan":
        if case == "1":
            return np.array([3, 5, 7, 1, 4, 6, 2]), np.array([0, 0, 1, 0, 0, 0, 1])
        elif case == "2":
            return np.array([7, 5, 3, 2, 6, 4, 1]), np.array([0, 0, 1, 0, 0, 0, 1])
    if instance == "O8":
        return np.array([7, 4, 1, 2, 3, 6, 8, 5]), np.array([0, 0, 0, 1, 0, 0, 0, 1])
    if instance == "O9-maoyan":
        if case == "1":
            return np.array([3, 1, 6, 9, 5, 4, 2, 7, 8]), np.array(
                [0, 0, 0, 0, 1, 0, 1, 0, 1]
            )
        elif case == "2":
            return np.array([7, 8, 4, 1, 2, 3, 6, 9, 5]), np.array(
                [0, 1, 0, 0, 1, 0, 0, 0, 1]
            )
    if instance == "AB20-ar3":
        permutation = np.array(
            [20, 18, 5, 8, 7, 2, 4, 6, 10, 3, 19, 1, 14, 9, 12, 15, 13, 17, 11, 16]
        )
        bay = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1])
    elif instance == "AB20":
        permutation = np.array(
            [20, 18, 5, 8, 7, 2, 4, 6, 10, 3, 19, 1, 14, 9, 12, 15, 13, 17, 11, 16]
        )
        bay = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1])
    elif instance == "Du62":
        if case == "1":
            permutation = np.array(
                [
                    6,
                    41,
                    13,
                    10,
                    7,
                    39,
                    22,
                    51,
                    25,
                    49,
                    48,
                    4,
                    36,
                    20,
                    42,
                    53,
                    45,
                    23,
                    35,
                    3,
                    56,
                    21,
                    38,
                    12,
                    28,
                    1,
                    61,
                    58,
                    62,
                    26,
                    34,
                    50,
                    60,
                    32,
                    16,
                    11,
                    57,
                    2,
                    43,
                    27,
                    44,
                    54,
                    33,
                    8,
                    30,
                    18,
                    5,
                    59,
                    24,
                    52,
                    29,
                    47,
                    14,
                    17,
                    9,
                    19,
                    31,
                    55,
                    40,
                    37,
                    46,
                    15,
                ]
            )
            bay = np.array(
                [
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                ]
            )
            array = FBSUtils.permutationToArray(permutation, bay)
            permutation, bay = FBSUtils.arrayToPermutation(array)
            permutation = np.array(permutation)
            bay = np.array(bay)
            return permutation, bay
        elif case == "2":
            array = [
                [45, 39, 6],
                [26, 35, 22, 4, 41],
                [33, 30, 60, 50, 36, 23, 13],
                [19, 51, 8, 18, 32, 20, 25, 53],
                [34, 55, 5, 24, 56, 21, 16, 3, 43, 29, 49, 7],
                [31, 47, 12, 59, 38, 61, 48],
                [9, 52, 28, 57, 11, 1, 44, 14],
                [15, 37, 27, 2, 40, 42, 10],
                [46, 58, 54],
                [17, 62],
            ]
            permutation, bay = FBSUtils.arrayToPermutation(array)
            permutation = np.array(permutation)
            # 打印permutation使用逗号分隔
            print(",".join(map(str, permutation)))
            bay = np.array(bay)
            print(",".join(map(str, bay)))
    elif instance == "O7-maoyan":
        permutation = np.array([3, 5, 7, 1, 4, 6, 2])
        bay = np.array([0, 0, 1, 0, 0, 0, 1])
    elif instance == "AEG20":
        permutation = np.array(
            [12, 20, 18, 16, 9, 7, 8, 15, 19, 6, 5, 13, 2, 10, 14, 4, 17, 3, 1, 11]
        )
        bay = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    elif instance == "SC35-maoyan":
        if case == "1":
            permutation = np.array(
                [
                    3,
                    4,
                    18,
                    21,
                    15,
                    35,
                    13,
                    26,
                    16,
                    10,
                    31,
                    8,
                    32,
                    12,
                    9,
                    7,
                    20,
                    6,
                    25,
                    34,
                    24,
                    23,
                    19,
                    2,
                    5,
                    14,
                    28,
                    27,
                    30,
                    33,
                    11,
                    22,
                    17,
                    1,
                    29,
                ]
            )
            bay = np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            )
        elif case == "2":
            array = [
                [27, 21, 7, 8, 9, 11, 31, 16],
                [24, 2, 6, 5, 12, 32, 14],
                [23, 1, 25, 30, 10, 13],
                [19, 29, 17, 15],
                [28, 33, 18],
                [20, 34, 22, 4, 26],
                [3, 35],
            ]
            permutation, bay = FBSUtils.arrayToPermutation(array)
            permutation = np.array(permutation)
            bay = np.array(bay)
    else:
        raise ValueError(f"Unknown instance: {instance}")
    return permutation, bay


instance = "AB20-ar3"
case = "2"
permutation, bay = load_instance_data(instance, case)
env = gym.make("fbs-v0", instance=instance, mode="human")

env.reset(layout=(permutation, bay))
# 打印相关信息
print(f"距离矩阵：{env.D}")
print(f"流量矩阵：{env.F}")
# D的上三角矩阵
D_triu = np.triu(env.D)
print(f"D的上三角矩阵：{D_triu}")
print(f"计算的MHC为：{np.sum(D_triu * env.F)}")
AuxiliaryUtils.printMatrix(env.F)
AuxiliaryUtils.printMatrix(env.D)
env.render()
env.close()

# print(FBSUtils.permutationToArray(permutation, bay))
# print(FBSUtils.arrayToPermutation(FBSUtils.permutationToArray(permutation, bay)))
