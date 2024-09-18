import gym
import gym_flp
import numpy as np

instance = "Du62"
# 加载模型
# 创建环境
env = gym.make("fbs-v0", instance=instance, mode="human")
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
obs = env.reset(layout=(permutation, bay))

env.render()
env.close()


# instance = "O7-maoyan"
# # 加载模型
# # 创建环境
# env = gym.make("fbs-v0", instance=instance, mode="human")
# permutation = np.array([3, 5, 7, 1, 4, 6, 2])
# bay = np.array([0, 0, 1, 0, 0, 0, 1])
# obs = env.reset(layout=(permutation, bay))

# env.render()
# env.close()
