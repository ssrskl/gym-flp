import gym
import gym_flp
import numpy as np


def load_instance_data(instance):
    if instance == "AB20-ar3":
        permutation = np.array(
            [20, 6, 2, 4, 18, 5, 8, 7, 19, 3, 9, 12, 15, 14, 10, 11, 13, 1, 16, 17]
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
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
            ]
        )
    elif instance == "Du62":
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
    elif instance == "O7-maoyan":
        permutation = np.array([3, 5, 7, 1, 4, 6, 2])
        bay = np.array([0, 0, 1, 0, 0, 0, 1])
    elif instance == "AEG20":
        permutation = np.array(
            [12, 20, 18, 16, 9, 7, 8, 15, 19, 6, 5, 13, 2, 10, 14, 4, 17, 3, 1, 11]
        )
        bay = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    else:
        raise ValueError(f"Unknown instance: {instance}")
    return permutation, bay


instance = "AB20-ar3"
permutation, bay = load_instance_data(instance)
env = gym.make("fbs-v0", instance=instance, mode="human")

env.reset(layout=(permutation, bay))
env.render()
env.close()
