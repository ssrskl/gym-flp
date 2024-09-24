from itertools import permutations, product
import itertools
import random
from matplotlib import patches
import numpy as np
import gym
import pickle
import os
import math
import matplotlib.pyplot as plt

from gym import spaces
from numpy.random import default_rng
from PIL import Image
from gym_flp import rewards
from gym_flp.util import FBSUtils

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


class FbsEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array", "human"]}

    def __init__(
        self,
        mode=None,
        instance=None,
        distance=None,
        aspect_ratio=None,
        step_size=None,
        greenfield=None,
        box=False,
        multi=False,
    ):
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        # 加载实例数据
        (
            self.problems,  # 问题模型
            self.FlowMatrices,  # 流量矩阵
            self.sizes,  # 尺寸数据
            self.LayoutWidths,  # 布局宽度 X W
            self.LayoutLengths,  # 布局长度 Y H
        ) = pickle.load(
            open(
                os.path.join(
                    __location__, "instances/continual", "maoyan_cont_instances.pkl"
                ),
                "rb",
            )
        )
        self.mode = mode
        self.max_steps = 100000  # 设置最大步数
        self.current_step = 0
        self.best_fitness = float("inf")
        self.no_improve_threshold = 1000  # 连续1000步没有改善就结束

        self.instance = instance
        # 确保实例存在，否则提示用户选择实例
        while not (
            self.instance in self.FlowMatrices.keys() or self.instance in ["Brewery"]
        ):
            print("Available Problem Sets:", self.FlowMatrices.keys())
            self.instance = input("选择一个问题模型:").strip()  # 清除输入两端的空格
        self.F = self.FlowMatrices[self.instance]  # 物流强度矩阵
        # self.F = FBSUtils.transfer_matrix(self.F)  # 物流强度矩阵转换
        self.n = self.problems[self.instance]  # 问题模型的设施数量
        # 获得面积数据（横纵比，长度，宽度，面积，最小长度）
        self.fac_limit_aspect, self.l, self.w, self.area, self.min_side_length = (
            FBSUtils.getAreaData(self.sizes[self.instance])
        )
        self.L = int(self.LayoutLengths[self.instance])
        self.W = int(self.LayoutWidths[self.instance])

        self.min_length = 1
        self.min_width = 1

        self.action_space = spaces.Discrete(5)
        self.actions = {
            0: "Randomize",  # 随机交换两个元素
            1: "Bit Swap",  # 将bay中的随机的1转换为0，或者将0转换为1
            2: "Bay Exchange",  # 随机选择两个不同的bay，并交换它们的位置
            3: "Inverse",  # 设施在某个bay中随机选择，并翻转它们的位置
            4: "Shuffle",  # 随机打乱某个bay中的设施顺序
            5: "Repair",  # 修复某个设施的位置
            # 6: "Idle",  # 什么都不做
        }
        self.bay_space = spaces.Box(low=0, high=1, shape=(self.n,), dtype=np.int)

        self.state = None
        self.permutation = None  # 设施的排列顺序
        self.bay = None
        self.done = False
        self.MHC = rewards.mhc.MHC()
        # 观察空间的模式
        if self.mode == "rgb_array":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.W, self.L, 3), dtype=np.uint8
            )  # Image representation
        elif self.mode == "human":

            observation_low = np.tile(
                np.array([0, 0, self.min_length, self.min_width], dtype=int), self.n
            )
            observation_high = np.tile(
                np.array([self.W, self.L, self.W, self.L], dtype=int), self.n
            )
            self.observation_space = spaces.Box(
                low=observation_low, high=observation_high, dtype=int
            )  # Vector representation of coordinates
        else:
            print("Nothing correct selected")

    def reset(self, layout=None, best_fitness=float("inf")):
        self.current_step = 0
        self.best_fitness = float("inf")
        self.no_improve_count = 0
        self.best_fitness = best_fitness
        # 如果layout不为空，则使用layout
        if layout is not None:
            self.permutation, self.bay = layout
        else:
            # self.permutation, self.bay = self.sampler() # 生成一个随机排列和bay
            self.permutation, self.bay = FBSUtils.binary_solution_generator(
                self.area, self.n, self.fac_limit_aspect, self.L
            )  # 采用k分初始解生成器
            self.bay[-1] = 1  # bay的最后一个位置必须是1，表示最后一个设施是bay的结束
        (
            self.fac_x,
            self.fac_y,
            self.fac_b,
            self.fac_h,
            self.fac_aspect_ratio,
            self.D,
            self.TM,
            self.MHC,
            self.Fitness,
        ) = FBSUtils.StatusUpdatingDevice(
            self.permutation, self.bay, self.area, self.W, self.F, self.fac_limit_aspect
        )
        self.state = self.constructState(
            self.fac_x, self.fac_y, self.fac_b, self.fac_h, self.n
        )
        return self.state

    # 构建状态，
    def constructState(self, x, y, l, w, n):
        # 顺序转换
        x = x[self.permutation - 1]
        y = y[self.permutation - 1]
        l = l[self.permutation - 1]
        w = w[self.permutation - 1]
        state_prelim = np.zeros((4 * n,), dtype=float)
        state_prelim[0::4] = y
        state_prelim[1::4] = x
        state_prelim[2::4] = w
        state_prelim[3::4] = l

        if self.mode == "human":
            self.state = np.array(state_prelim)

        elif self.mode == "rgb_array":
            self.state = self.ConvertCoordinatesToState(state_prelim)

        return self.state[:]

    def ConvertCoordinatesToState(self, state_prelim):
        data = (
            np.zeros((self.observation_space.shape))
            if self.mode == "rgb_array"
            else np.zeros((self.W, self.L, 3), dtype=np.uint8)
        )

        sources = np.sum(self.TM, axis=1)
        sinks = np.sum(self.TM, axis=0)

        R = np.array(
            (self.permutation - np.min(self.permutation))
            / (np.max(self.permutation) - np.min(self.permutation))
            * 255
        ).astype(int)
        G = np.array(
            (sources - np.min(sources)) / (np.max(sources) - np.min(sources)) * 255
        ).astype(int)
        B = np.array(
            (sinks - np.min(sinks)) / (np.max(sinks) - np.min(sinks)) * 255
        ).astype(int)

        # 存储设施数据
        labels = np.zeros(len(self.permutation))  # 设施编号
        positions = np.zeros((len(self.permutation), 4))
        aspect_ratio = np.zeros(len(self.permutation))
        for x, p in enumerate(self.permutation):
            x_from = state_prelim[4 * x + 1] - 0.5 * state_prelim[4 * x + 3]
            y_from = state_prelim[4 * x + 0] - 0.5 * state_prelim[4 * x + 2]
            x_to = state_prelim[4 * x + 1] + 0.5 * state_prelim[4 * x + 3]
            y_to = state_prelim[4 * x + 0] + 0.5 * state_prelim[4 * x + 2]
            # print("设施编号: ", p)
            # print("x_from: ", x_from)
            # print("y_from: ", y_from)
            # print("x_to: ", x_to)
            # print("y_to: ", y_to)
            # 存储到labels和positions中
            labels[x] = p
            positions[x] = [x_from, y_from, x_to, y_to]
            x_length = x_to - x_from
            y_length = y_to - y_from
            aspect_ratio[x] = max(x_length, y_length) / min(x_length, y_length)
            data[int(y_from) : int(y_to), int(x_from) : int(x_to)] = [
                R[p - 1],
                G[p - 1],
                B[p - 1],
            ]
        # print("labels: ", labels)
        # print("positions: ", positions)
        # 创建图形和坐标轴
        fig, ax = plt.subplots()
        # 绘制设施
        for i, label in enumerate(labels):
            x_from, y_from, x_to, y_to = positions[i]
            # 创建设施，如果横纵比超出范围则使用红色
            # print("设施编号: ", int(label))
            # print(f"横纵比: {aspect_ratio[i]}")
            # print(f"最大宽高比: {self.fac_limit_aspect.max()}")
            if aspect_ratio[i] > self.fac_limit_aspect.max():
                rect = patches.Rectangle(
                    (x_from, y_from, x_to - x_from, y_to - y_from),
                    width=x_to - x_from,
                    height=y_to - y_from,
                    edgecolor="red",
                    facecolor="none",
                    angle=0.5,
                )
            else:
                rect = patches.Rectangle(
                    (x_from, y_from, x_to - x_from, y_to - y_from),
                    width=x_to - x_from,
                    height=y_to - y_from,
                    edgecolor="green",
                    facecolor="none",
                    angle=0,
                )

            ax.add_patch(rect)
            # 添加标签
            ax.text(
                x_from + (x_to - x_from) / 2,
                y_from + (y_to - y_from) / 2,
                f"{int(label)}, AR={aspect_ratio[i]:.2f}",
                ha="center",
                va="center",
            )
        ax.set_title("设施布局图")
        ax.set_xlabel("X轴")
        ax.set_ylabel("Y轴")
        # 显示MHC
        plt.figtext(
            0.5,
            0.93,
            "MHC: {:.2f}".format(FBSUtils.getMHC(self.D, self.F, self.permutation)),
            ha="center",
            fontsize=12,
        )
        # 显示Fitness
        plt.figtext(
            0.5,
            0.96,
            "Fitness: {:.2f}".format(
                FBSUtils.getFitness(
                    self.MHC, self.fac_b, self.fac_h, self.fac_limit_aspect
                )
            ),
            ha="center",
            fontsize=12,
        )
        # 设置坐标范围
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.W)
        # 添加网格
        plt.grid(False)
        plt.gca().set_aspect("equal", adjustable="box")
        # 显示图形
        plt.show()
        return np.array(data, dtype=np.uint8)

    # 随机生成一个排列和布局
    def sampler(self):
        return (
            default_rng().choice(
                range(1, self.n + 1), size=self.n, replace=False
            ),  # 生成一个随机排列，即随机顺序的整数数组
            self.bay_space.sample(),
        )

    def step(self, action):
        self.current_step += 1

        a = self.actions[action]  # 根据动作索引获取动作名称
        fromState = np.array(self.permutation)  # 原先的排列
        facilities = np.where(self.bay == 1)[0]  # 查找bay为1的位置，即区带的划分点
        bay_breaks = np.split(
            self.bay, facilities[:-1] + 1
        )  # 将bay按照划分点分割成多个子数组，每个子数组代表一个区带的bay
        bays = np.split(
            self.permutation, facilities[:-1] + 1
        )  # 将排列按照划分点分割成多个子数组，每个子数组代表一个区段的排列

        if a == "Randomize":
            # 随机选择排列中的两个元素，并交换它们的位置
            k = default_rng().choice(
                range(len(self.permutation) - 1), size=1, replace=False
            )
            l = default_rng().choice(
                range(len(self.permutation) - 1), size=1, replace=False
            )
            fromState[k], fromState[l] = fromState[l], fromState[k]
            self.permutation = np.array(fromState)

        elif a == "Bit Swap":
            # 将bay中的随机的1转换为0，或者将0转换为1
            j = default_rng().choice(
                range(len(self.bay - 1)), size=1, replace=False
            )  # 随机选择一个区带序号

            temp_bay = np.array(self.bay)  # Make a copy of bay
            temp_bay[j] = 1 if temp_bay[j] == 0 else 0

            self.bay = np.array(temp_bay)

        elif a == "Bay Exchange":
            # 随机选择两个不同的bay，并交换它们的位置
            o = int(default_rng().choice(range(len(bays)), size=1, replace=False))
            p = int(default_rng().choice(range(len(bays)), size=1, replace=False))
            # while p == o:  # 确保不在同一个区带，但是会造成死循环，例如只有一个区带
            # p = int(default_rng().choice(range(len(bays)), size=1, replace=False))
            if p == o:  # 确保不在同一个区带，但是会造成死循环，例如只有一个区带
                p = int(default_rng().choice(range(len(bays)), size=1, replace=False))
            # Swap bays and break points accordingly:
            bays[o], bays[p] = bays[p], bays[o]
            bay_breaks[o], bay_breaks[p] = bay_breaks[p], bay_breaks[o]

            new_bay = np.concatenate(bay_breaks)
            new_state = np.concatenate(bays)

            # Make sure state is saved as copy
            self.permutation = np.array(new_state)
            self.bay = np.array(new_bay)

        elif a == "Inverse":
            # 设施在某个bay中随机选择，并翻转它们的位置
            q = default_rng().choice(range(len(bays)))  # 随机选择一个区带
            bays[q] = np.flip(bays[q])  # 反转，例如[1,2,3,4,5] -> [5,4,3,2,1]

            new_bay = np.concatenate(bay_breaks)
            new_state = np.concatenate(bays)

            # Make sure state is saved as copy
            self.permutation = np.array(new_state)
            self.bay = np.array(new_bay)
        elif a == "Shuffle":
            # 随机打乱某个bay中的设施顺序
            q = default_rng().choice(range(len(bays)))  # 随机选择一个区带
            bays[q] = default_rng().choice(
                bays[q], size=len(bays[q]), replace=False
            )  # 打乱顺序
            new_bay = np.concatenate(bay_breaks)
            new_state = np.concatenate(bays)

            # Make sure state is saved as copy
            self.permutation = np.array(new_state)
            self.bay = np.array(new_bay)
        elif a == "Repair":
            # 随机选择一个bay，判断其中的设施满足条件的是否超过一半，如果不超过一般，则判断是太宽了还是太窄了，然后进行相应的调整
            # 如果太宽了，说明这个bay中的设施过多，则将其对半分（太宽：横坐标长度/纵坐标长度 > 横纵比）
            # 如果太窄了，说明这个bay中的设施过少，则将当前bay与相邻的bay进行合并（太窄：纵坐标长度/横坐标长度 > 横纵比）
            # 如果满足条件，则保持不变
            # 查找有不合格设施的bay
            # -------------查找不合格的bay-------------
            invalidBay = []
            for q in range(len(bays)):
                bay_length = len(bays[q])
                facilities_in_bay = bays[q]
                current_beta = self.fac_limit_aspect[facilities_in_bay - 1]
                actual_ratio = self.fac_aspect_ratio[facilities_in_bay - 1]
                # 如果有对应不合规的设施
                irregularityNum = np.sum(
                    np.logical_or(
                        actual_ratio < current_beta[:, 0],
                        actual_ratio > current_beta[:, 1],
                    )
                )
                if irregularityNum > 0:
                    invalidBay.append(q)
            # print("不合格的bay:", invalidBay)
            if len(invalidBay) > 0:
                q = invalidBay[random.randint(0, len(invalidBay) - 1)]
                # print("不合格的bay:", q)
                # -------------查找不合格bay中的设施-------------
                facilities_in_bay = bays[q]
                invalid_facilities = []
                for facility in facilities_in_bay:
                    current_beta = self.fac_limit_aspect[facility - 1]
                    actual_ratio = self.fac_aspect_ratio[facility - 1]
                    if actual_ratio < current_beta[0] or actual_ratio > current_beta[1]:
                        invalid_facilities.append(facility)
                # print("不合格的设施:", invalid_facilities)
                # -------------修复不合格bay中的设施-------------
                facility = invalid_facilities[
                    random.randint(0, len(invalid_facilities) - 1)
                ]
                # 判断设施是太宽还是太窄
                if (
                    self.fac_b[facility - 1] / self.fac_h[facility - 1]
                ) > self.fac_limit_aspect[facility - 1][1]:
                    # 设施太宽了，需要分割
                    target_bay_length = len(bay_breaks[q])
                    # 将当前bay从中间分割为两个bay
                    split_point = target_bay_length // 2 - 1
                    bay_breaks[q][split_point] = 1  # 设置分割点
                    # print("Bay太宽，已分割")
                else:
                    # 设施太窄了，需要合并，如果当前bay是最后一个bay，则与上一个bay合并，如果当前bay是第一个bay，则与下一个bay合并，否则与随机与左右合并
                    if q == len(bays) - 1:
                        # 与上一个bay合并， 将上一个bay的最后一位1设置为0
                        bays[q - 1] = np.concatenate((bays[q - 1], bays[q]))
                        bay_breaks[q - 1][-1] = 0
                        bay_breaks[q - 1] = np.concatenate(
                            (bay_breaks[q - 1], bay_breaks[q])
                        )
                        del bays[q]
                        del bay_breaks[q]
                    elif q == 0:
                        # 与下一个bay合并，将当前bay的最后一位1设置为0
                        bays[q + 1] = np.concatenate((bays[q + 1], bays[q]))
                        bay_breaks[q][-1] = 0
                        bay_breaks[q + 1] = np.concatenate(
                            (bay_breaks[q + 1], bay_breaks[q])
                        )
                        del bays[q]
                        del bay_breaks[q]
                    else:
                        # 与左右随机合并
                        if random.randint(0, 1) == 0:
                            # 与左边合并
                            bays[q - 1] = np.concatenate((bays[q - 1], bays[q]))
                            bay_breaks[q - 1][-1] = 0
                            bay_breaks[q - 1] = np.concatenate(
                                (bay_breaks[q - 1], bay_breaks[q])
                            )
                        else:
                            # 与右边合并
                            bays[q + 1] = np.concatenate((bays[q + 1], bays[q]))
                            bay_breaks[q][-1] = 0
                            bay_breaks[q + 1] = np.concatenate(
                                (bay_breaks[q + 1], bay_breaks[q])
                            )
                        del bays[q]
                        del bay_breaks[q]
                # 更新permutation和bay
                self.permutation = np.concatenate(bays)
                self.bay = np.concatenate(bay_breaks)
            else:
                pass
                # print("没有不合格的bay")
        # 当设施变换完成之后，重新计算适应度函数
        (
            self.fac_x,
            self.fac_y,
            self.fac_b,
            self.fac_h,
            self.fac_aspect_ratio,
            self.D,
            self.TM,
            self.MHC,
            self.Fitness,
        ) = FBSUtils.StatusUpdatingDevice(
            self.permutation, self.bay, self.area, self.W, self.F, self.fac_limit_aspect
        )
        self.state = self.constructState(
            self.fac_x, self.fac_y, self.fac_b, self.fac_h, self.n
        )

        # reward = self.best_fitness - self.Fitness
        # if reward > 0:
        #     self.best_fitness = self.Fitness
        #     self.no_improve_count = 0
        # else:
        #     reward = 0
        #     self.no_improve_count += 1
        # # 结束条件
        # self.done = (
        #     self.current_step >= self.max_steps
        #     or self.no_improve_count >= self.no_improve_threshold
        # )
        if self.MHC == self.Fitness:
            self.done = True
        else:
            self.done = False
        reward = self.MHC - self.Fitness
        # reward = -self.Fitness
        info = {"mhc": self.MHC, "fitness": self.Fitness, "TimeLimit.truncated": False}
        if self.current_step >= self.max_steps:
            info["TimeLimit.truncated"] = True
        return (
            self.state[:],
            reward,
            self.done,
            info,
        )

    def render(self, mode=None):
        if self.mode == "human":
            data = self.ConvertCoordinatesToState(self.state[:])
            img = Image.fromarray(data, "RGB")

        if self.mode == "rgb_array":
            data = self.state[:]
            img = Image.fromarray(self.state, "RGB")

        plt.imshow(img)
        # 显示MHC
        plt.figtext(
            0.5,
            0.95,
            "MHC: {:.2f}".format(FBSUtils.getMHC(self.D, self.F, self.permutation)),
            ha="center",
            fontsize=12,
        )
        # plt.axis('off') # 关闭坐标轴
        plt.show()

        return img
