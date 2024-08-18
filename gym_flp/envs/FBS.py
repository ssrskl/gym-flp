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

        self.max_steps = 10000  # 设置最大步数
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
        self.n = self.problems[self.instance]  # 问题模型的设施数量
        self.AreaData = self.sizes[self.instance]  # 面积数据
        # 获得面积数据（横纵比，长度，宽度，面积，最小长度）
        self.beta, self.l, self.w, self.a, self.min_side_length = getAreaData(
            self.AreaData
        )  # Investigate available area data and compute missing values if needed
        # print("面积数据:", self.beta)
        # print("长度限制条件:", self.l)
        # print("宽度限制条件:", self.w)
        # print("面积:", self.a)
        # print("最小长度:", self.min_side_length)
        # 如果有L和W的信息则使用，否则则将A设置为所有的设施面积之和，并且设置一个正方形布局，通过计算得到L和W
        if (
            self.instance in self.LayoutWidths.keys()
            and self.instance in self.LayoutLengths.keys()
        ):
            self.L = int(
                self.LayoutLengths[self.instance]
            )  # We need both values to be integers for converting into image
            self.W = int(self.LayoutWidths[self.instance])
        else:
            self.A = np.sum(self.a)
            self.L = int(
                round(math.sqrt(self.A), 0)
            )  # We want the plant dimensions to be integers to fit them into an image
            self.W = self.L

            # Design a layout with l = 1,5 * w
            # self.L = divisor(int(self.A))
            # self.W = self.A/self.L

        # These values need to be set manually, e.g. acc. to data from literature. Following Eq. 1 in Ulutas & Kulturel-Konak (2012), the minimum side length can be determined by assuming the smallest facility will occupy alone.
        # self.aspect_ratio = int(max(self.beta)) if not self.beta is None else 1
        self.min_length = np.min(self.a) / self.L
        self.min_width = np.min(self.a) / self.W

        # We define minimum side lengths to be 1 in order to be displayable in array
        self.min_length = 1
        self.min_width = 1

        self.action_space = spaces.Discrete(4)
        self.actions = {
            0: "Randomize",  # 随机交换两个元素
            1: "Bit Swap",  # 将bay中的随机的1转换为0，或者将0转换为1
            2: "Bay Exchange",  #
            3: "Inverse",
            4: "Repair",
        }
        # self.state_space = spaces.Box(low=1, high = self.n, shape=(self.n,), dtype=np.int)
        self.bay_space = spaces.Box(
            low=0, high=1, shape=(self.n,), dtype=np.int
        )  # binary vector indicating bay breaks (i = 1 means last facility in bay)

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

    def reset(self, layout=None):

        self.current_step = 0
        self.best_fitness = float("inf")
        self.no_improve_count = 0

        # 如果layout不为空，则使用layout
        if layout is not None:
            self.permutation, self.bay = layout
        # if flag:
        #     # O7的最佳设置
        #     # self.permutation = np.array([3, 5, 7, 1, 4, 6, 2])
        #     # self.bay = np.array([0, 0, 1, 0, 0, 0, 1])
        #     # self.permutation = np.array([5, 6, 7, 1, 2, 3, 4])
        #     # self.bay = np.array([0, 0, 0, 0, 1, 0, 1])
        #     # O8的最佳设置
        #     # self.permutation = np.array([3, 6, 8, 5, 7, 4, 1, 2])
        #     # self.bay = np.array([0, 0, 0, 1, 0, 0, 0, 1])
        #     # O9的最佳设置
        #     # self.permutation = np.array([5, 1, 2, 9, 6, 4, 8, 3, 7])
        #     # self.bay = np.array([0, 1, 0, 0, 0, 1, 0, 0, 1])
        #     self.permutation = np.array([6, 7, 8, 9, 1, 2, 3, 4, 5])
        #     self.bay = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1])
        #     # VC10设置
        #     # self.permutation = np.array([5, 8, 6, 4, 9, 10, 2, 3, 7, 1])
        #     # self.bay = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
        #     # AEG20 设置
        #     # self.permutation = np.array(
        #     #     [12, 20, 18, 16, 9, 7, 8, 15, 19, 6, 5, 13, 2, 10, 14, 4, 17, 3, 1, 11]
        #     # )
        #     # self.bay = np.array(
        #     #     [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        #     # )
        else:
            # 生成一个随机排列和bay
            # self.permutation, self.bay = self.sampler()
            # 采用k分初始解生成器
            self.permutation, self.bay = self.binary_solution_generator()
            # print(type(self.permutation))
            print("初始排列: ", self.permutation)
            print("初始区带布局: ", self.bay)
            # bay的最后一个位置必须是1，表示最后一个设施是bay的结束
            self.bay[-1] = 1
        self.fac_x, self.fac_y, self.fac_b, self.fac_h = self.getCoordinates()
        self.D = self.MHC.getDistances(self.fac_x, self.fac_y)
        reward, self.TM = self.MHC._computeplus(self.D, self.F, self.permutation[:])
        self.Fitness = self.MHC.getFitness(
            self.D, self.F, self.permutation[:], self.fac_b, self.fac_h, self.beta
        )
        self.state = self.constructState(
            self.fac_x, self.fac_y, self.fac_b, self.fac_h, self.n
        )

        return self.state

    def getMHC(self):
        self.fac_x, self.fac_y, self.fac_b, self.fac_h = self.getCoordinates()
        self.D = self.MHC.getDistances(self.fac_x, self.fac_y)
        # print("fac_x: ", self.fac_x)
        # print("fac_y: ", self.fac_y)
        # print("fac_b: ", self.fac_b)
        # print("fac_h: ", self.fac_h)
        # print("距离: ", self.MHC._getDistances(self.fac_x, self.fac_y))
        mhc, self.TM = self.MHC._computeplus(self.D, self.F, self.permutation[:])
        return mhc

    def getFitness(self):
        self.fac_x, self.fac_y, self.fac_b, self.fac_h = self.getCoordinates()
        self.D = self.MHC.getDistances(self.fac_x, self.fac_y)
        return self.MHC.getFitness(
            self.D, self.F, self.permutation[:], self.fac_b, self.fac_h, self.beta
        )

    # 构建状态
    def constructState(self, x, y, l, w, n):

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

        labels = np.zeros(len(self.permutation))
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
            if aspect_ratio[i] > 3:
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
                str(label),
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
            "MHC: {:.2f}".format(
                self.MHC._computeplus(self.D, self.F, self.permutation)[0]
            ),
            ha="center",
            fontsize=12,
        )
        # 显示Fitness
        plt.figtext(
            0.5,
            0.96,
            "Fitness: {:.2f}".format(
                self.MHC.getFitness(
                    self.D,
                    self.F,
                    self.permutation[:],
                    self.fac_b,
                    self.fac_h,
                    self.beta,
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

    # k分初始解生成器
    def binary_solution_generator(self):
        print("k分初始解生成器")
        # print(self.n)
        # print(self.L)
        # print(self.W)
        # print(self.a)
        # print(self.beta)
        # 存储可行的k分解
        bay_list = []
        # 分界参数
        k = 2
        # 计算面积之和
        total_area = np.sum(self.a)
        # 生成一个设施默认的编号序列
        permutation = np.arange(1, self.n + 1)
        # 根据a对序列进行排序
        permutation = permutation[np.argsort(self.a[permutation - 1])]
        # 对a也进行排序
        a = np.sort(self.a)
        # 对beta也按照a的顺序进行排序
        beta = np.array([self.beta[i - 1] for i in permutation])
        # print("排序后的排列: ", permutation)
        # print("排序后的a: ", a)
        # print("self.a: ", self.a)
        while k <= self.n:
            # 计算W的k分
            l = self.L / k
            # print("l: ", l)
            w = a / l  # 每个设施的宽度
            # print("w: ", w)
            # print("w/l: ", np.maximum(w, l) / np.minimum(w, l))
            aspect_ratio = np.maximum(w, l) / np.minimum(w, l)
            # 验证k分是否可行
            # print("a/l", a / l)
            # 合格个数
            qualified_number = np.sum(
                (aspect_ratio >= beta[:, 0]) & (aspect_ratio <= beta[:, 1])
            )
            # print(f"k: {k}, 合格个数: {qualified_number}")
            # 如果合格个数大于等于3/4*n，即此k值可行
            if qualified_number >= self.n * 3 / 4:
                # print("可行的k: ", k)
                # print("符合的个数: ", qualified_number)
                # 根据面积和找到k分界点
                best_partition, partitions = self._find_best_partition(a, k)
                # print("序列分界点: ", best_partition)
                # 将k分界点转换为bay
                bay = np.zeros(self.n)
                for i, p in enumerate(best_partition):
                    bay[p - 1] = 1
                # 将最后一个分界点设为1
                bay[self.n - 1] = 1
                bay_list.append(bay)
            k += 1

        # print("可行的bay: ", bay_list)
        # 从可行的bay中随机选择一个
        if len(bay_list) > 0:
            bay = random.choice(bay_list)
        return (permutation, bay)

    # k分划分法
    def find_best_partition(self, arr, k):
        target_sum = np.sum(arr) // k
        n = len(arr)
        best_diff = float("inf")
        best_partition = None

        for comb in itertools.combinations(range(1, n), k - 1):
            partitions = np.split(arr, comb)
            partition_sums = [np.sum(part) for part in partitions]
            diff = sum(abs(target_sum - s) for s in partition_sums)

            if diff < best_diff:
                best_diff = diff
                best_partition = comb

        return best_partition, np.split(arr, best_partition)

    # k分划分法的动态规划版
    def _find_best_partition(self, arr, k):
        print("k分划分法")
        n = len(arr)
        target_sum = np.sum(arr) // k

        # dp[i][j] 表示前i个设施被划分为j个组的最小差异和
        dp = np.full((n + 1, k + 1), float("inf"))
        dp[0][0] = 0

        # sum[i] 表示arr[0:i]的累积和
        cum_sum = np.cumsum(arr)

        partition_idx = [[[] for _ in range(k + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(1, k + 1):
                for m in range(i):
                    current_sum = cum_sum[i - 1] - (cum_sum[m - 1] if m > 0 else 0)
                    current_diff = abs(target_sum - current_sum)
                    total_diff = dp[m][j - 1] + current_diff

                    if total_diff < dp[i][j]:
                        dp[i][j] = total_diff
                        partition_idx[i][j] = partition_idx[m][j - 1] + [i]

        best_partition = partition_idx[-1][-1][:-1]  # 排除最后一个分界点
        return best_partition, np.split(arr, best_partition)

    # 随机生成一个排列和布局
    def sampler(self):
        return (
            default_rng().choice(
                range(1, self.n + 1), size=self.n, replace=False
            ),  # 生成一个随机排列，即随机顺序的整数数组
            self.bay_space.sample(),
        )

    # 计算设施在布局中的坐标和尺寸
    def getCoordinates(self):
        facilities = np.where(self.bay == 1)[0]  # 查找bay为1的位置，即区带的划分点
        bays = np.split(
            self.permutation, facilities[:-1] + 1
        )  # 将排列按照划分点分割成多个子数组，每个子数组代表一个区段的排列

        lengths = np.zeros(
            (
                len(
                    self.permutation,
                )
            )
        )
        widths = np.zeros(
            (
                len(
                    self.permutation,
                )
            )
        )
        fac_x = np.zeros(
            (
                len(
                    self.permutation,
                )
            )
        )
        fac_y = np.zeros(
            (
                len(
                    self.permutation,
                )
            )
        )

        x = 0
        start = 0
        for b in bays:  # 遍历每一个区带中的设施
            # 将b格式化为整形numpy数组
            b = np.array(b, dtype=int)
            areas = self.a[b - 1]  # 获取当前区带中的设施面积
            end = start + len(areas)

            # 计算每个设施的长度和宽度
            lengths[start:end] = (
                np.sum(areas) / self.W
            )  # Calculate all facility widhts in bay acc. to Eq. (1) in https://doi.org/10.1016/j.eswa.2011.11.046
            widths[start:end] = areas / lengths[start:end]

            fac_x[start:end] = lengths[start:end] * 0.5 + x
            x += np.sum(areas) / self.W

            y = np.ones(len(b))
            ll = 0
            for idx, l in enumerate(widths[start:end]):
                y[idx] = ll + 0.5 * l
                ll += l
            fac_y[start:end] = y

            start = end

        return fac_x, fac_y, lengths, widths

    def getCoordinates_mao(self, bay, permutation, a, W):
        facilities = np.where(bay == 1)[0]  # 查找bay为1的位置，即区带的划分点
        bays = np.split(
            permutation, facilities[:-1] + 1
        )  # 将排列按照划分点分割成多个子数组，每个子数组代表一个区段的排列

        lengths = np.zeros(
            (
                len(
                    permutation,
                )
            )
        )
        widths = np.zeros(
            (
                len(
                    permutation,
                )
            )
        )
        fac_x = np.zeros(
            (
                len(
                    permutation,
                )
            )
        )
        fac_y = np.zeros(
            (
                len(
                    permutation,
                )
            )
        )

        x = 0
        start = 0
        # print("当前Bays: ", bays)
        # print("当前Permutation: ", permutation)
        for b in bays:  # 遍历每一个区带中的设施
            areas = a[b - 1]  # Get the area associated with the facilities
            # print("areas: ", areas)
            end = start + len(areas)

            # 计算每个设施的长度和宽度
            lengths[start:end] = (
                np.sum(areas) / W
            )  # Calculate all facility widhts in bay acc. to Eq. (1) in https://doi.org/10.1016/j.eswa.2011.11.046
            widths[start:end] = areas / lengths[start:end]

            fac_x[start:end] = lengths[start:end] * 0.5 + x
            x += np.sum(areas) / W

            y = np.ones(len(b))
            ll = 0
            for idx, l in enumerate(widths[start:end]):
                y[idx] = ll + 0.5 * l
                ll += l
            fac_y[start:end] = y

            start = end

        return fac_x, fac_y, lengths, widths

    def step(self, action):
        self.current_step += 1

        a = self.actions[action]  # 根据动作索引获取动作名称
        # print("当前动作: ", a)
        # k = np.count_nonzero(self.bay)
        fromState = np.array(self.permutation)  # 原先的排列
        facilities = np.where(self.bay == 1)[0]  # 查找bay为1的位置，即区带的划分点
        bay_breaks = np.split(
            self.bay, facilities[:-1] + 1
        )  # 将bay按照划分点分割成多个子数组，每个子数组代表一个区带的bay
        bays = np.split(
            self.permutation, facilities[:-1] + 1
        )  # 将排列按照划分点分割成多个子数组，每个子数组代表一个区段的排列

        # print("当前的行为: ", a)
        # print("FromState: ", fromState)
        # print("facilities: ", facilities)
        # print(
        #     "bay_breaks: ", bay_breaks
        # )  # bay_breaks:  [array([0, 0, 0, 1]), array([0, 0, 0, 0, 1]), array([0, 0, 0, 0, 0, 0, 1]), array([0, 0, 0, 1])]
        # print(
        #     "Bays: ", bays
        # )  # Bays:  [array([12, 20, 18, 16]), array([ 9,  7,  8, 15, 19]), array([ 6,  5, 13,  2, 10, 14,  4]), array([17,  3,  1, 11])]
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
            bays[q] = np.flip(bays[q])

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
            # print("开始修复")
            q = default_rng().choice(range(len(bays)))  # 随机选择一个区带
            bay_length = len(bays[q])
            facilities_in_bay = bays[q]
            print("当前bay中的设施编号:", facilities_in_bay)

            # 获取当前bay中设施的规定长宽比
            current_beta = self.beta[facilities_in_bay - 1]
            # print("当前bay中的设施的规定长宽比:", current_beta)

            # 获得子排列在排列中的索引
            indices = np.where(np.isin(self.permutation, facilities_in_bay))[0]
            # 获得子排列的设施横坐标长度和纵坐标长度
            fac_b_bay = self.fac_b[indices]
            fac_h_bay = self.fac_h[indices]
            # actual_ratio = fac_b_bay / fac_h_bay
            actual_ratio = np.maximum(fac_b_bay, fac_h_bay) / np.minimum(
                fac_b_bay, fac_h_bay
            )

            print("当前bay中的设施的实际长宽比:", actual_ratio)

            # 判断满足条件的设施数量
            satisfied = np.sum(
                (actual_ratio >= current_beta[:, 0])
                & (actual_ratio <= current_beta[:, 1])
            )
            print("当前bay中满足条件的设施数量:", satisfied)
            if satisfied < bay_length / 2:
                # 如果满足条件的设施少于一半

                # 判断是否太宽
                if fac_b_bay.mean() / fac_h_bay.mean() > self.beta.mean():
                    # 如果平均实际比率大于平均规定比率，说明太宽了，需要分割
                    split_point = bay_length // 2
                    new_bay1 = facilities_in_bay[:split_point]
                    new_bay2 = facilities_in_bay[split_point:]

                    # 更新bays和bay_breaks
                    bays[q] = new_bay1
                    bays.insert(q + 1, new_bay2)
                    bay_breaks[q][-1] = 1  # 设置分割点
                    bay_breaks.insert(q + 1, np.zeros(len(new_bay2)))
                    bay_breaks[q + 1][-1] = 1

                    print("Bay太宽，已分割")
                else:
                    # 如果平均实际比率小于平均规定比率，说明太窄了，需要合并
                    if q < len(bays) - 1:
                        # 与下一个bay合并
                        bays[q] = np.concatenate((bays[q], bays[q + 1]))
                        bay_breaks[q] = np.concatenate(
                            (bay_breaks[q][:-1], bay_breaks[q + 1])
                        )
                        del bays[q + 1]
                        del bay_breaks[q + 1]
                    elif q > 0:
                        # 与上一个bay合并
                        bays[q - 1] = np.concatenate((bays[q - 1], bays[q]))
                        bay_breaks[q - 1] = np.concatenate(
                            (bay_breaks[q - 1][:-1], bay_breaks[q])
                        )
                        del bays[q]
                        del bay_breaks[q]

                    print("Bay太窄，已合并")
            else:
                print("当前bay满足条件，无需修复")

            # 更新permutation和bay
            self.permutation = np.concatenate(bays)
            self.bay = np.concatenate(bay_breaks)

            print("修复后的排列:", self.permutation)
            print("修复后的bay:", self.bay)

        # elif a == "Idle":
        #     pass  # 保持状态不变
        # 对当前的状态进行局部搜索，返回新的状态和适应度函数值
        # self.local_search()

        # 当设施变换完成之后，重新计算适应度函数
        self.fac_x, self.fac_y, self.fac_b, self.fac_h = self.getCoordinates()
        # print("当前设施信息: ", self.fac_x, self.fac_y, self.fac_b, self.fac_h)
        # print("通用计算下的当前设施信息: ", facx, facy, facb, fach)
        self.D = self.MHC.getDistances(self.fac_x, self.fac_y)
        # mhc, self.TM = self.MHC.compute(self.D, self.F, fromState)
        mhc, self.TM = self.MHC._computeplus(self.D, self.F, fromState)
        self.Fitness = self.MHC.getFitness(
            self.D, self.F, self.permutation[:], self.fac_b, self.fac_h, self.beta
        )
        self.state = self.constructState(
            self.fac_x, self.fac_y, self.fac_b, self.fac_h, self.n
        )
        # 训练收敛优化
        # if self.Fitness == self.MHC.getMHC(self.D, self.F, self.permutation[:]):
        #     self.done = True
        # else:
        #     self.done = False

        # 训练最优优化
        if self.Fitness < self.best_fitness:
            self.best_fitness = self.Fitness
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
        self.done = False
        if (
            self.current_step >= self.max_steps
            or self.no_improve_count >= self.no_improve_threshold
        ):
            self.done = True
        reward = self.best_fitness - self.Fitness
        info = {"mhc": mhc, "fitness": self.Fitness, "TimeLimit.truncated": False}
        if self.current_step >= self.max_steps:
            info["TimeLimit.truncated"] = True
        return (
            self.state[:],
            -self.Fitness,
            self.done,
            info,
        )

    # 全排列局部优化
    def local_search(self):
        # 对当前的状态进行局部搜索，返回新的状态和适应度函数值
        # print("开始局部搜索优化")
        # 局部搜索优化，全排列每一个bay中的设施，并计算适应度函数值，选择最优的排列
        best_perm = np.array(self.permutation)
        best_fitness = float("inf")
        split_indices = np.where(self.bay == 1)[0] + 1
        split_indices = split_indices[split_indices < len(self.permutation)]
        bays = np.split(self.permutation, split_indices)
        # print("bays:", bays)
        perms = [list(permutations(bay)) for bay in bays]  # 对每个bay中的设施进行全排列
        # 对排列后的结果进行笛卡尔积进行组合
        combinations = list(product(*perms))
        combined_permutations = [list(comb) for comb in combinations]
        for perm in combined_permutations:
            convert_perm = np.concatenate(perm)
            # print("convert_perm:", convert_perm)
            # 计算当前排列下的设施参数信息
            facx, facy, facb, fach = self.getCoordinates_mao(
                self.bay, convert_perm, self.a, self.W
            )
            # 计算距离矩阵
            D = self.MHC.getDistances(facx, facy)
            # 计算适应度函数值
            fitness = self.MHC.getFitness(
                D, self.F, convert_perm, facb, fach, self.beta
            )
            # print("当前排列下的设施参数信息: ", facx, facy, facb, fach)
            # print("当前排列下的适应度函数值: ", fitness)
            if fitness < best_fitness:
                best_fitness = fitness
                best_perm = convert_perm
        # print("局部搜索优化后的最优排列: ", best_perm)
        # print("局部搜索优化后的最优适应度函数值: ", best_fitness)
        self.permutation = np.array(best_perm)
        # print("局部搜索优化完成:", self.permutation)

    # 相邻交换局部优化算法
    def adjacent_exchange(self):
        # 对当前的状态进行局部搜索，返回新的状态和适应度函数值
        best_perm = np.array(self.permutation)
        perm = np.array(self.permutation)
        best_fitness = self.getFitness()

        for i in range(len(self.permutation) - 1):
            perm = np.array(self.permutation)
            perm[i], perm[i + 1] = perm[i + 1], perm[i]
            # 计算当前排列下的设施参数信息
            facx, facy, facb, fach = self.getCoordinates_mao(
                self.bay, perm, self.a, self.W
            )
            # 计算距离矩阵
            D = self.MHC.getDistances(facx, facy)
            # 计算适应度函数值
            fitness = self.MHC.getFitness(D, self.F, perm, facb, fach, self.beta)
            if fitness < best_fitness:
                best_fitness = fitness
                best_perm = perm
        # print("局部搜索优化后的最优排列: ", best_perm)
        # print("局部搜索优化后的最优适应度函数值: ", best_fitness)
        self.permutation = best_perm

    def render(self, mode=None):
        if self.mode == "human":

            # Mode 'human' needs intermediate step to convert state vector into image array
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
            "MHC: {:.2f}".format(
                self.MHC._computeplus(self.D, self.F, self.permutation)[0]
            ),
            ha="center",
            fontsize=12,
        )
        # plt.axis('off') # 关闭坐标轴
        plt.show()

        return img

    def close(self):
        pass


# 获得面积数据
def getAreaData(df):
    import re

    # 检查Area数据是否存在，存在则转换为numpy数组，否则为None
    if np.any(df.columns.str.contains("Area", na=False, case=False)):
        a = df.filter(regex=re.compile("Area", re.IGNORECASE)).to_numpy()
        # a = np.reshape(a, (a.shape[0],))

    else:
        a = None

    if np.any(df.columns.str.contains("Length", na=False, case=False)):
        l = df.filter(regex=re.compile("Length", re.IGNORECASE)).to_numpy()
        l = np.reshape(l, (l.shape[0],))

    else:
        l = None

    if np.any(df.columns.str.contains("Width", na=False, case=False)):
        w = df.filter(regex=re.compile("Width", re.IGNORECASE)).to_numpy()
        w = np.reshape(w, (w.shape[0],))

    else:
        w = None
    # 横纵比
    if np.any(df.columns.str.contains("Aspect", na=False, case=False)):
        ar = df.filter(regex=re.compile("Aspect", re.IGNORECASE)).to_numpy()
        # print("横纵比数据: ", ar)
        # ar = np.reshape(a, (a.shape[0],))

    else:
        ar = None

    l_min = 1  # 最小长度
    # 面积数据不存在，则根据长度和宽度计算面积
    if a is None:
        if not l is None and not w is None:
            a = l * w
        elif not l is None:
            a = l * max(l_min, max(l))
        else:
            a = w * max(l_min, max(w))

    # 如果横纵比存在上下限则不变，否则下限设置为1
    if not ar is None and ar.ndim > 1:
        if ar.shape[1] == 1:
            ar = np.hstack((np.ones((ar.shape[0], 1)), ar))
        else:
            pass
    if not a is None and a.ndim > 1:
        # a = a[np.where(np.max(np.sum(a, axis = 0))),:]
        a = a[
            :, 0
        ]  # We choose the maximum value here. Can be changed if something else is needed

    a = np.reshape(a, (a.shape[0],))

    return ar, l, w, a, l_min
