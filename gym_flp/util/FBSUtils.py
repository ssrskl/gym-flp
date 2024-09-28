import random
import numpy as np
import re
from itertools import permutations, product


# 物流强度矩阵转换
def transfer_matrix(matrix: np.ndarray):
    """
    转置矩阵
    :param matrix: 矩阵
    :return: 转置后的矩阵
    """
    print("转换前: ", matrix)
    LowerTriangular = np.tril(matrix, -1).T
    resultMatrix = LowerTriangular + matrix
    resultMatrix = np.triu(resultMatrix)
    print("转换后: ", resultMatrix)
    return resultMatrix


# 获取面积数据
def getAreaData(df):
    # 检查Area数据是否存在，存在则转换为numpy数组，否则为None
    if np.any(df.columns.str.contains("Area", na=False, case=False)):
        a = df.filter(regex=re.compile("Area", re.IGNORECASE)).to_numpy()
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
        a = a[:, 0]
    a = np.reshape(a, (a.shape[0],))
    return ar, l, w, a, l_min


# k分初始解生成器(输入：面积数据a，设施数n，横纵比限制beta，厂房x轴长度L)
def binary_solution_generator(area, n, beta, L):
    # 存储可行的k分解
    bay_list = []
    # 分界参数
    k = 2
    # 计算面积之和
    total_area = np.sum(area)
    print("总面积: ", total_area)
    # 生成一个设施默认的编号序列
    permutation = np.arange(1, n + 1)
    # 根据area对序列进行排序
    permutation = permutation[np.argsort(area[permutation - 1])]
    # 对a也进行排序
    area = np.sort(area)
    # 对beta也按照a的顺序进行排序
    if beta is not None:
        beta = np.array([beta[i - 1] for i in permutation])
    while k <= n:
        # 计算W的k分
        l = L / k
        w = area / l  # 每个设施的宽度
        aspect_ratio = np.maximum(w, l) / np.minimum(w, l)
        # 验证k分是否可行
        # print("a/l", a / l)
        # 合格个数
        if beta is not None:
            qualified_number = np.sum(
                (aspect_ratio >= beta[:, 0]) & (aspect_ratio <= beta[:, 1])
            )
        else:
            qualified_number = np.sum((w > 1) & (l > 1))
        # 如果合格个数大于等于3/4*n，即此k值可行
        bay = np.zeros(n)
        if qualified_number >= n * 3 / 4:
            # print("可行的k: ", k)
            # print("符合的个数: ", qualified_number)
            # 根据面积和找到k分界点
            best_partition, partitions = _find_best_partition(area, k)
            # print("序列分界点: ", best_partition)
            # 将k分界点转换为bay
            for i, p in enumerate(best_partition):
                bay[p - 1] = 1
            # 将最后一个分界点设为1
            bay[n - 1] = 1
            bay_list.append(bay)
        k += 1
    # print("可行的bay: ", bay_list)
    # 从可行的bay中随机选择一个
    if len(bay_list) > 0:
        bay = random.choice(bay_list)
    #  TODO 对permutation使用bay进行划分，并对每个bay中的设施进行随机排列
    j = 0
    for i in range(len(bay)):
        if bay[i] == 1:
            np.random.shuffle(permutation[j:i])
            j = i + 1
    return (permutation, bay)


# k分划分法的动态规划版（输入：排列序列a，划分数k）
def _find_best_partition(arr, k):
    print(f"k分划分法-->k = {k}")
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


# 计算设施坐标和尺寸
def getCoordinates_mao(permutation, bay, a, W):
    facilities = np.where(bay == 1)[0]  # 查找bay为1的位置，即区带的划分点
    bays = np.split(
        permutation, indices_or_sections=facilities[:-1] + 1
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
    # 顺序恢复
    fac_x = fac_x[np.argsort(permutation)]
    fac_y = fac_y[np.argsort(permutation)]
    lengths = lengths[np.argsort(permutation)]
    widths = widths[np.argsort(permutation)]
    return fac_x, fac_y, lengths, widths


# 计算欧几里得距离矩阵
def getEuclideanDistances(x, y):
    """计算欧几里得距离矩阵
    Args:
        x (np.ndarray): 设施x坐标
        y (np.ndarray): 设施y坐标
    Returns:
        np.ndarray: 距离矩阵
    """
    return np.sqrt(
        np.array(
            [
                [(x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 for j in range(len(x))]
                for i in range(len(x))
            ]
        )
    )


# 计算曼哈顿距离矩阵
def getManhattanDistances(x, y):
    """计算曼哈顿距离矩阵
    Args:
        x (np.ndarray): 设施x坐标
        y (np.ndarray): 设施y坐标
    """
    return np.array(
        [
            [abs(x[i] - x[j]) + abs(y[i] - y[j]) for j in range(len(x))]
            for i in range(len(x))
        ],
        dtype=float,
    )


def permutationMatrix(a):
    P = np.zeros((len(a), len(a)))
    for idx, val in enumerate(a):
        P[idx][val - 1] = 1
    return P


def getTransportIntensity(D, F, s):
    P = permutationMatrix(s)
    return np.dot(np.dot(D, P), np.dot(F, P.T))


def getMHC(D, F, permutation):
    P = permutationMatrix(permutation)
    # MHC = np.sum(np.tril(np.dot(P.T, np.dot(D, P))) * (F.T))
    MHC = np.sum(np.triu(D) * (F))
    return MHC


def getFitness(mhc, fac_b, fac_h, fac_limit_aspect):
    aspect_ratio_list = []
    k = 1
    non_feasible_counter = 0
    MHC = mhc

    if fac_limit_aspect is None:
        for i, (b, h) in enumerate(zip(fac_b, fac_h)):
            if b < 1 or h < 1:
                non_feasible_counter += 1
    else:
        for i, (b, h) in enumerate(zip(fac_b, fac_h)):
            facility_aspect_ratio = max(b, h) / min(b, h)
            aspect_ratio_list.append(facility_aspect_ratio)
            if not (
                min(fac_limit_aspect[i])
                <= facility_aspect_ratio
                <= max(fac_limit_aspect[i])
            ):
                non_feasible_counter += 1
    aspect_ratio = np.array(aspect_ratio_list)
    fitness = MHC + MHC * (non_feasible_counter**k)
    return fitness


def StatusUpdatingDevice(permutation, bay, a, W, F, fac_limit_aspect_ratio):
    fac_x, fac_y, fac_b, fac_h = getCoordinates_mao(permutation, bay, a, W)
    fac_aspect_ratio = np.maximum(fac_b, fac_h) / np.minimum(fac_b, fac_h)
    D = getManhattanDistances(fac_x, fac_y)
    TM = getTransportIntensity(D, F, permutation)
    mhc = getMHC(D, F, permutation)
    fitness = getFitness(mhc, fac_b, fac_h, fac_limit_aspect_ratio)
    return (fac_x, fac_y, fac_b, fac_h, fac_aspect_ratio, D, TM, mhc, fitness)


# 全排列局部优化
def fullPermutationOptimization(permutation, bay, a, W, D, F, fac_limit_aspect):
    # 对当前的状态进行局部搜索，返回新的状态和适应度函数值
    # print("开始局部搜索优化")
    # 局部搜索优化，全排列每一个bay中的设施，并计算适应度函数值，选择最优的排列
    best_perm = np.array(permutation)
    best_fitness = float("inf")
    split_indices = np.where(bay == 1)[0] + 1
    split_indices = split_indices[split_indices < len(permutation)]
    bays = np.split(permutation, split_indices)
    # print("bays:", bays)
    perms = [list(permutations(bay)) for bay in bays]  # 对每个bay中的设施进行全排列
    # 对排列后的结果进行笛卡尔积进行组合
    combinations = list(product(*perms))
    combined_permutations = [list(comb) for comb in combinations]
    for perm in combined_permutations:
        convert_perm = np.concatenate(perm)
        print("convert_perm:", convert_perm)
        # 计算当前排列下的设施参数信息
        facx, facy, facb, fach = getCoordinates_mao(convert_perm, bay, a, W)
        MHC = getMHC(D, F, convert_perm)
        # 计算适应度函数值
        fitness = getFitness(MHC, facb, fach, fac_limit_aspect)
        # print("当前排列下的设施参数信息: ", facx, facy, facb, fach)
        # print("当前排列下的适应度函数值: ", fitness)
        if fitness < best_fitness:
            best_fitness = fitness
            best_perm = convert_perm
    # print("局部搜索优化后的最优排列: ", best_perm)
    # print("局部搜索优化后的最优适应度函数值: ", best_fitness)
    return np.array(best_perm)

    # k分划分法
    # def find_best_partition(self, arr, k):
    #     target_sum = np.sum(arr) // k
    #     n = len(arr)
    #     best_diff = float("inf")
    #     best_partition = None

    #     for comb in itertools.combinations(range(1, n), k - 1):
    #         partitions = np.split(arr, comb)
    #         partition_sums = [np.sum(part) for part in partitions]
    #         diff = sum(abs(target_sum - s) for s in partition_sums)

    #         if diff < best_diff:
    #             best_diff = diff
    #             best_partition = comb

    #     return best_partition, np.split(arr, best_partition)


# 交换局部优化算法
def exchangeOptimization(
    permutation: np.ndarray,
    bay: np.ndarray,
    a,
    W,
    D,
    F,
    fac_limit_aspect,
):
    best_perm = permutation.copy()  # 最佳排列
    best_fitness = float("inf")  # 最佳适应度函数值
    improved = True  # 标记是否有改进
    while improved:
        improved = False
        for i in range(len(permutation) - 1):
            new_perm = permutation.copy()
            new_perm[i], new_perm[i + 1] = new_perm[i + 1], new_perm[i]
            # 计算当下排列的适应度函数值
            mhc = getMHC(D, F, new_perm)
            fac_x, fac_y, fac_b, fac_h = getCoordinates_mao(new_perm, bay, a, W)
            fitness = getFitness(mhc, fac_b, fac_h, fac_limit_aspect)
            if fitness < best_fitness:
                best_fitness = fitness
                best_perm = new_perm
                improved = True
        permutation = best_perm.copy()
    return best_perm


# # 相邻交换局部优化算法
# def adjacent_exchange(self):
#     # 对当前的状态进行局部搜索，返回新的状态和适应度函数值
#     best_perm = np.array(self.permutation)
#     perm = np.array(self.permutation)
#     best_fitness = self.getFitness()

#     for i in range(len(self.permutation) - 1):
#         perm = np.array(self.permutation)
#         perm[i], perm[i + 1] = perm[i + 1], perm[i]
#         # 计算当前排列下的设施参数信息
#         facx, facy, facb, fach = FBSUtils.getCoordinates_mao(
#             self.bay, perm, self.a, self.W
#         )
#         # 计算距离矩阵
#         D = self.MHC.getDistances(facx, facy)
#         # 计算适应度函数值
#         fitness = self.MHC.getFitness(D, self.F, perm, facb, fach, self.beta)
#         if fitness < best_fitness:
#             best_fitness = fitness
#             best_perm = perm
#     # print("局部搜索优化后的最优排列: ", best_perm)
#     # print("局部搜索优化后的最优适应度函数值: ", best_fitness)
#     self.permutation = best_perm


# 将排列和bay转换为二维数组，例如：
# 输入：permutation = [1,2,3,4,5,6,7] bay = [0,0,0,1,0,0,1]
# 输出：array = [[1,2,3,4],[5,6,7]]
def permutationToArray(permutation, bay):
    array = []
    start = 0
    for i, val in enumerate(bay):
        if val == 1:
            array.append(permutation[start : i + 1])
            start = i + 1
    return array


# 将二维数组转换为排列和bay
def arrayToPermutation(array):
    permutation = []
    bay = []
    for sub_array in array:
        permutation.extend(sub_array)
        bay.extend([0] * (len(sub_array) - 1) + [1])
    return permutation, bay
