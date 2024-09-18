# FBS模型中的辅助函数
import numpy as np


# 计算每个设施的长宽
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
    for b in bays:  # 遍历每一个区带中的设施
        areas = a[b - 1]  # 获得每个设施的大小
        end = start + len(areas)
        # 计算设施的长度，为x轴方向上的长度
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
