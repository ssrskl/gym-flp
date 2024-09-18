import numpy as np


class MHC:

    def __init__(self, shape=None, dtype=np.float32):
        self.shape = shape

    def compute(self, D, F, s):
        # Compute reward for taking action on old state:
        # Calculate permutation matrix for new state
        print("Computing MHC reward...")
        P = self.permutationMatrix(s)
        # Deduct results from known optimal value
        # reward = self.best if np.array_equal(fromState, self.opt) else self.best - 0.5*np.trace(np.dot(np.dot(self.F,P), np.dot(self.D,P.T))) #313 is optimal for NEOS n6, needs to be replaced with a dict object
        print(F)

        transport_intensity = np.dot(np.dot(D, P), np.dot(F, P.T))
        MHC = np.trace(transport_intensity)

        return MHC, transport_intensity

    def _compute(self, D, F, s):
        T = np.zeros((len(s), len(s)))
        for i in range(len(s)):
            for j in range(len(s)):
                if j > i:
                    d = D[i][j]
                    f = F[s[i] - 1][s[j] - 1]
                    T[i][j] = d * f
                else:
                    T[i][j] = 0
        return np.sum(T), T

    def _computeplus(self, D, F, s):
        # print("Computing MHCPlus reward...")
        # print("s:", s)
        P = self.permutationMatrix(s)
        # print("P: ", P)
        # print("D: ", D)
        # print("F: ", F)
        # print("dot(P,D): ", np.dot(P, D))
        # Deduct results from known optimal value
        # reward = self.best if np.array_equal(fromState, self.opt) else self.best - 0.5*np.trace(np.dot(np.dot(self.F,P), np.dot(self.D,P.T))) #313 is optimal for NEOS n6, needs to be replaced with a dict object
        # P.T·D·P得到按照1,2,3..这样的顺序的距离矩阵,再使用tril转换为下三角矩阵,再与物流矩阵的转置相乘，因为物流强度矩阵为上三角矩阵，所以需要转置,最后求和得到MHC
        transport_intensity = np.dot(np.dot(D, P), np.dot(F, P.T))
        # print("下三角距离矩阵：", np.tril(np.dot(P.T, np.dot(D, P))))
        # print("物流矩阵：", F.T)
        MHC = np.sum(np.tril(np.dot(P.T, np.dot(D, P))) * (F.T))

        return MHC, transport_intensity

        """获得适应度值
        Args:
            D: 距离矩阵
            F: 物流矩阵
            s: 选择的设施
            fac_b: 设施的宽度
            fac_h: 设施的高度
            fac_aspect_ratio: 设施的长宽比
            
        """

    def getMHC(self, D, F, s):
        P = self.permutationMatrix(s)
        MHC = np.sum(np.tril(np.dot(P.T, np.dot(D, P))) * (F.T))
        return MHC

    def getFitness(self, D, F, s, fac_b, fac_h, fac_aspect_ratio):
        aspect_ratio_list = []
        k = 3
        non_feasible_counter = 0
        # P矩阵用于将后续的距离矩阵转换为设施ID升序的距离矩阵，方便与物流矩阵相乘
        P = self.permutationMatrix(s)
        MHC = np.sum(np.tril(np.dot(P.T, np.dot(D, P))) * (F.T))

        if fac_aspect_ratio is None:
            for i, (b, h) in enumerate(zip(fac_b, fac_h)):
                if b < 1 or h < 1:
                    non_feasible_counter += 1
        else:
            for i, (b, h) in enumerate(zip(fac_b, fac_h)):
                facility_aspect_ratio = max(b, h) / min(b, h)
                aspect_ratio_list.append(facility_aspect_ratio)
                if not (
                    min(fac_aspect_ratio[i])
                    <= facility_aspect_ratio
                    <= max(fac_aspect_ratio[i])
                ):
                    non_feasible_counter += 1
        aspect_ratio = np.array(aspect_ratio_list)
        # print("aspect_ratio: ", aspect_ratio)
        fitness = MHC + MHC * (non_feasible_counter**k)
        return fitness

    def permutationMatrix(self, a):
        # print("a的值: ", a)
        P = np.zeros((len(a), len(a)))
        for idx, val in enumerate(a):
            P[idx][val - 1] = 1
        return P

    def getDistances(self, x, y):
        return np.array(
            [
                [
                    abs(float(x[j]) - float(valx)) + abs(float(valy) - float(y[i]))
                    for (j, valy) in enumerate(y)
                ]
                for (i, valx) in enumerate(x)
            ],
            dtype=float,
        )


