import numpy as np
import itertools


def find_best_partition(arr, k):
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


# 示例
arr = np.array([9, 9, 9, 9, 16, 16, 16, 36, 36])
k = 2

best_partition, partitions = find_best_partition(arr, k)
print("划分的下标:", best_partition)
for i, part in enumerate(partitions):
    print(f"第 {i+1} 组: {part}，和为 {np.sum(part)}")
