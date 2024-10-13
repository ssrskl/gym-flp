# 顺序交叉测试

import random


def order_crossover_with_bay(parent1, bay1, parent2, bay2):
    size = len(parent1)

    # 随机选择两个交叉点
    start, end = sorted(random.sample(range(size), 2))
    print(f"start:{start}, end:{end}")

    # 初始化子本和子bay为None
    child1 = [None] * size
    child2 = [None] * size
    child_bay1 = [None] * size
    child_bay2 = [None] * size

    # 将父本和父bay在交叉点之间的部分复制到子本中
    child1[start : end + 1] = parent1[start : end + 1]
    child_bay1[start : end + 1] = bay1[start : end + 1]
    child2[start : end + 1] = parent2[start : end + 1]
    child_bay2[start : end + 1] = bay2[start : end + 1]

    # 填充子本的剩余部分
    fill_positions = [i for i in range(size) if i < start or i > end]
    fill_values1 = [item for item in parent2 if item not in child1]
    # 查找bay1中与fill_values1中元素对应的值
    fill_bay_values1 = [bay1[parent1.index(item)] for item in fill_values1]
    for i, pos in enumerate(fill_positions):
        child1[pos] = fill_values1[i]
        child_bay1[pos] = fill_bay_values1[i]

    fill_values2 = [item for item in parent1 if item not in child2]
    fill_bay_values2 = [bay2[parent2.index(item)] for item in fill_values2]
    for i, pos in enumerate(fill_positions):
        child2[pos] = fill_values2[i]
        child_bay2[pos] = fill_bay_values2[i]

    return child1, child_bay1, child2, child_bay2


# 示例父本和对应的bay
parent1 = [6, 7, 8, 9, 1, 2, 3, 4, 5]
bay1 = [0, 0, 0, 1, 0, 0, 1, 0, 1]
parent2 = [3, 1, 4, 2, 5, 6, 7, 8, 9]
bay2 = [0, 0, 1, 0, 0, 0, 0, 1, 1]

# 执行顺序交叉
child1, child_bay1, child2, child_bay2 = order_crossover_with_bay(
    parent1, bay1, parent2, bay2
)

print(f"Parent 1:  {parent1}")
print(f"Bay 1:     {bay1}")
print(f"Parent 2:  {parent2}")
print(f"Bay 2:     {bay2}")
print(f"Child 1:   {child1}")
print(f"Child Bay 1:{child_bay1}")
print(f"Child 2:   {child2}")
print(f"Child Bay 2:{child_bay2}")
