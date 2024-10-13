# list = [(1, 2), (3, 4), (5, 6)]
# for i in list:
#     print(i)

# 测试np的全排列

import numpy as np
import itertools

list = [1, 2, 3, 4]
for i in itertools.permutations(list):
    print(i)
