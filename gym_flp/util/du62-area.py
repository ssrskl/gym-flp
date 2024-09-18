# 读取文件
with open(
    r"E:\projects\pythonprojects\gym-flp\algorithm\src\gym-flp\gym_flp\util\Du62-area.prn",
    "r",
) as file:
    data = file.read()

# 将数据转换为二维列表
data_list = [line.split() for line in data.split("\n")]
# 将列表中的字符串转换为整数
data_list = [[int(item) for item in line] for line in data_list]

print(data_list)


# 写入到文件
with open(
    r"E:\projects\pythonprojects\gym-flp\algorithm\src\gym-flp\gym_flp\util\du62-area.txt",
    "w",
) as file:
    # 计算每一行的面积
    for line in data_list:
        area = 0
        for i in range(len(line) - 1):
            area += line[i] * line[i + 1]
        file.write(str(area) + "\n")
