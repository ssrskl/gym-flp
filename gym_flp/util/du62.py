# 读取du62.prn文件
with open(
    r"E:\projects\pythonprojects\gym-flp\algorithm\src\gym-flp\gym_flp\util\Du62.prn",
    "r",
) as file:
    data = file.read()

# 将数据转换为二维列表
data_list = [line.split() for line in data.split("\n")]
# 输出每行长度与第几行
print([(len(line), i + 1) for i, line in enumerate(data_list)])
# 如果长度不为62，则打印该行
for i, line in enumerate(data_list):
    if len(line) != 62:
        print("不等于62的行：", i + 1)

# 将列表中的字符串转换为整数
data_list = [[int(item) for item in line] for line in data_list]


# 写入文件， 用空格分割， 每行一个列表
with open(
    r"E:\projects\pythonprojects\gym-flp\algorithm\src\gym-flp\gym_flp\util\du62.txt",
    "w",
) as file:
    for line in data_list:
        file.write(" ".join(str(item) for item in line) + "\n")

print(data_list)
print(len(data_list))
