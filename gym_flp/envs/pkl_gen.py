# pkl实例数据生成器
import os
import pickle

import numpy as np
import pandas as pd

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
areas_file_dir_path = os.path.join(__location__, "instances/continual/areas")
flows_file_dir_path = os.path.join(__location__, "instances/continual/flows")

# 设施数量字典
facilities_number_dict = {}
# 设施信息字典
facilities_info = {}
# 厂房宽度字典
layout_width_dict = {}
# 厂房长度字典
layout_height_dict = {}
# 流量信息字典
flows_info = {}

# 遍历areas目录下的所有文件
for filename in os.listdir(areas_file_dir_path):
    # 检查文件是否以.prn结尾
    if filename.endswith(".prn"):
        # 打开文件
        prn_file_path = os.path.join(areas_file_dir_path, filename)
        # 文件名格式化
        filename = filename.replace(".prn", "")
        # 读取文件内容
        with open(prn_file_path, "r") as file:
            lines = file.readlines()
            facility_number = int(lines[0].strip())
            facilities_number_dict[filename] = facility_number
            title_values = lines[2].strip().split()
            # 提取行数据
            data_lines = [
                line.strip().split() for line in lines[3 : 3 + facility_number]
            ]
            # 封装为DataFrame
            df = pd.DataFrame(data_lines, columns=title_values, dtype=float)
            # 添加到字典中
            facilities_info[filename] = df
            # 如果厂房尺寸信息存在，则添加到layout_info字典中
            print(f"filename: {filename}")
            if len(lines) > 4 + facility_number:
                # W
                layout_width = float(lines[4 + facility_number].strip().split()[1])
                # H
                layout_height = float(lines[5 + facility_number].strip().split()[1])
                print(f"layout_width: {layout_width}, layout_height: {layout_height}")
                # 封装到厂房尺寸字典中
                layout_width_dict[filename] = layout_width
                layout_height_dict[filename] = layout_height

# 遍历flows目录下的所有文件
for filename in os.listdir(flows_file_dir_path):
    # 检查文件是否以.prn结尾
    if filename.endswith(".prn"):
        # 打开文件
        prn_file_path = os.path.join(flows_file_dir_path, filename)
        # 文件名格式化
        filename = filename.replace(".prn", "")
        # 读取文件内容
        with open(prn_file_path, "r") as file:
            lines = file.readlines()
            facility_number = int(lines[0].strip())
            flows_array_data = [
                line.strip().split() for line in lines[2 : 2 + facility_number]
            ]
            flows_array = np.array(flows_array_data, dtype=float)
            # 封装到流量信息字典中
            flows_info[filename] = flows_array

    # print(f"facilities_info: {facilities_info}")
    # print(f"layout_width_dict: {layout_width_dict}")
    # print(f"layout_height_dict: {layout_height_dict}")
    # print(f"flows_info: {flows_info}")

    # 打包为pickle文件
pickle_file_path = os.path.join(
    __location__, "instances/continual", "maoyan_cont_instances.pkl"
)
with open(pickle_file_path, "wb") as file:
    pickle.dump(
        [
            facilities_number_dict,
            flows_info,
            facilities_info,
            layout_height_dict,
            layout_width_dict,
        ],
        file=file,
    )
