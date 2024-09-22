from datetime import datetime
import os
import pandas as pd
from gym_flp.util.ExperimentDataGenerator import ExperimentDataGenerator
import numpy as np

class TSExperimentDataGenerator(ExperimentDataGenerator):
    def __init__(
        self,
        experiment_name,
        experiment_id,
        start_time,
        end_time,
        duration,
        best_permutation,
        best_bay,
        best_result,
        # TS参数
        tabu_list_size,
        num_iterations,
        step_size,
    ):
        super().__init__(
            experiment_name,
            experiment_id,
            start_time,
            end_time,
            duration,
            best_permutation,
            best_bay,
            best_result,
        )
        self.tabu_list_size = tabu_list_size
        self.num_iterations = num_iterations
        self.step_size = step_size

    def saveExcel(self, file_path):
        data = {
            "实验次数": self.experiment_id,
            "开始时间": self.start_time,
            "结束时间": self.end_time,
            "持续时间": self.duration,
            "禁忌表大小": self.tabu_list_size,
            "迭代次数": self.num_iterations,
            "邻域步长": self.step_size,
            "最优排列": np.array2string(self.best_permutation, separator=','),
            "最优区带": np.array2string(self.best_bay, separator=','),
            "最优结果": self.best_result,
        }
        df = pd.DataFrame(data, index=[0])  # 添加 index=[0] 以确保 DataFrame 只有一行

        # 转换日期列为日期时间格式
        df["开始时间"] = pd.to_datetime(df["开始时间"])
        df["结束时间"] = pd.to_datetime(df["结束时间"])

        # 如果目录不存在，则创建目录
        base_path = os.path.dirname(file_path)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        try:
            with pd.ExcelWriter(
                file_path, engine="openpyxl", mode="a", if_sheet_exists="overlay", date_format="YYYY-MM-DD HH:MM:SS"
            ) as writer:
                # 获取现有数据的行数
                book = writer.book
                sheet = book.active
                startrow = sheet.max_row
                # 将新数据追加到最后一行
                df.to_excel(writer, index=False, header=False, startrow=startrow)
        except FileNotFoundError:
            # 如果文件不存在，则创建一个新的文件
            with pd.ExcelWriter(file_path, engine="openpyxl", mode="w", date_format="YYYY-MM-DD HH:MM:SS") as writer:
                df.to_excel(writer, index=False)
        print(f"实验数据已保存到 {file_path}")
