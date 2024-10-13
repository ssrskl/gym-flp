import pandas as pd
from datetime import datetime


class ExperimentDataSaver:
    def __init__(self, file_path="./实验结果/遗传算法实验数据.xlsx"):
        self.file_path = file_path

    def save_data(
        self,
        experiments_number,
        start_time,
        end_time,
        duration,
        population_size,
        generations,
        mutation_rate,
        best_permutation,
        best_bay,
        best_fitness,
    ):
        data = {
            "实验次数": experiments_number,
            "开始时间": start_time,
            "结束时间": end_time,
            "持续时间": duration,
            "种群大小": population_size,
            "迭代次数": generations,
            "变异率": mutation_rate,
            "最优排列": [best_permutation],
            "最优区带": [best_bay],
            "最优MHC": best_fitness,
        }
        df = pd.DataFrame(data)
        try:
            with pd.ExcelWriter(
                self.file_path, engine="openpyxl", mode="a", if_sheet_exists="overlay"
            ) as writer:
                # 获取现有数据的行数
                book = writer.book
                sheet = book.active
                startrow = sheet.max_row
                # 将新数据追加到最后一行
                df.to_excel(writer, index=False, header=False, startrow=startrow)
        except FileNotFoundError:
            # 如果文件不存在，则创建一个新的文件
            with pd.ExcelWriter(self.file_path, engine="openpyxl", mode="w") as writer:
                df.to_excel(writer, index=False)
        print(f"实验数据已保存到 {self.file_path}")
