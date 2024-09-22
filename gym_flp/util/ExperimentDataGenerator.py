# 实验数据生成器基类
import random
import string
from datetime import datetime, timedelta


class ExperimentDataGenerator:
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
    ):
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.best_permutation = best_permutation
        self.best_bay = best_bay
        self.best_result = best_result

    def end_experiment(self):
        self.end_time = datetime.now()
        self.duration = self.end_time - self.start_time

    def set_best_results(self, best_permutation, best_bay, best_result):
        self.best_permutation = best_permutation
        self.best_bay = best_bay
        self.best_result = best_result

    def get_experiment_data(self):
        return {
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "best_permutation": self.best_permutation,
            "best_bay": self.best_bay,
            "best_result": self.best_result,
        }
