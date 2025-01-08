import pandas as pd
import numpy as np
import psutil
import os

class SoundFieldData:
    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        self.x_column = data['x_distance'].values
        self.y_column = data['y_distance'].values
        self.spl_column = data['spl'].values

        # Handle NaN and Inf values
        min_numeric_value = np.nanmin(self.spl_column)
        max_numeric_value = np.nanmax(self.spl_column[np.isfinite(self.spl_column)])
        self.spl_column[np.isnan(self.spl_column)] = min_numeric_value
        self.spl_column[np.isinf(self.spl_column)] = max_numeric_value

        # Normalize sound intensity values
        mean_spl_true = np.mean(self.spl_column, axis=0)
        std_spl_true = np.std(self.spl_column, axis=0)
        self.norm_spl_column = (self.spl_column - mean_spl_true) / std_spl_true

    def get_spl(self, x, y):
        index = np.where((self.x_column == x) & (self.y_column == y))[0]
        if index.size > 0:
            return self.norm_spl_column[index[0]]
        else:
            raise ValueError(f"No data found for coordinates ({x}, {y}).")

    def get_data(self):
        return self.x_column, self.y_column, self.norm_spl_column, self.spl_column

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1e6:.2f} MB")