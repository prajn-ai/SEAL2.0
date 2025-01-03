import pandas as pd
import numpy as np

class SoundFieldData:
    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        self.x_column = data['x_distance'].values
        self.y_column = data['y_distance'].values
        spl_column = data['spl'].values

        # Handle NaN and Inf values
        min_numeric_value = np.nanmin(spl_column)
        max_numeric_value = np.nanmax(spl_column[np.isfinite(spl_column)])
        spl_column[np.isnan(spl_column)] = min_numeric_value
        spl_column[np.isinf(spl_column)] = max_numeric_value

        # Normalize sound intensity values
        mean_spl_true = np.mean(spl_column, axis=0)
        std_spl_true = np.std(spl_column, axis=0)
        self.norm_spl_column = (spl_column - mean_spl_true) / std_spl_true

    def get_spl(self, x, y):
        index = np.where((self.x_column == x) & (self.y_column == y))[0]
        if index.size > 0:
            return self.norm_spl_column[index[0]]
        else:
            raise ValueError(f"No data found for coordinates ({x}, {y}).")

    def get_data(self):
        return self.x_column, self.y_column, self.norm_spl_column, self.spl_column
