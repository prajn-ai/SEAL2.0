import pandas as pd
import numpy as np

def load_data(file_path):
    data = pd.read_csv(file_path)
    x_column = data['x_distance'].values
    y_column = data['y_distance'].values
    sound_intensity_column = data['sound_intensity'].values

    # Handle NaN and Inf values
    min_numeric_value = np.nanmin(sound_intensity_column)
    max_numeric_value = np.nanmax(sound_intensity_column[np.isfinite(sound_intensity_column)])
    sound_intensity_column[np.isnan(sound_intensity_column)] = min_numeric_value
    sound_intensity_column[np.isinf(sound_intensity_column)] = max_numeric_value

    # Normalize sound intensity values
    mean_SIL_true = np.mean(sound_intensity_column, axis=0)
    std_SIL_true = np.std(sound_intensity_column, axis=0)
    sound_intensity_column = (sound_intensity_column - mean_SIL_true) / std_SIL_true

    return x_column, y_column, sound_intensity_column
