import torch
import gpytorch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils.data_utils import load_data, normalize_sound_intensity
from utils.gp_utils import initialize_gp_model, train_gp_model, evaluate_gp_model, RKHS_norm
from utils.plotting_utils import plot_gp_results
from utils.sampling_utils import PotentialNeighbors, SamplingMuAndVar, MutualInformationLocal, CheckConvergence

# Main Workflow
def main():
    data_file = '/home/prajna/SEAL/sound_field_data_2source.csv'
    x_column, y_column, sound_intensity_column = load_data(data_file)

    all_coordinates = np.column_stack((x_column, y_column))
    all_coordinates_tensor = torch.tensor(all_coordinates)

    sound_intensity_tensor = torch.tensor(sound_intensity_column, dtype=torch.float32)
    x_train, x_test, y_train, y_test = train_test_split(
        all_coordinates_tensor, sound_intensity_tensor, test_size=0.2, random_state=42
    )

    model, likelihood = train_gp_model(x_train, y_train, training_iter=75)
    observed_pred, f_mean, f_var, f_covar = evaluate_gp_model(model, likelihood, x_test)

    # Plot the results
    plot_all_graphs(coordinates_passed, sound_intensity_sampled, intensity_loss_2d,
                x_test_global, observed_pred_global, lower_local, upper_local,
                var_iter_local, var_iter_global, rmse_local_true, rmse_global_true,
                lengthscale, noise, covar_trace, covar_totelements, covar_nonzeroelements,
                AIC, BIC, f2_H_local, f2_H_global, x_max, image_path, iteration)
    
    # Example usage of utility functions
    neighbors = PotentialNeighbors(torch.tensor([0.0, 0.0]), torch.tensor([[1.0, 1.0], [2.0, 2.0]]), [])
    print("Neighbors:", neighbors)

if __name__ == '__main__':
    main()
