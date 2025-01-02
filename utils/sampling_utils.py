import torch
import numpy as np

def PotentialNeighbors(current_coordinate, potential_neighbors, coordinates_passed):
    """
    Identify potential neighboring coordinates.
    """
    current_coordinate = current_coordinate.numpy()
    potential_neighbors = potential_neighbors.numpy()

    all_potential_neighbors = current_coordinate + potential_neighbors
    tup_all_potential_neighbors = set(map(tuple, all_potential_neighbors))
    tup_coordinates_passed = set(map(tuple, coordinates_passed))

    filtered_neighbors = tup_all_potential_neighbors.difference(tup_coordinates_passed)
    return torch.tensor(list(filtered_neighbors), dtype=torch.float32)

def SamplingMuAndVar(var, mean, potential_neighbors):
    """
    Sampling strategy based on mean and variance.
    """
    alpha = 12
    beta = 1
    sampling_mu_var = beta * mean + alpha * var
    max_index = np.argmax(sampling_mu_var)
    next_sample_location = potential_neighbors[max_index]
    return next_sample_location, sampling_mu_var

def MutualInformationLocal(prev_var, curr_var, all_coordinates, potential_neighbors):
    """
    Compute mutual information locally.
    """
    mutual_information = 0.5 * np.log(prev_var / curr_var)
    max_index = np.argmax(mutual_information)
    next_sample_location = potential_neighbors[max_index]
    return next_sample_location, mutual_information

def CheckConvergence(rmse_history, threshold=0.2, min_iter=20):
    """
    Check for convergence based on RMSE history.
    """
    if len(rmse_history) > min_iter:
        recent_rmse = rmse_history[-min_iter:]
        if np.mean(recent_rmse) < threshold:
            return True
    return False
