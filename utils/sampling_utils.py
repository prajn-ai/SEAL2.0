import torch
import numpy as np
import random
import itertools

def RandomWalk(start_location, initial_sample_size, coordinates_passed, spl_sampled, sound_data):
    initial_training_set = []
    asv_location = np.array(start_location)  # Start at the given start location

    for _ in range(initial_sample_size):
        # Get SPL and angles for the current location
        y = sound_data.get_spl(asv_location[0], asv_location[1])
        x = np.concatenate([asv_location])
        
        # Store the sampled data
        spl_sampled.append(y)
        initial_training_set.append((x, y))
        coordinates_passed.append(asv_location)

        # Determine the next location
        filtered_neighbors = PotentialNeighbors(asv_location, coordinates_passed)
        new_location = np.array(random.choice(filtered_neighbors))
        asv_location = new_location

    return initial_training_set, coordinates_passed, spl_sampled



def PotentialNeighbors(current_coordinate, coordinates_passed, distance_horizon=3):
    """
    Identify potential neighboring coordinates.
    """
    coordinates_range = range(-distance_horizon, distance_horizon)
    potential_neighbors = np.array(list(itertools.product(coordinates_range, repeat=2)))
    current_coordinate = current_coordinate.numpy()

    all_potential_neighbors = current_coordinate + potential_neighbors
    tup_all_potential_neighbors = set(map(tuple, all_potential_neighbors))
    tup_coordinates_passed = set(map(tuple, coordinates_passed))

    filtered_neighbors = tup_all_potential_neighbors.difference(tup_coordinates_passed)
    return torch.tensor(list(filtered_neighbors), dtype=torch.float32)

def AcquisitionFunction(var, mean, potential_neighbors):
    alpha = 0
    beta = 1
    sampling_mu_var = beta * mean + alpha * var
    max_index = np.argmax(sampling_mu_var)
    next_sample_location = potential_neighbors[max_index]
    return next_sample_location, sampling_mu_var

def CheckConvergence(rmse_history, threshold=0.2, min_iter=20):

    if len(rmse_history) > min_iter:
        recent_rmse = rmse_history[-min_iter:]
        if np.mean(recent_rmse) < threshold:
            return True
    return False
