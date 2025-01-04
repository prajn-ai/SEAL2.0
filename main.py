import torch
import gpytorch
import random
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from utils.data_utils import SoundFieldData
from utils.gp_utils import train_gp_model, evaluate_gp_model, RKHS_norm
from utils.plotting_utils import plot_gp_results
from utils.sampling_utils import RandomWalk, PotentialNeighbors, AcquisitionFunction, CheckConvergence

def main():
    ##SET your data file here
    data_file = '/home/prajna/SEAL2.0/data/sim/2_source_sim.csv'
    sound_data = SoundFieldData(data_file)
    x_column, y_column, norm_spl_column, spl_column = sound_data.get_data()
    survey_grid = np.column_stack((x_column, y_column))

    ##Initialize Variables
    sampled_data = []
    sampled_region = []
    sampled_spl = []
    start_location = [48.0, 48.0] #Start location of the robot in the survey region
    n_samples = len(norm_spl_column)
    distance_horizon = 3  # Set your desired distance horizon
    prediction_horizon = 3  # Set your desired prediction horizon
    training_iter = 75  # Set your desired number of iterations
    initial_sample_size = 8 # Set your desired initial training set size
    convergence_criteria = False
    var_iter_local = []
    var_iter_global = []
    rmse_local_true = []
    rmse_global_true = []
    covar_trace = []
    covar_totelements = []
    covar_nonzeroelements = []
    noise = []
    lengthscale = []
    AIC = [] # Akaike Information Criterion
    BIC = [] # Bayesian Information Criterion
    sigma = 0.02
    f2_H_global = []
    f2_H_local = []
    image_path = '/home/prajna/SEAL2.0/plots/'

    #Get intial training data with a random walk
    initial_training_set, sampled_region, sampled_spl = RandomWalk(start_location, initial_sample_size, sampled_region, sampled_spl, sound_data, survey_grid)
    asv_location = sampled_region[-1]
    training_set = initial_training_set.copy()
    print('Training Set Initial: ', training_set)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    ## Main SEAL Algorithm
    while convergence_criteria is False:
        x_train, y_train = zip(*training_set)
        likelihood, model, optimizer, output, loss = train_gp_model(x_train, y_train, training_iter)
        noise.append(model.likelihood.noise.detach().numpy())
        lengthscale.append(model.covar_module.base_kernel.lengthscale.detach().numpy()[0])
        model.eval()
        likelihood.eval()

        potential_sampling_locations = PotentialNeighbors(asv_location, sampled_region, survey_grid)
        potential_spls = []
        for i in potential_sampling_locations:
            potential_spl = sound_data.get_spl(i[0], i[1])
            potential_spls.append(potential_spl)
        
        local_x_true = potential_sampling_locations
        potential_sampling_locations = np.array(potential_sampling_locations)
        observed_pred_local, lower_local, upper_local, f_var_local, f_covar_local, f_mean_local = evaluate_gp_model(local_x_true, model, likelihood)
        mse_local_true = mean_squared_error(potential_spls, observed_pred_local.mean.numpy())
        rmse_local_true.append(math.sqrt(mse_local_true))
        observed_pred_global, lower_global, upper_global, f_var_global, f_covar_global, f_mean_global = evaluate_gp_model(survey_grid, model, likelihood)
        mse_global_true = mean_squared_error(norm_spl_column, observed_pred_global.mean.numpy())
        rmse_global_true.append(math.sqrt(mse_global_true))

        #Determining Next sampling location
        if var_iter_global:
            next_sampling_location, sampling_mu_var_g = AcquisitionFunction(f_var_local, f_mean_local, potential_sampling_locations)
        else:
            local_var = f_var_local.numpy()
            highest_var = np.nanmax(local_var)
            index = np.where(local_var == highest_var)[0]
            next_sampling_location = potential_sampling_locations[int(index)]

        var_iter_local.append(max(f_var_local.numpy()))
        var_iter_global.append(max(f_var_global.numpy()))
        covar_trace.append(np.trace(f_covar_global.detach().numpy()))
        covar_totelements.append(np.size(f_covar_global.detach().numpy()))
        covar_nonzeroelements.append(np.count_nonzero(f_covar_global.detach().numpy()))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        AIC_sample = 2*np.log(covar_nonzeroelements[-1]) - 2*np.log(mse_global_true)
        AIC.append(AIC_sample)
        # BIC calculated from https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
        BIC_sample = np.size(sampled_region)*np.log(covar_nonzeroelements[-1]) - 2*np.log(mse_global_true)
        BIC.append(BIC_sample)
        K_global = output._covar.detach().numpy()
        y_global = np.array(y_train)
        y_global = y_global.reshape(len(y_train),1)
        f2_H_sample = RKHS_norm(y_global,sigma,K_global)
        f2_H_global.append(f2_H_sample[0,0])
        n_set = len(initial_training_set)
        n_sub = math.floor(n_set/2)
        i_sub = random.sample(range(1,n_set),n_sub)
        i_sub.sort()
        K_local = K_global[np.ix_(i_sub,i_sub)]
        y_local = y_global[i_sub]
        f2_H_sample = RKHS_norm(y_local,sigma,K_local)
        f2_H_local.append(f2_H_sample[0,0])

        fig = plt.figure(figsize=(24, 16))
        # plot real surface and the observed measurements
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10 = plot_gp_results(fig, sampled_region, sampled_spl, norm_spl_column, potential_sampling_locations, 
                    survey_grid, lower_local, upper_local, var_iter_local, var_iter_global, 
                    rmse_local_true, rmse_global_true, lengthscale, noise, covar_trace, 
                    covar_totelements, covar_nonzeroelements, AIC, BIC, f2_H_local, f2_H_global, next_sampling_location)
        fig.tight_layout()
        fig.savefig(image_path+str(len(training_set))+'.png')
        plt.close(fig)

        convergence_criteria = CheckConvergence(rmse_global_true)
        asv_location = next_sampling_location
        print('Next Sampling Location: ', next_sampling_location)
        print(convergence_criteria)
        sampled_region.append(asv_location.tolist())
        new_spl = sound_data.get_spl(asv_location[0], asv_location[1])
        sampled_spl.append(new_spl)
        sampled_data.append((asv_location[0], asv_location[1], new_spl))
        new_data_point = (asv_location, new_spl)
        training_set.append(new_data_point)
    



if __name__ == '__main__':
    main()
