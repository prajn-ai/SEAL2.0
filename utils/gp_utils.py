import torch
import gpytorch
from numpy.linalg import inv
import numpy as np

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=2, nu=0.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp_model(x_train, y_train, training_iter=75):
    x_train = [torch.tensor(arr, dtype=torch.float32) for arr in x_train]
    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x_train, y_train, likelihood)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

    return likelihood, model, optimizer, output, loss

def evaluate_gp_model(x_test, model, likelihood):
    x_test = [torch.tensor(arr, dtype=torch.float32) for arr in x_test]
    x_test = torch.stack(x_test)
    model.eval()
    likelihood.eval()
    
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x_test))
        f_preds = model(x_test)
        f_mean = f_preds.mean
        f_var = f_preds.variance
        #f_var = np.diag(f_preds.lazy_covariance_matrix.numpy())
        f_covar = f_preds.covariance_matrix
    
    with torch.no_grad():
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()

    # print('mean', f_mean)
    # print('var', f_var)
        
    return observed_pred, lower, upper, f_var, f_covar, f_mean

def RKHS_norm(y,sigma,K):
    n_row, n_col = K.shape
    alpha = inv(K + sigma**2 * np.eye(n_row)) @ y
    return alpha.transpose() @ K @ alpha