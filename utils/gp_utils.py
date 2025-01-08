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
    print('data prepped')
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
        kernel_print(i, training_iter, loss, model)
        optimizer.step()

    return likelihood, model, optimizer, output, loss

def kernel_print(i, training_iter, loss, model):
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.detach().numpy()[0][0],
            model.covar_module.base_kernel.lengthscale.detach().numpy()[0][1],
            model.likelihood.noise.detach().numpy()
        ))

# @profile
def evaluate_gp_model(x_test, model, likelihood, batch_size=50):
    if not isinstance(x_test, torch.Tensor):
        x_test = torch.tensor(x_test, dtype=torch.float32)
    elif x_test.dtype != torch.float32:
        x_test = x_test.to(torch.float32)

    # Initialize lists to hold batch results
    observed_pred_list = []
    lower_list = []
    upper_list = []
    f_var_list = []
    f_covar_list = []
    f_mean_list = []

    # Set model and likelihood to evaluation mode
    model.eval()
    likelihood.eval()

    # Process in batches
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, len(x_test), batch_size):
            batch = x_test[i:i + batch_size]

            # Make predictions for the batch
            observed_pred = likelihood(model(batch))
            f_preds = model(batch)

            # Extract components
            f_mean = f_preds.mean
            f_var = f_preds.variance
            f_covar = f_preds.lazy_covariance_matrix
            lower, upper = observed_pred.confidence_region()

            # Append results
            observed_pred_list.append(f_mean)
            lower_list.append(lower)
            upper_list.append(upper)
            f_var_list.append(f_var)
            f_covar_list.append(f_covar)  # Keep as list for memory efficiency
            f_mean_list.append(f_mean)

    # Concatenate tensor results
    observed_pred = torch.cat(observed_pred_list, dim=0)
    lower = torch.cat(lower_list, dim=0)
    upper = torch.cat(upper_list, dim=0)
    f_var = torch.cat(f_var_list, dim=0)
    f_mean = torch.cat(f_mean_list, dim=0)
    # Convert f_covar to a single tensor (if memory allows)
    # f_covar = torch.cat([covar.evaluate() for covar in f_covar_list], dim=0)


    return observed_pred, lower, upper, f_var, f_covar_list, f_mean

def RKHS_norm(y,sigma,K):
    n_row, n_col = K.shape
    alpha = inv(K + sigma**2 * np.eye(n_row)) @ y
    return alpha.transpose() @ K @ alpha