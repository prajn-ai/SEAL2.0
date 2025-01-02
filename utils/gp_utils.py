import torch
import gpytorch

def initialize_gp_model():
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=2, nu=0.5))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    return ExactGPModel

def train_gp_model(x_train, y_train, training_iter=75):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = initialize_gp_model()(x_train, y_train, likelihood)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

    return model, likelihood

def evaluate_gp_model(model, likelihood, x_test):
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x_test))
        f_mean = observed_pred.mean
        f_var = observed_pred.variance
        f_covar = observed_pred.covariance_matrix

    return observed_pred, f_mean, f_var, f_covar
