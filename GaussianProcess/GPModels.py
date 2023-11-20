from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

import torch
import gpytorch

from botorch.models.gp_regression import FixedNoiseGP 
from botorch.fit import fit_gpytorch_mll

import numpy as np

class GaussianProcessScikit:

	def __init__(self, scenario_map, initial_lengthscale=5.0, kernel_bounds=(0.1, 100)):
		""" Gaussian Process with Scikit-Learn to predict the map """

		# Assign input variables #
		self.scenario_map = scenario_map

		# Set the initial lengthscale and kernel bounds
		self.initial_lengthscale = initial_lengthscale
		self.kernel_bounds = kernel_bounds

		# Calculate visitable cells #
		self.visitable_indexes = np.where(self.scenario_map == 1) # visitable cells indexes
		self.X_cells = np.vstack(self.visitable_indexes).T # visitable cells coords

		self.reset()

	def fit_gp(self, X_new, y_new, variances_new):
		""" Fit Gaussian Process to data """

		# Add new measurements or update existing if new variance better than older #
		self.all_train_measures_dict.update({tuple(X): (y, variance) for X, y, variance in zip(X_new, y_new, variances_new) if tuple(X) not in self.all_train_measures_dict or variance < self.all_train_measures_dict[tuple(X)][1]})

		# Get data from dictionary:  X_meas: stored in keys, y_meas: stored in first column of values tuple, variance_meas: stored in second column of values tuple #
		X_meas, y_meas, variance_meas = zip(*[(key, value[0], value[1]) for key, value in self.all_train_measures_dict.items()])

		# Update internal gp alpha variable (variance) #
		self.gp.alpha = np.array(variance_meas)

		# FIT #
		self.gp.fit( np.array(X_meas), np.array(y_meas)) 

	def predict_gt(self):
		""" Ground truth prediction for visitable cells (X_cells) """
		
		model_out, uncertainty_out = self.gp.predict(self.X_cells, return_std=True)

		# Assign GP predictions to corresponding map cells, "flatten" to match dimensions
		self.model_map[self.visitable_indexes] = model_out.flatten()
		self.uncertainty_map[self.visitable_indexes] = uncertainty_out.flatten()

		return self.model_map, self.uncertainty_map

	def reset(self):

		# Initialize dataset dictionary: keys = tuple(position of measure), values = tuple(measure, variance) #
		self.all_train_measures_dict = {}

		# Define empty maps to predict the model #
		self.model_map = np.zeros_like(self.scenario_map)
		self.uncertainty_map = np.zeros_like(self.scenario_map)

		# Define Gaussian Process with Kernel #
		# self.kernel = ConstantKernel(1) #* Matern(length_scale=5, length_scale_bounds=(1, 1000)) + WhiteKernel(0.005, noise_level_bounds=(0.00001, 0.1))
		self.kernel = RBF(length_scale=self.initial_lengthscale, length_scale_bounds=self.kernel_bounds) # RBF kernel
		self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.001, n_restarts_optimizer=5)


class GaussianProcessGPyTorch:
	""" Gaussian Process with GPyTorch to predict the map """

	def __init__(self, scenario_map, initial_lengthscale=5.0, kernel_bounds=(0.1, 100), training_iterations=50, scale_kernel:bool=True, device='cpu'):

		# Set input map #
		self.scenario_map = scenario_map

		# Set the initial lengthscale and kernel bounds
		self.initial_lengthscale = initial_lengthscale
		self.kernel_bounds = kernel_bounds
		
		# Set the training iterations
		self.training_iterations = training_iterations

		# Set other config #
		self.scale_kernel = scale_kernel
		self.device = device

		# Calculate visitable cells #
		self.visitable_indexes = np.where(self.scenario_map == 1) # visitable cells indexes
		self.X_cells = np.vstack(self.visitable_indexes).T # visitable cells coords

		# Call reset function to create regression model #
		self.reset()

	def fit_gp(self, X_new, y_new, variances_new, verbose=False):#, train_x:np.ndarray, train_y:np.ndarray, verbose=False):

		# Add new measurements or update existing if new variance better than older #
		self.all_train_measures_dict.update({tuple(X): (y, variance) for X, y, variance in zip(X_new, y_new, variances_new) if tuple(X) not in self.all_train_measures_dict or variance < self.all_train_measures_dict[tuple(X)][1]})

		# Get data from dictionary:  X_meas: stored in keys, y_meas: stored in first column of values tuple, variance_meas: stored in second column of values tuple #
		X_meas, y_meas, variance_meas = zip(*[(key, value[0], value[1]) for key, value in self.all_train_measures_dict.items()])

		# Convert train data and noise to TENSOR #
		self.train_x = torch.FloatTensor(X_meas).to(self.device)
		self.train_y = torch.FloatTensor(y_meas).to(self.device)
		self.train_noise = torch.FloatTensor(variance_meas).to(self.device)

		# Update with new train data and noise (samples variance) # 
		self.model.set_train_data(self.train_x, self.train_y, strict=False)
		self.likelihood.noise = self.train_noise

		# Find optimal model hyperparameters (training) #
		self.model.train()
		self.likelihood.train()

		# Iterate over training iterations
		converged = 0
		it = 0
		#kernel_lengthscale = self.model.covar_module.base_kernel.lengthscale.item()
		loss_ant = None
		while it < self.training_iterations and not converged:
			# Zero backprop gradients
			self.optimizer.zero_grad()
			# Get output from model
			output = self.model(self.train_x)
			# Calc loss and backprop gradients
			loss = -self.mll(output, self.train_y)
			loss.backward()
			if verbose:
				print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
					it + 1, self.training_iterations, loss.item(),
					self.model.covar_module.base_kernel.lengthscale.item(),
					self.model.likelihood.noise.item()
				))

			self.optimizer.step()

			if loss_ant is None:
				loss_ant = loss.item()
			else:
				converged = np.abs(loss.item() - loss_ant) < 0.01

			#kernel_lengthscale = self.model.covar_module.base_kernel.lengthscale.item()

			it += 1

	def predict_gt(self): #, eval_x:np.ndarray):
		""" Evaluate the model """

		eval_x = torch.FloatTensor(self.X_cells).to(self.device)

		# Set into eval mode
		self.model.eval()
		self.likelihood.eval()

		# Make predictions
		with torch.no_grad(), gpytorch.settings.fast_pred_var():
			# observed_pred = self.likelihood(self.model(eval_x))
			observed_pred = self.model(eval_x)

		# Extract predictions #
		model_out = observed_pred.mean.detach().cpu().numpy()
		uncertainty_out = observed_pred.stddev.detach().cpu().numpy()

		# Assign GP predictions to corresponding map cells, "flatten" to match dimensions
		self.model_map[self.visitable_indexes] = model_out.flatten()
		self.uncertainty_map[self.visitable_indexes] = uncertainty_out.flatten()
		
		return self.model_map, self.uncertainty_map

	def reset(self):

		# Initialize dataset dictionary: keys = tuple(position of measure), values = tuple(measure, variance) #
		self.all_train_measures_dict = {}

		# Define empty maps to predict the model #
		self.model_map = np.zeros_like(self.scenario_map)
		self.uncertainty_map = np.zeros_like(self.scenario_map)

		# Initialize train and noise tensors #
		self.train_x = torch.FloatTensor([]).to(self.device)
		self.train_y = torch.FloatTensor([]).to(self.device)
		self.train_noise = torch.FloatTensor([]).to(self.device)

		# Set likelihood and model #
		# self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
		self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=self.train_noise).to(self.device) # FixedNoiseGaussianLikelihood to add measurements noise
		self.model = ExactGPModel(train_x=self.train_x, train_y=self.train_y, likelihood=self.likelihood, initial_lengthscale=self.initial_lengthscale, kernel_bounds=self.kernel_bounds, scale_kernel=self.scale_kernel).to(self.device)

		# Set "loss" for GPs - the marginal log likelihood (mll)
		self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

		# Set up to use the adam optimizer
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

""" ExactGPModel is a class that inherits from gpytorch.models.ExactGP. It is used to create a GP model for regression."""
class ExactGPModel(gpytorch.models.ExactGP):

	def __init__(self, train_x, train_y, likelihood, initial_lengthscale=5.0, kernel_bounds=(0.5, 10), scale_kernel:bool=True):
		super().__init__(train_x, train_y, likelihood)

		# Declare the mean and covariance modules #
		#self.mean_module = gpytorch.means.ConstantMean()
		self.mean_module = gpytorch.means.ZeroMean()
		
		# Declare the covariance module and set the initial lengthscale and constraints #
		#First, check if the initial lengthscale is in the bounds
		assert kernel_bounds[0] <= initial_lengthscale <= kernel_bounds[1], "The initial lengthscale is not in the bounds"
		
		if scale_kernel == True:
			# Declare the kernel #
			self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.Interval(*kernel_bounds)))
			# Set the initial lengthscale #
			self.covar_module.base_kernel.lengthscale = initial_lengthscale
		else:
			# Declare the kernel #
			self.covar_module = gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.Interval(*kernel_bounds))
			# Set the initial lengthscale #
			self.covar_module.lengthscale = initial_lengthscale

	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcessBoTorch:
	""" Gaussian Process with BoTorch to predict the map """

	def __init__(self, scenario_map, initial_lengthscale=5.0, kernel_bounds=(0.1, 100), device='cpu'):

		# Set input map #
		self.scenario_map = scenario_map

		# Set the initial lengthscale and kernel bounds
		self.initial_lengthscale = initial_lengthscale
		self.kernel_bounds = kernel_bounds

		# Set calculate device #
		self.device = device

		# Calculate visitable cells #
		self.visitable_indexes = np.where(self.scenario_map == 1) # visitable cells indexes
		self.X_cells = np.vstack(self.visitable_indexes).T # visitable cells coords

		# Call reset function to create regression model #
		self.reset()

	def fit_gp(self, X_new, y_new, variances_new, verbose=False):#, train_x:np.ndarray, train_y:np.ndarray, verbose=False):

		# Add new measurements or update existing if new variance better than older #
		self.all_train_measures_dict.update({tuple(X): (y, variance) for X, y, variance in zip(X_new, y_new, variances_new) if tuple(X) not in self.all_train_measures_dict or variance < self.all_train_measures_dict[tuple(X)][1]})

		# Get data from dictionary:  X_meas: stored in keys, y_meas: stored in first column of values tuple, variance_meas: stored in second column of values tuple #
		X_meas, y_meas, variance_meas = zip(*[(key, value[0], value[1]) for key, value in self.all_train_measures_dict.items()])

		# Convert train data and noise to TENSOR #
		self.train_x = torch.FloatTensor(X_meas).to(self.device)
		self.train_y = torch.FloatTensor(y_meas).to(self.device)
		self.train_noise = torch.FloatTensor(variance_meas).to(self.device)

		# Update with new train data and noise (samples variance) # 
		self.model.set_train_data(self.train_x, self.train_y, strict=False)
		self.model.likelihood.noise = self.train_noise
		self.model.likelihood._aug_batch_shape = len(self.train_noise)

		fit_gpytorch_mll(self.mll)

	def predict_gt(self): #, eval_x:np.ndarray):
		""" Evaluate the model """

		eval_x = torch.FloatTensor(self.X_cells).to(self.device)

		# Make predictions
		with torch.no_grad():
			posterior = self.model.posterior(eval_x)
			model_out = posterior.mean.cpu()
			uncertainty_out = posterior.stddev.cpu()
    
		# Assign GP predictions to corresponding map cells, "flatten" to match dimensions
		self.model_map[self.visitable_indexes] = model_out.flatten()
		self.uncertainty_map[self.visitable_indexes] = uncertainty_out.flatten()
		
		return self.model_map, self.uncertainty_map

	def reset(self):

		# Initialize dataset dictionary: keys = tuple(position of measure), values = tuple(measure, variance) #
		self.all_train_measures_dict = {}

		# Define empty maps to predict the model #
		self.model_map = np.zeros_like(self.scenario_map)
		self.uncertainty_map = np.zeros_like(self.scenario_map)

		# Initialize train and noise tensors #
		self.train_x = torch.empty((0, 2), device=self.device)
		self.train_y = torch.empty((0,1), device=self.device)
		self.train_noise = torch.empty((0,1), device=self.device)

		# Set likelihood and model #
		self.model = FixedNoiseGP(train_X=self.train_x, train_Y=self.train_y, train_Yvar=self.train_noise).to(self.device)

		# Set "loss" for GPs - the marginal log likelihood (mll)
		self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

		# Set up to use the adam optimizer
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
