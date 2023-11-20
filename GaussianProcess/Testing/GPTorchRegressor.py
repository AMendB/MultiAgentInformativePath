import torch
import torch.nn as nn
import gpytorch
import numpy as np
import sys
from scipy.spatial import distance_matrix
from scipy.optimize import fmin_bfgs

sys.path.append('.')


""" ExactGPModel is a class that inherits from gpytorch.models.ExactGP. It is used to create a GP model for regression."""
class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, initial_lengthscale=5.0, kernel_bounds=(0.5, 10)):
        super().__init__(train_x, train_y, likelihood)

        # Declare the mean and covariance modules
        self.mean_module = gpytorch.means.ZeroMean()
        # Declare the covariance module and set the initial lengthscale and constraints
        #First, check if the initial lengthscale is in the bounds
        assert kernel_bounds[0] <= initial_lengthscale <= kernel_bounds[1], "The initial lengthscale is not in the bounds"
        # Declare the kernel 
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.Interval(*kernel_bounds)))
        # Set the initial lengthscale
        self.covar_module.base_kernel.lengthscale = initial_lengthscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class SingleGPRegressor:

    def __init__(self, initial_lengthscale=5.0, kernel_bounds=(0.5, 10), device='cpu'):
        # Set the train_x and train_y
        self.train_x = None
        self.train_y = None
        self.device = device

        # Set the initial lengthscale and kernel bounds
        self.initial_lengthscale = initial_lengthscale
        self.kernel_bounds = kernel_bounds

        # Set the likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood, initial_lengthscale=self.initial_lengthscale, kernel_bounds=self.kernel_bounds).to(self.device)

        # Set the mll
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Set the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        # Set the training iterations
        self.training_iterations = 15

    def fit(self, train_x:np.ndarray, train_y:np.ndarray, verbose=False):

        # Set the train_x and train_y
        self.train_x = torch.FloatTensor(train_x).to(self.device)
        self.train_y = torch.FloatTensor(train_y).to(self.device)

        self.model.set_train_data(self.train_x, self.train_y, strict=False)

        # Train the model
        self.model.train()
        self.likelihood.train()

        # Iterate over training iterations
        converged = 0
        it = 0
        kernel_lengthscale = self.model.covar_module.base_kernel.lengthscale.item()
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

            kernel_lengthscale = self.model.covar_module.base_kernel.lengthscale.item()

            it += 1


    def predict(self, eval_x:np.ndarray):
        """ Evaluate the model """

        eval_x = torch.FloatTensor(eval_x).to(self.device)

        # Set into eval mode
        self.model.eval()
        self.likelihood.eval()

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(eval_x))
        
        return observed_pred.mean.detach().cpu().numpy(), observed_pred.stddev.detach().cpu().numpy()
    
    def reset(self):
        # Set the train_x and train_y
        self.train_x = None
        self.train_y = None
        self.model.set_train_data(self.train_x, self.train_y, strict=False)

        # Set the likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood, initial_lengthscale=self.initial_lengthscale, kernel_bounds=self.kernel_bounds).to(self.device)

        # Set the mll
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Set the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        # Set the training iterations
        self.training_iterations = 15

class MultipleGPRegressor:
    """ This class is used to create a GP model for regression with multiple outputs. """

    def __init__(self, number_of_regressors: int = 1, initial_lengthscale=5.0, kernel_bounds=(0.5, 10), device='cpu'):
        
        # Set the initial lengthscale and kernel bounds
        self.initial_lengthscale = initial_lengthscale
        self.kernel_bounds = kernel_bounds
        self.number_of_regressors = number_of_regressors

        # Set the device
        self.device = device

        # Set the likelihood and model
        self.likelihood = [gpytorch.likelihoods.GaussianLikelihood().to(self.device).to(self.device)  for _ in range(number_of_regressors)]
        self.model = [ExactGPModel(train_x=None, train_y=None, likelihood=self.likelihood[i]).to(self.device) for i in range(number_of_regressors)]

        # Set the mll
        self.mll = [gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) for likelihood, model in zip(self.likelihood, self.model)]

        # Set the optimizer
        self.optimizer = [torch.optim.Adam(model.parameters(), lr=0.5) for model in self.model]

        # Set the training iterations
        self.training_iterations = 5

    def fit(self, train_x:np.ndarray, train_y:np.ndarray,  model_index: int = 0, verbose=False):

        # Set the train_x and train_y
        self.train_x = torch.FloatTensor(train_x).to(self.device)
        self.train_y = torch.FloatTensor(train_y).to(self.device).flatten()

        self.model[model_index].set_train_data(self.train_x, self.train_y, strict=False)

        # Train the model
        self.model[model_index].train()
        self.likelihood[model_index].train()

        # Iterate over training iterations
        converged = 0
        it = 0
        loss_ant = None
        while it < self.training_iterations and not converged:
            # Zero backprop gradients
            self.optimizer[model_index].zero_grad()
            # Get output from model
            output = self.model[model_index](self.train_x)
            # Calc loss and backprop gradients
            loss = -self.mll[model_index](output, self.train_y).mean()
            loss.backward()
            if verbose:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    it + 1, self.training_iterations, loss.item(),
                    self.model[model_index].covar_module.base_kernel.lengthscale.item(),
                    self.model[model_index].likelihood.noise.item()
                ))

            self.optimizer[model_index].step()

            if loss_ant is None:
                loss_ant = loss.item()
            else:
                converged = np.abs(loss.item() - loss_ant) < 0.01

            it += 1

    def predict(self, eval_x:np.ndarray, model_index: int = 0):
        """ Evaluate the model """

        eval_x = torch.FloatTensor(eval_x).to(self.device) if isinstance(eval_x, np.ndarray) else eval_x

        # Set into eval mode
        self.model[model_index].eval()
        self.likelihood[model_index].eval()

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood[model_index](self.model[model_index](eval_x))
        
        return observed_pred.mean.detach().cpu().numpy(), observed_pred.stddev.detach().cpu().numpy()

class LocalGPRegressorCoordinator:
    """ This class train localy the GPs based on the data provided by the environment. """

    def __init__(self, scenario_map: np.ndarray, regresor_positions: np.ndarray, number_of_regressors: int, influence_radius: float, initial_lengthscale=5.0, kernel_bounds=(0.5, 10), device='cpu'):
        
        # Set the initial lengthscale and kernel bounds
        self.initial_lengthscale = initial_lengthscale
        self.kernel_bounds = kernel_bounds
        self.number_of_regressors = number_of_regressors
        self.regresor_positions = regresor_positions
        self.scenario_map = scenario_map
        self.device = device 
        self.influence_radius = influence_radius

        # Select the evaluation points where scenario_map == 1
        self.np_eval_X = np.array(np.where(self.scenario_map == 1)).T

        self.points_distance_matrix = np.exp(-distance_matrix(self.np_eval_X, self.regresor_positions))
        self.sum_distance_matrix = np.sum(self.points_distance_matrix, axis=1)
        self.weight = self.points_distance_matrix / self.sum_distance_matrix[:, None]

        self.eval_X = torch.FloatTensor(self.np_eval_X).to(device)

        # Create the data_x and data_y
        self.data_x = [None for _ in range(number_of_regressors)]
        self.data_y = [None for _ in range(number_of_regressors)]
        self.mu = np.zeros((self.number_of_regressors, len(self.np_eval_X)))
        self.sigma = np.ones((self.number_of_regressors, len(self.np_eval_X)))
        self.mu_map = np.zeros_like(self.scenario_map)
        self.sigma_map = np.ones_like(self.scenario_map)

        # Set the device
        self.device = device

        # Create the multiple GP regressor
        self.multiple_gp_regressor = MultipleGPRegressor(number_of_regressors=number_of_regressors, initial_lengthscale=initial_lengthscale, kernel_bounds=kernel_bounds, device=device)

        # Set the train_x and train_y
        self.train_x = None
        self.train_y = None

        # Set the training iterations
        self.training_iterations = 15

    def fit(self, new_train_x:np.ndarray, new_train_y:np.ndarray, verbose=False):
        
        """ First, find the corresponding regressors to the data provided. Then, train the corresponding regressors.
        """

        # Train the corresponding regressors
        for model_indx in range(self.number_of_regressors):

            # Get the indexes of the positions that are close to the position of the regressor
            indexes = self.get_close_index(position_origin=self.regresor_positions[model_indx], position_samples=new_train_x)

            if indexes is not None:
                # Save the data in the corresponding regressor
                self.data_x[model_indx] = new_train_x[indexes] if self.data_x[model_indx] is None else np.vstack((self.data_x[model_indx], new_train_x[indexes]))
                self.data_y[model_indx] = new_train_y[indexes] if self.data_y[model_indx] is None else np.vstack((self.data_y[model_indx], new_train_y[indexes]))

                # If there are positions close to the position of the regressor, train the regressor
                self.multiple_gp_regressor.fit(train_x=self.data_x[model_indx], train_y=self.data_y[model_indx], model_index=model_indx, verbose=verbose)
                # Predict the values of the evaluation points
                self.mu[model_indx,:], self.sigma[model_indx,:] = self.multiple_gp_regressor.predict(eval_x=self.eval_X, model_index=model_indx)

    def get_close_index(self, position_origin:np.ndarray, position_samples: np.ndarray):
        """ Returns the index of those positions that are close to the position provided. """

        # Get the distance between the position provided and the position samples
        distance = np.linalg.norm(position_samples - position_origin, axis=1)

        # Get the index of the those positions that are closer that self.influence_radius to the position provided
        index = np.where(distance < self.influence_radius)[0]

        return index if len(index) > 0 else None
    
    def predict(self):
        """ Evaluate the model by combining the predictions of the different regressors."""

        mu, sigma = self.generate_mean_map()

        self.mu_map[self.np_eval_X[:, 0], self.np_eval_X[:, 1]] = mu
        self.sigma_map[self.np_eval_X[:, 0], self.np_eval_X[:, 1]] = sigma

        return self.mu_map, self.sigma_map

    def generate_mean_map(self):
        """ Generate a map to show the points that are nearest to the local GP models """
        
        map_sigma = np.zeros((self.eval_X.shape[0],))
        map_mu = np.zeros((self.eval_X.shape[0],))
        
        # For every position in X, find the nearest GP model, take the mu in this position and assing to this mu value to nearest_map #
        for i in range(len(map_sigma)):

            # Compute the mu as a weighted sum of the mu of the local GP models #
            map_mu[i] = np.sum([self.mu[j,i] * self.weight[i,j] for j in range(self.number_of_regressors)])
            map_sigma[i] = np.sum([self.sigma[j,i] * self.weight[i,j] for j in range(self.number_of_regressors)])

        return map_mu, map_sigma
        

if __name__ == '__main__':

    # Create a MultipleGPRegressor #
    # ---------------------------- #
    # Create a search space 
    import matplotlib.pyplot as plt
    from Environment.GroundTruthsModels.AlgaeBloomGroundTruth import algae_bloom
    from Environment.GroundTruthsModels.ShekelGroundTruth import GroundTruth as shekel
    import numpy as np
    import time
    from copy import deepcopy

    navigation_map = np.ones((50, 50))
    # Create a GT #
    #gt = algae_bloom(navigation_map, dt=0.05)
    gt = shekel(navigation_map, seed=5456)
    gt2 = algae_bloom(navigation_map, seed=42136)
    gt.reset()
    gt2.reset()

    for _ in range(50):
        gt.step()

    gt_field = gt.read() + gt2.read()
    gt_field = gt_field / np.max(gt_field)
    # Obtain the valid position of the search space, i.e the positions that makes navigation_map == 1
    valid_positions = np.argwhere(navigation_map == 1)

    eval_x = valid_positions
    # gp_multiple_regressor = SingleGPRegressor(initial_lengthscale=5.0, kernel_bounds=(0.5, 10), device='cpu')
    #gp_multiple_regressor = MultipleGPRegressor(number_of_regressors=2, initial_lengthscale=5.0, kernel_bounds=(0.5, 10), device='cpu')
    # create a grid of points, begining in 5,5 and ending in 25,25

    
    gp_local_coordinate_regressor = LocalGPRegressorCoordinator(scenario_map=navigation_map,
                                                                influence_radius=1000,
                                                                number_of_regressors=1, 
                                                                regresor_positions = np.array([[0,0]]), 
                                                                initial_lengthscale=900, 
                                                                kernel_bounds=(0.1, 1000), 
                                                                device='cpu')

    # Create a 1x2 figure to represent the mu and the std #
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].set_title('mu')
    ax[1].set_title('std')
    ax[2].set_title('GT')
    mu_fig = ax[0].imshow(np.zeros(navigation_map.shape), vmin=0, vmax=1)
    std_fig = ax[1].imshow(np.zeros(navigation_map.shape), vmin=0, vmax=1)
    ax[2].imshow(gt_field, vmin=0, vmax=1)
    sampled_points_fig = ax[0].scatter([], [], c='r', s=10)
    sampled_points_fig2 = ax[2].scatter([], [], c='r', s=10)



    # Draw circles around the initial positions #

    # update the figure #
    fig.canvas.draw()
    plt.pause(0.1)
    T = []
    position = np.array([5, 5])
    S = 25

    train_x = []
    train_y = []

    def onclick(event):
        """ Key handler """

        key_pressed = event.key
        # Obtain the key pressed. Save only the AWSD #
        if key_pressed == '8':
            position[0] -= S
        elif key_pressed == '2':
            position[0] += S
        elif key_pressed == '4':
            position[1] -= S
        elif key_pressed == '6':
            position[1] += S
        elif key_pressed == '3':
            position[0] += S
            position[1] += S
        elif key_pressed == '7':
            position[0] -= S
            position[1] -= S
        elif key_pressed == '1':
            position[0] += S
            position[1] -= S
        elif key_pressed == '9':
            position[0] -= S
            position[1] += S
        else:
            return
        
        # Get the position of the click and transform to pixel position#
        new_position = position
        

        train_x.append(deepcopy(new_position))
        new_y = gt_field[int(new_position[0]), int(new_position[1])]
        train_y.append(new_y)
        
        t0 = time.time()
        gp_local_coordinate_regressor.fit(np.atleast_2d(new_position), np.atleast_2d(new_y), verbose=False)
        mu, std = gp_local_coordinate_regressor.predict()

        T.append(time.time() - t0)
        print('Time to update: ', T[-1])
        
        # Update the figure #
        mu_fig.set_data(mu)
        std_fig.set_data(std)
        #♠ Plot the sampled points with flipped columns #
        sampled_points_fig.set_offsets(np.asarray(train_x)[:, ::-1])
        sampled_points_fig2.set_offsets(np.asarray(train_x)[:, ::-1])
        
        
        fig.canvas.draw()
        
        
    cid = fig.canvas.mpl_connect('key_press_event', onclick)
    plt.show()


    plt.figure()
    plt.plot(np.arange(0,len(T)), T, 'r-', label='GP')
    plt.legend()
    plt.show()
    print("Total time: ", np.sum(T))