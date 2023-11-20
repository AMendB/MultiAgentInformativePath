import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator
import pandas as pd
from GaussianProcess.GPModels import GaussianProcessScikit, GaussianProcessGPyTorch, GaussianProcessBoTorch
import torch

class LoadingExecutedAlgorithms:

    def __init__(self, scenario_map, selected_algorithm, n_agents, reward_function, influence_length, mean_sensormeasure, std_sensormeasure, variance_sensormeasure, regression_library, activate_plot_render):

        self.scenario_map = scenario_map
        self.non_water_mask = self.scenario_map != 1 # mask with True where no water
        self.selected_algorithm = selected_algorithm
        self.n_agents = n_agents
        self.reward_function = reward_function
        self.influence_length = influence_length
        self.mean_sensormeasure = mean_sensormeasure
        self.std_sensormeasure = std_sensormeasure
        self.variance_sensormeasure  = variance_sensormeasure
        self.activate_plot_render = activate_plot_render

        self.relative_path = 'Experiments/Results/' + ''.join(filter(str.isupper, self.selected_algorithm)) + '.' + str(self.n_agents) + '.' + self.reward_function + '/'

        GTs_path = self.relative_path + 'GroundTruths.npy'
        waypoints_path = self.relative_path + 'waypoints.csv'
        metrics_path = self.relative_path + 'metrics.csv'

        self.GTs = np.load(GTs_path) 
        self.waypoints_df = MetricsDataCreator.load_csv_as_df(waypoints_path) 
        self.metrics_df = MetricsDataCreator.load_csv_as_df(metrics_path) 

        # algorithm_info = self.metrics_df['Algorithm'][0].split('.')
        # self.selected_algorithm = algorithm_info[0]
        # self.n_agents = int(algorithm_info[1])
        # self.reward_function = algorithm_info[2]

        self.runs = np.unique(self.metrics_df['Run'])

        for run in self.runs:
            # Divide dfs in runs #
            individual_run_waypoints = self.waypoints_df[self.waypoints_df['Run'] == run] 
            individual_run_metrics = self.metrics_df[self.metrics_df['Run'] == run] 

            # Group by "Step" #
            groups_by_step = individual_run_waypoints.groupby('Step')

            if self.activate_plot_render is True:
                # Create Gaussian Process #
                if regression_library == 'scikit':
                    self.gaussian_process = GaussianProcessScikit(scenario_map = self.scenario_map, initial_lengthscale = 5.0, kernel_bounds = (0.1, 100))
                elif regression_library == 'gpytorch':
                    self.gaussian_process = GaussianProcessGPyTorch(scenario_map = self.scenario_map, initial_lengthscale = 5.0, kernel_bounds = (0.1, 100), training_iterations = 50, device = 'cuda' if torch.cuda.is_available() else 'cpu')
                elif regression_library == 'botorch':
                    self.gaussian_process = GaussianProcessBoTorch(scenario_map = self.scenario_map, initial_lengthscale = 5.0, kernel_bounds = (0.1, 100), device = 'cuda' if torch.cuda.is_available() else 'cpu')
                else:
                    raise NotImplementedError("This library is not implemented. Choose one that is.")
                
                self.model_mean_map = np.zeros_like(self.scenario_map) 
                self.model_uncertainty_map = np.zeros_like(self.scenario_map) 
                self.gaussian_process.reset()
                self.ground_truth = self.GTs[run]
                
                self.render_fig = None
                
                for step, group in groups_by_step:

                    group.reset_index(drop=True, inplace=True)
                    self.done = group['Done'].to_dict()
                    self.active_agents = {key: not value for key, value in self.done.items()}

                    self.fleet_positions = group[['x', 'y']].to_numpy()

                    # Sensor samples #
                    position_new_measures, new_measures, variance_measures = self.take_samples()

                    # Fit gaussian process with new samples #
                    self.gaussian_process.fit_gp(X_new=position_new_measures, y_new=new_measures, variances_new=variance_measures)

                    # Update model: prediction of ground truth #
                    self.model_mean_map, self.model_uncertainty_map = self.gaussian_process.predict_gt()

                    self.render(step)
                plt.show()
            
            self.first_accreward_agent_index = individual_run_metrics.columns.get_loc('AccRw0')
            self.R_agents_acc = individual_run_metrics.iloc[:, self.first_accreward_agent_index:self.first_accreward_agent_index + self.n_agents].values.tolist()
            self.R_acc = individual_run_metrics['R_acc'].values.tolist()
            self.rmse = individual_run_metrics['RMSE'].values.tolist()
            self.rmse_peaks = individual_run_metrics['RMSE_peaks'].values.tolist()
            self.R2_error = individual_run_metrics['R2_error'].values.tolist()
            self.Uncert_mean = individual_run_metrics['Uncert_mean'].values.tolist()
            self.Uncert_max = individual_run_metrics['Uncert_max'].values.tolist()
            self.Traveled_distance = individual_run_metrics['Traveled_distance'].values.tolist()
            self.Max_Redundancy = individual_run_metrics['Max_Redundancy'].values.tolist()

            self.plot_metrics_figs(run)
            plt.close('all')

    def compute_influence_mask(self, actual_agent_position): 
        """ Compute influence area (circular mask over the scenario map) around actual position """

        influence_mask = np.zeros_like(self.scenario_map) 

        pose_x, pose_y = actual_agent_position.astype(int) 

        # State - coverage area #
        range_x_axis = np.arange(0, self.scenario_map.shape[0]) # posible positions in x-axis
        range_y_axis = np.arange(0, self.scenario_map.shape[1]) # posible positions in y-axis

        # Compute the circular mask (area) #
        mask = (range_x_axis[np.newaxis, :] - pose_x) ** 2 + (range_y_axis[:, np.newaxis] - pose_y) ** 2 <= self.influence_length ** 2 

        influence_mask[mask.T] = 1.0 # converts True values to 1 and False values to 0

        return influence_mask

    def plot_metrics_figs(self, run):
            # Reward and Error final graphs #
            fig, ax = plt.subplots(2, 3, figsize=(17,9))

            ax[0][0].plot(self.R_agents_acc, '-o')
            ax[0][0].set(title = 'Reward', xlabel = 'Step', ylabel = 'Individual Reward')
            ax[0][0].legend([f'Agent {i}' for i in range(self.n_agents)])
            ax[0][0].plot(self.R_acc, 'b-', linewidth=4)
            ax[0][0].grid()

            ax[0][1].plot(self.rmse, '-o')
            ax[0][1].plot(self.rmse_peaks, '-o')
            ax[0][1].set(title = 'RMSE', xlabel = 'Step')
            ax[0][1].legend(['Total', 'In Peaks'])
            ax[0][1].grid()

            ax[0][2].plot(self.R2_error, '-o')
            ax[0][2].set(title = 'R2_error', xlabel = 'Step')
            ax[0][2].grid()

            ax[1][0].plot(self.Uncert_mean, '-o')
            ax[1][0].plot(self.Uncert_max, '-o')
            ax[1][0].set(title = 'Uncertainty', xlabel = 'Step')
            ax[1][0].legend(['Mean', 'Max'])
            ax[1][0].grid()

            ax[1][1].plot(self.Traveled_distance, '-o')
            ax[1][1].set(title = 'Traveled_distance', xlabel = 'Step')
            ax[1][1].grid()

            ax[1][2].plot(self.Max_Redundancy, '-o')
            ax[1][2].set(title = 'Max_Redundancy', xlabel = 'Step')
            ax[1][2].grid()

            fig.suptitle('Run ' + str(run))

            plt.show()

    def render(self, step):
        """ Update the photograph/state for every vehicle. Every channel will be an input of the Neural Network. """

        if not any(self.active_agents.values()):
            return

        first_available_agent = np.argmax(list(self.active_agents.values())) # first True in active_agents

        # Create fleet position map #
        fleet_position_map = np.zeros_like(self.scenario_map)
        fleet_position_map[self.fleet_positions[:,0], self.fleet_positions[:,1]] = 1.0 # set 1 where there is an agent

        influence_masks = [self.compute_influence_mask(self.fleet_positions[i]) for i in range(self.n_agents)]
        redundancy_mask = np.sum([influence_masks[idx] for idx in range(self.n_agents) if self.active_agents[idx]], axis = 0)
        
        # Channels 3 and 4
        agent_observation_of_position = np.zeros_like(self.scenario_map)
        agent_observation_of_position[self.fleet_positions[first_available_agent,0], self.fleet_positions[first_available_agent,1]] = 1.0 # map only with the position of the observing agent
        
        agent_observation_of_fleet = fleet_position_map.copy()
        agents_to_remove_positions = np.array([pos for idx, pos in enumerate(self.fleet_positions) if (idx == first_available_agent) or (not self.active_agents[idx])])  # if observing agent, or not active
        agent_observation_of_fleet[agents_to_remove_positions[:,0], agents_to_remove_positions[:,1]] = 0.0 # agents map without the observing agent

        """Each key from photographs dictionary is an agent, all photographs associated to that agent are concatenated in its value:"""
        state_first_active_agent = np.concatenate(( 
            self.scenario_map[np.newaxis], # Channel 0 -> Known boundaries/map
            self.ground_truth[np.newaxis],
            self.model_mean_map[np.newaxis], # Channel 1 -> Model mean map
            self.model_uncertainty_map[np.newaxis], # Channel 2 -> Model uncertainty map
            agent_observation_of_position[np.newaxis], # Channel 3 -> Observing agent position map
            agent_observation_of_fleet[np.newaxis], # Channel 4 -> Others active agents position map
            redundancy_mask[np.newaxis]
        ))

        """ Print visual representation/photographs of each state of the scenario. """
        
        if self.render_fig is None: # create first frame of fig, if not already created
            
            self.render_fig, self.render_axs = plt.subplots(1, 7, figsize=(17,5))
            
            # AXIS 0: Print the obstacles map #
            self.im0 = self.render_axs[0].imshow(state_first_active_agent[0], cmap = 'jet')
            self.render_axs[0].set_title('Navigation map')

            # AXIS 1: Print the Ground Truth #
            state_first_active_agent[1][self.non_water_mask] = np.nan
            self.im1 = self.render_axs[1].imshow(state_first_active_agent[1],  cmap ='jet', vmin = 0.0, vmax = 1.0)
            self.render_axs[1].set_title("Real Importance (GT)")

            # AXIS 2: Print model mean #
            state_first_active_agent[2][self.non_water_mask] = np.nan
            self.im2 = self.render_axs[2].imshow(state_first_active_agent[2],  cmap ='jet', vmin = 0.0, vmax = 1.0)
            self.render_axs[2].set_title("Model Mean/Importance")

            # AXIS 3: Print model uncertainty #
            state_first_active_agent[3][self.non_water_mask] = np.nan
            self.im3 = self.render_axs[3].imshow(state_first_active_agent[3],  cmap ='gray', vmin = 0.0, vmax = 1.0)#, norm='log')
            self.render_axs[3].set_title("Model Uncertainty")

            # AXIS 4: Agent 0 position #
            state_first_active_agent[4][self.non_water_mask] = 0.75
            self.im4 = self.render_axs[4].imshow(state_first_active_agent[4], cmap = 'gray', vmin = 0.0, vmax = 1.0)
            self.render_axs[4].set_title("Agent 0 position")

            # AXIS 5: Others-than-Agent 0 position #
            state_first_active_agent[5][self.non_water_mask] = 0.75
            self.im5 = self.render_axs[5].imshow(state_first_active_agent[5], cmap = 'gray', vmin = 0.0, vmax = 1.0)
            self.render_axs[5].set_title("Others agents position")

            # AXIS 6: Redundancy mask #
            state_first_active_agent[6][self.non_water_mask] = np.nan
            self.im6 = self.render_axs[6].imshow(state_first_active_agent[6], cmap = 'jet', vmin = 0.0, vmax = 4.0)
            self.render_axs[6].set_title("Redundancy mask")

            self.render_fig.suptitle('Step ' + str(step) + ' RwFn: ' + self.reward_function)

        else:
            # UPDATE FIG INFO/DATA IN EVERY RENDER CALL #
            # AXIS 0: Print the obstacles map #
            self.im0.set_data(state_first_active_agent[0])
            # AXIS 1: Print the Ground Truth #
            state_first_active_agent[1][self.non_water_mask] = np.nan
            self.im1.set_data(state_first_active_agent[1])
            # AXIS 2: Print model mean #
            state_first_active_agent[2][self.non_water_mask] = np.nan
            self.im2.set_data(state_first_active_agent[2])
            # AXIS 3: Print model uncertainty #
            state_first_active_agent[3][self.non_water_mask] = np.nan
            self.im3.set_data(state_first_active_agent[3])
            # AXIS 4: Agent 0 position #
            state_first_active_agent[4][self.non_water_mask] = 0.75
            self.im4.set_data(state_first_active_agent[4])
            # AXIS 5: Others-than-Agent 0 position #
            state_first_active_agent[5][self.non_water_mask] = 0.75
            self.im5.set_data(state_first_active_agent[5])
            # AXIS 6: Redundancy mask #
            state_first_active_agent[6][self.non_water_mask] = np.nan
            self.im6.set_data(state_first_active_agent[6])

            self.render_fig.suptitle('Step ' + str(step) + ' RwFn: ' + self.reward_function)

        plt.draw()	
        plt.pause(0.005)

    def take_samples(self):
        """ The active agents take a noisy sample from the ground truth """
        
        # Save positions where samples are taken #
        position_measures = [[self.fleet_positions[idx, 0], self.fleet_positions[idx, 1]] for idx in range(self.n_agents) if self.active_agents[idx]]
        
        # Take the sample and add noise, saturate between 0 and 1 with clip#
        noisy_measures = np.clip([self.ground_truth[pose_x, pose_y] + np.random.normal(mean, std) for (pose_x, pose_y), mean, std in zip(position_measures, self.mean_sensormeasure, self.std_sensormeasure)], 0, 1)

        # Variance associated to the measures #
        variance_measures = np.array([self.variance_sensormeasure[idx] for idx in self.active_agents if self.active_agents[idx]]) 

        return position_measures, noisy_measures, variance_measures

if __name__ == '__main__':

    SCENARIO_MAP = np.genfromtxt('Environment/Maps/ypacarai_map_low_res.csv', delimiter=',')
    SELECTED_ALGORITHM = "LawnMower" # LawnMower, WanderingAgent
    N_AGENTS = 4 # max 4
    REWARD_FUNCTION = 'influence_area_changes_model' # only_position_changes_model, influence_area_changes_model, error_with_model
    ACTIVATE_PLOT_RENDER = False

    # Agents info #
    N_AGENTS = 4 # max 4 
    INFLUENCE_LENGTH = 6

    # Sensors info #
    MEAN_SENSORMEASURE = np.array([0, 0, 0, 0])[:N_AGENTS] # mean of the measure of every agent
    STD_SENSORMEASURE = np.array([0.07, 0.07, 0.7, 0.7])[:N_AGENTS] # std of the measure of every agent
    VARIANCE_SENSORMEASURE = STD_SENSORMEASURE**2 # variance = std^2
    AGENT_TYPE = np.searchsorted(np.unique(STD_SENSORMEASURE), STD_SENSORMEASURE) # to difference between agents by its quality

    REGRESSION_LIBRARY = 'gpytorch' # scikit, gpytorch or botorch

    LoadingExecutedAlgorithms(SCENARIO_MAP, SELECTED_ALGORITHM, N_AGENTS, REWARD_FUNCTION, INFLUENCE_LENGTH, MEAN_SENSORMEASURE, STD_SENSORMEASURE, VARIANCE_SENSORMEASURE, REGRESSION_LIBRARY, ACTIVATE_PLOT_RENDER)



