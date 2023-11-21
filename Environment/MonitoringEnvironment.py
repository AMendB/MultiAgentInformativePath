import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import colorcet

from Environment.GroundTruthsModels.ShekelGroundTruth import GroundTruth
from Environment.GroundTruthsModels.AlgaeBloomGroundTruth import algae_bloom

import torch
from GaussianProcess.GPModels import GaussianProcessScikit, GaussianProcessGPyTorch 

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

class DiscreteVehicle: # class for single vehicle

	def __init__(self, initial_position, n_actions, movement_length, influence_length, navigation_map):
		
		""" Initial positions of the drones """
		self.initial_position = initial_position
		self.actual_agent_position = np.copy(initial_position) # set initial position to actual

		""" Initialize the waypoints """
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)

		""" Set other variables """
		self.navigation_map = navigation_map 
		self.distance_traveled = 0.0 
		self.num_of_collisions = 0 # 
		self.angle_set = np.linspace(0, 2 * np.pi, n_actions, endpoint=False) # array with the 8 cardinal points in RADIANS, dividing a cricle in 8 directions: [0. , 0.78539816, 1.57079633, 2.35619449, 3.14159265, 3.92699082, 4.71238898, 5.49778714]
		self.movement_length = movement_length 
		self.influence_length = influence_length 
		self.influence_mask = self.compute_influence_mask()
		

	def move_agent(self, action, valid=True):
		""" Move a vehicle in the direction of the action. If valid is False, the action is not performed. """

		next_position = self.calculate_next_position(action)
		self.distance_traveled += np.linalg.norm(self.actual_agent_position - next_position) # add to the total traveled distance the distance of the actual movement

		if self.check_agent_collision_with_obstacle(next_position) or not valid: # if next positions is a collision (with ground or between agents):
			collide = True
			self.num_of_collisions += 1 # add a collision to the count, but not move the vehicle
		else:
			collide = False
			self.actual_agent_position = next_position # set next position to actual
			self.waypoints = np.vstack((self.waypoints, [self.actual_agent_position])) # add actual position to visited locations array

		self.influence_mask = self.compute_influence_mask() # update influence mask after movement

		return collide # returns if the agent collide
	
	def calculate_next_position(self, action):
		angle = self.angle_set[action] # takes as the angle of movement the angle associated with the action taken, the action serves as the index of the array of cardinal points
		movement = (np.round(np.array([np.cos(angle), np.sin(angle)])) * self.movement_length).astype(int) # converts the angle into cartesian motion (how many cells are moved in x-axis and how many in y-axis).
		next_position = self.actual_agent_position + movement # next position, adds the movement to the current one

		return next_position
		
	def check_agent_collision_with_obstacle(self, next_position):
		""" Return True if the next position leads to a collision """

		if self.navigation_map[int(next_position[0]), int(next_position[1])] == 0: # if 0 in map, there's obstacle
			return True  # There is a collision

		return False
	
	def check_agent_action_with_obstacle(self, test_action):
		""" Return True if the action leads to a collision """
# 
		next_position = self.calculate_next_position(test_action)

		return self.check_agent_collision_with_obstacle(next_position) 
	
	def compute_influence_mask(self): 
		""" Compute influence area (circular mask over the scenario map) around actual position """

		influence_mask = np.zeros_like(self.navigation_map) 

		pose_x, pose_y = self.actual_agent_position.astype(int) 

		# State - coverage area #
		range_x_axis = np.arange(0, self.navigation_map.shape[0]) # posible positions in x-axis
		range_y_axis = np.arange(0, self.navigation_map.shape[1]) # posible positions in y-axis

		# Compute the circular mask (area) #
		mask = (range_x_axis[np.newaxis, :] - pose_x) ** 2 + (range_y_axis[:, np.newaxis] - pose_y) ** 2 <= self.influence_length ** 2 

		influence_mask[mask.T] = 1.0 # converts True values to 1 and False values to 0

		return influence_mask
	
	def reset_agent(self, initial_position):
		""" Reset the agent: Position, waypoints, influence mask, etc. """

		self.initial_position = initial_position
		self.actual_agent_position = np.copy(initial_position)
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)
		self.distance_traveled = 0.0
		self.num_of_collisions = 0
		self.influence_mask = self.compute_influence_mask()


class DiscreteFleet: # class to create FLEETS of class DiscreteVehicle
	""" Coordinator of the movements of the fleet. """

	def __init__(self,
				 number_of_vehicles,
				 fleet_initial_positions,
				 n_actions,
				 movement_length,
				 influence_length,
				 navigation_map,
				 check_collisions_within):

		""" Set init variables """
		self.number_of_vehicles = number_of_vehicles 
		self.initial_positions = fleet_initial_positions
		self.n_actions = n_actions 
		self.movement_length = movement_length 
		self.check_collisions_within = check_collisions_within


		""" Create the vehicles object array """
		self.vehicles = [DiscreteVehicle(initial_position = fleet_initial_positions[k],
										 n_actions = n_actions,
										 movement_length = movement_length,
										 influence_length = influence_length,
										 navigation_map = navigation_map) for k in range(self.number_of_vehicles)]

		self.fleet_positions = np.asarray([veh.actual_agent_position for veh in self.vehicles])

		# Reset fleet number of collisions #
		self.fleet_collisions = 0
									
	def check_collision_within_fleet(self, veh_actions):
		""" Check if there is any collision between agents. Returns boolean array with True to the vehicles with unique new position, i.e., valid actions. """
		
		new_positions = []

		for idx, veh_action in veh_actions.items():
			# Calculate next positions #
			angle = self.vehicles[idx].angle_set[veh_action]
			movement = (np.round(np.array([np.cos(angle), np.sin(angle)])) * self.vehicles[idx].movement_length).astype(int) 
			new_positions.append(list(self.vehicles[idx].actual_agent_position + movement))

		_, inverse_index, counts = np.unique(np.asarray(new_positions), return_inverse=True, return_counts=True, axis=0) # check if unique

		# True if NOT repeated #
		valid_actions_within_fleet = counts[inverse_index] == 1 

		return valid_actions_within_fleet

	def check_fleet_actions_with_obstacle(self, test_actions):
		""" Returns array of bools. True if the action leads to a collision """

		return [self.vehicles[k].check_agent_action_with_obstacle(test_actions[k]) for k in range(self.number_of_vehicles)] 

	def move_fleet(self, fleet_actions):

		if self.check_collisions_within:
			# Check if there are collisions between vehicles #
			valid_actions_within_fleet_mask = self.check_collision_within_fleet(fleet_actions)
			# Process the fleet actions and move the vehicles # 
			collisions_dict = {k: self.vehicles[k].move_agent(fleet_actions[k], valid=valid) for k, valid in zip(list(fleet_actions.keys()), valid_actions_within_fleet_mask)}
		else: 
			collisions_dict = {k: self.vehicles[k].move_agent(fleet_actions[k], valid=True) for k in fleet_actions.keys()}

		# Update vector with agents positions #
		self.fleet_positions = np.asarray([veh.actual_agent_position for veh in self.vehicles])
		# Sum up the collisions for termination #
		self.fleet_collisions = np.sum([self.vehicles[k].num_of_collisions for k in range(self.number_of_vehicles)])

		return collisions_dict # return dict with number of collisions of each vehicle 
		
	def get_fleet_distances_traveled(self):

		return [self.vehicles[k].distance_traveled for k in range(self.number_of_vehicles)]

	def get_fleet_positions(self):

		return np.array([veh.actual_agent_position for veh in self.vehicles])

	def get_distances_between_agents(self):

		distances_dict = {}

		# Calculate the euclidean distances between each pair of agents #
		for i in range(self.number_of_vehicles-1):
			for j in range(i + 1, self.number_of_vehicles):
				x1, y1 = self.vehicles[i].actual_agent_position
				x2, y2 = self.vehicles[j].actual_agent_position
				distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
				
				distances_dict[f'Distance_{i}{j}'] = distance
	

		return distances_dict
	
	def reset_fleet(self, initial_positions=None):
		""" Reset the fleet """

		if initial_positions is None:
			initial_positions = self.initial_positions

		# Reset each agent #
		for k in range(self.number_of_vehicles):
			self.vehicles[k].reset_agent(initial_position=initial_positions[k])

		# Assign initial positions to each agent #
		self.fleet_positions = np.asarray([veh.actual_agent_position for veh in self.vehicles])

		# Reset number of collisions #
		self.fleet_collisions = 0

	def get_positions(self): 
		return np.array([veh.actual_agent_position for veh in self.vehicles])

class MultiAgentMonitoring:

	def __init__(self, 
	      		 scenario_map,
				 number_of_agents,
				 max_distance_travelled,
				 mean_sensormeasure,
				 range_std_sensormeasure,
				 std_sensormeasure,
				 fleet_initial_positions = None,
				 seed = 0,
				 movement_length = 2,
				 influence_length = 2,
				 flag_to_check_collisions_within = False,
				 max_collisions = 5,
				 reward_function = 'Influence_area_changes_model', 
				 ground_truth_type = 'algae_bloom', 
				 peaks_location = 'Random', 
				 dynamic = False, 
				 obstacles = False,
				 regression_library = 'scikit',
				 scale_kernel = True,
				 reward_weights = (1.0, 0.1),
				 show_plot_graphics = True,
				 ):

		""" The gym environment """

		# Load the scenario config
		self.seed = seed
		np.random.seed(self.seed)
		self.rng_positions = np.random.default_rng(seed=self.seed)
		self.rng_std_sensormeasure = np.random.default_rng(seed=self.seed)
		self.scenario_map = scenario_map
		self.n_agents = number_of_agents
		self.n_actions = 8
		self.dynamic = dynamic
		self.obstacles = obstacles
		self.reward_function = reward_function
		self.visitable_locations = np.vstack(np.where(self.scenario_map != 0)).T # coords of visitable cells
		self.flag_to_check_collisions_within = flag_to_check_collisions_within

		# Initial positions #
		self.backup_fleet_initial_positions_entry = fleet_initial_positions
		if isinstance(fleet_initial_positions, np.ndarray): # Set initial positions if indicated #
			self.random_inititial_positions = False
			self.initial_positions = fleet_initial_positions
		elif fleet_initial_positions is None: # Random positions all visitable map #
			self.random_inititial_positions = True
			random_positions_indx = self.rng_positions.choice(np.arange(0, len(self.visitable_locations)), self.n_agents, replace=False) # a random index is selected as the maximum number of cells that can be visited
			self.initial_positions = self.visitable_locations[random_positions_indx] 
		elif fleet_initial_positions == 'fixed': # Random choose between 4 fixed deployment positions #
			self.random_inititial_positions = 'fixed'
			self.deployment_positions = np.zeros_like(self.scenario_map)
			self.deployment_positions[[46,46,49,49], [28,31,28,31]] = 1
			self.initial_positions = np.argwhere(self.deployment_positions == 1)[self.rng_positions.choice(len(np.argwhere(self.deployment_positions == 1)), self.n_agents, replace=False)]
		elif fleet_initial_positions == 'area': # Random deployment positions inside an area #
			self.random_inititial_positions = 'area'
			self.deployment_positions = np.zeros_like(self.scenario_map)
			self.deployment_positions[slice(45,50), slice(27,32)] = 1
			self.initial_positions = np.argwhere(self.deployment_positions == 1)[self.rng_positions.choice(len(np.argwhere(self.deployment_positions == 1)), self.n_agents, replace=False)]
		else:
			raise NotImplementedError("Check initial positions!")

		# Load sensors noise #
		self.mean_sensormeasure = mean_sensormeasure
		self.range_std_sensormeasure = range_std_sensormeasure
		self.backup_std_sensormeasure_entry = std_sensormeasure
		if isinstance(std_sensormeasure, np.ndarray):
			self.random_std_sensormeasure = False
			self.std_sensormeasure = std_sensormeasure
		elif std_sensormeasure == 'random':
			self.random_std_sensormeasure = True
			self.std_sensormeasure = self.rng_std_sensormeasure.uniform(self.range_std_sensormeasure[0], self.range_std_sensormeasure[1], self.n_agents)
		
		self.scale_std = (1 - 0.25) / (self.range_std_sensormeasure[1] - self.range_std_sensormeasure[0]) # scale std between 0.25 and 1
		self.scaled_std_sensormeasure = np.abs((self.std_sensormeasure - self.range_std_sensormeasure[0]) * self.scale_std - 1) # used to inform network the std of every agent though the state (0.25 worst, 1 best)
		self.variance_sensormeasure = self.std_sensormeasure**2 # variance = std^2
		self.normalized_variance_sensormeasure =  (self.variance_sensormeasure - self.range_std_sensormeasure[0]**2) / (self.range_std_sensormeasure[1]**2 - self.range_std_sensormeasure[0]**2 ) # used in DQN as input for network_with_sensornoises to normalize variance between 0 to 1

		self.sensors_type = np.searchsorted(np.unique(self.std_sensormeasure), self.std_sensormeasure) # to difference between agents by its quality
		self.n_sensors_type = len(np.unique(self.std_sensormeasure))
		self.masks_by_type = [self.sensors_type == type for type in range(self.n_sensors_type)]
	
		# Limits to be declared a death/done agent and initialize done dict #
		self.max_distance_travelled = max_distance_travelled
		self.max_collisions = max_collisions
		self.done = {i:False for i in range(self.n_agents)} 
		self.dones_by_sensors_types = {type: False for type in range(self.n_sensors_type)}  
		self.active_agents = {key: not value for key, value in self.done.items()}
		self.n_active_agents = sum(self.active_agents.values())
 
		# Fleet of N vehicles #
		self.movement_length = movement_length
		self.influence_length = influence_length
		self.reward_weights = reward_weights
		
		# Create the fleets #
		self.fleet = DiscreteFleet(number_of_vehicles = self.n_agents,
								   fleet_initial_positions = self.initial_positions,
								   n_actions = self.n_actions,
								   movement_length = movement_length,
								   influence_length = influence_length,
								   navigation_map = self.scenario_map,
								   check_collisions_within = flag_to_check_collisions_within)

		# Generate Ground Truth #
		self.ground_truth_type = ground_truth_type
		if ground_truth_type == 'shekel':
			self.ground_truth = GroundTruth(self.scenario_map, max_number_of_peaks = 4, is_bounded = True, seed = self.seed, peaks_location=peaks_location)
		elif ground_truth_type == 'algae_bloom':
			self.ground_truth = algae_bloom(self.scenario_map, seed = self.seed)
		else:
			raise NotImplementedError("This Benchmark is not implemented. Choose one that is.")

		# Model maps #
		self.model_mean_map = np.zeros_like(self.scenario_map) 
		self.model_uncertainty_map = np.zeros_like(self.scenario_map) 


		# Init the redundancy mask #
		self.redundancy_mask = np.sum([agent.influence_mask for idx, agent in enumerate(self.fleet.vehicles) if self.active_agents[idx]], axis = 0)

		# Create Gaussian Process #
		self.regression_library = regression_library
		self.scale_kernel = scale_kernel
		if self.regression_library == 'scikit':
			self.gaussian_process = GaussianProcessScikit(scenario_map = self.scenario_map, initial_lengthscale = 5.0, kernel_bounds = (0.1, 20))
		elif self.regression_library == 'gpytorch':
			if self.scale_kernel == True:
				self.gaussian_process = GaussianProcessGPyTorch(scenario_map = self.scenario_map, initial_lengthscale = 5.0, kernel_bounds = (0.1, 20), training_iterations = 50, scale_kernel=True, device = 'cuda' if torch.cuda.is_available() else 'cpu')
				self.max_std_scale = 0.19 # empirically selectioned to scale uncertainty map to obtain more contrast and higher rewards differences
			else:
				self.gaussian_process = GaussianProcessGPyTorch(scenario_map = self.scenario_map, initial_lengthscale = 5.0, kernel_bounds = (0.1, 20), training_iterations = 50, scale_kernel=False, device = 'cuda' if torch.cuda.is_available() else 'cpu')
		else:
			raise NotImplementedError("This library is not implemented. Choose one that is.")

		# Visualization #
		self.activate_plot_graphics = show_plot_graphics
		self.states = None
		self.state_to_render_first_active_agent = None
		self.render_fig = None
		self.colored_agents = True
		if self.colored_agents:
			self.colors_agents = ['black', 'gainsboro']
			palettes_by_sensors_type = {0: ['green', 'mediumseagreen', 'seagreen', 'olive'], 1: ['darkred', 'indianred', 'tomato'], 2: ['sandybrown', 'peachpuff'], 3: ['darkmagenta']}
			for agent in range(self.n_agents):
				self.colors_agents.extend([palettes_by_sensors_type[self.sensors_type[agent]].pop(0)])
			self.agents_colormap = matplotlib.colors.ListedColormap(self.colors_agents)
			self.n_colors_agents_render = len(self.colors_agents)

		# Info for training # 
		self.observation_space_shape = (5, *self.scenario_map.shape)

	def reset_env(self):
		""" Reset the environment """
			
		# Reset the ground truth #
		self.ground_truth.reset()

		# Randomly generated obstacles #
		if self.obstacles:
			self.inside_obstacles_map = np.zeros_like(self.scenario_map)
			# Generate a random inside obstacles map #
			obstacles_pos_indx = np.random.choice(np.arange(0, len(self.visitable_locations)), size = 20, replace = False)
			self.inside_obstacles_map[self.visitable_locations[obstacles_pos_indx, 0], self.visitable_locations[obstacles_pos_indx, 1]] = 1.0

			# Update the obstacle map for every agent #
			for i in range(self.n_agents):
				self.fleet.vehicles[i].navigation_map = self.scenario_map - self.inside_obstacles_map
		else:
			self.inside_obstacles_map = np.zeros_like(self.scenario_map)
		
		self.non_water_mask = self.scenario_map != 1 - self.inside_obstacles_map # mask with True where no water

		# Create an empty model after reset #
		self.model_mean_map = np.zeros_like(self.scenario_map) 
		self.model_uncertainty_map = np.zeros_like(self.scenario_map) 
		self.previous_model_mean_map = self.model_mean_map.copy()
		self.previous_model_uncertainty_map = self.model_uncertainty_map.copy()
		self.gaussian_process.reset()

		# Get new std if random True #
		if self.random_std_sensormeasure:
			self.std_sensormeasure = self.rng_std_sensormeasure.uniform(self.range_std_sensormeasure[0], self.range_std_sensormeasure[1], self.n_agents)
			self.scaled_std_sensormeasure = np.abs((self.std_sensormeasure - self.range_std_sensormeasure[0]) * self.scale_std - 1) # used to inform network the std of every agent though the state (0.25 worst, 1 best)
			self.variance_sensormeasure = self.std_sensormeasure**2 # variance = std^2
			self.normalized_variance_sensormeasure =  (self.variance_sensormeasure - self.range_std_sensormeasure[0]**2) / (self.range_std_sensormeasure[1]**2 - self.range_std_sensormeasure[0]**2 ) # used in DQN as input for network_with_sensornoises to normalize variance between 0 to 1

			self.sensors_type = np.searchsorted(np.unique(self.std_sensormeasure), self.std_sensormeasure) # to difference between agents by its quality
			self.n_sensors_type = len(np.unique(self.std_sensormeasure))
			self.masks_by_type = [self.sensors_type == type for type in range(self.n_sensors_type)]

			# New info for visualization #
			if self.colored_agents:
				self.colors_agents = ['black', 'gainsboro']
				palettes_by_sensors_type = {0: ['green', 'mediumseagreen', 'seagreen', 'olive'], 1: ['darkred', 'indianred', 'tomato'], 2: ['sandybrown', 'peachpuff'], 3: ['darkmagenta']}
				for agent in range(self.n_agents):
					self.colors_agents.extend([palettes_by_sensors_type[self.sensors_type[agent]].pop(0)])
				self.agents_colormap = matplotlib.colors.ListedColormap(self.colors_agents)
				self.n_colors_agents_render = len(self.colors_agents)

		# Get the N random initial positions #
		if self.random_inititial_positions == 'area' or self.random_inititial_positions == 'fixed':
			self.initial_positions = np.argwhere(self.deployment_positions == 1)[self.rng_positions.choice(len(np.argwhere(self.deployment_positions == 1)), self.n_agents, replace=False)]
		elif self.random_inititial_positions is True:
			random_positions_indx = self.rng_positions.choice(np.arange(0, len(self.visitable_locations)), self.n_agents, replace=False)
			self.initial_positions = self.visitable_locations[random_positions_indx]
		
		# Reset the positions of the fleet #
		self.fleet.reset_fleet(initial_positions=self.initial_positions)
		self.done = {agent_id: False for agent_id in range(self.n_agents)}
		self.dones_by_sensors_types = {type: False for type in range(self.n_sensors_type)}  
		self.active_agents = {agent_id: True for agent_id in range(self.n_agents)}
		self.n_active_agents = sum(self.active_agents.values())
		
		# Compute the redundancy mask after reset #
		self.redundancy_mask = np.sum([agent.influence_mask for idx, agent in enumerate(self.fleet.vehicles) if self.active_agents[idx]], axis = 0)

		# Take samples and update model #
		self.update_model()
		self.error_with_gt_backup = np.abs(self.ground_truth.read() - self.model_mean_map)

		# Update the photograph/state of the agents #
		self.capture_states()
		
		# Reset visualization #
		if self.render_fig is not None and self.activate_plot_graphics:
			plt.close(self.render_fig)
			self.render_fig = None

		return self.states

	def take_samples(self):
		""" The active agents take a noisy sample from the ground truth """
		
		# Get the Ground Truth #
		ground_truth = self.ground_truth.read() 

		# Save positions where samples are taken #
		position_measures = [[agent.actual_agent_position[0], agent.actual_agent_position[1]] for idx, agent in enumerate(self.fleet.vehicles) if self.active_agents[idx]]
		
		# Take the sample and add noise, saturate between 0 and 1 with clip#
		noisy_measures = np.clip([ground_truth[pose_x, pose_y] + np.random.normal(mean, std) for (pose_x, pose_y), mean, std in zip(position_measures, self.mean_sensormeasure, self.std_sensormeasure)], 0, 1)

		# Variance associated to the measures #
		variance_measures = np.array([self.variance_sensormeasure[idx] for idx in self.active_agents if self.active_agents[idx]]) 

		return position_measures, noisy_measures, variance_measures
	
	def update_model(self):

		# Sensor samples #
		position_new_measures, new_measures, variance_measures = self.take_samples()
		
		# Fit gaussian process with new samples #
		self.gaussian_process.fit_gp(X_new=position_new_measures, y_new=new_measures, variances_new=variance_measures)
		
		# Update model: prediction of ground truth #
		self.previous_model_mean_map = self.model_mean_map.copy()
		self.previous_model_uncertainty_map = self.model_uncertainty_map.copy()
		self.model_mean_map, self.model_uncertainty_map = self.gaussian_process.predict_gt()

		# Saturate prediction between [0, 1] as expected #
		self.model_mean_map = np.clip( self.model_mean_map, 0, 1 )
		if self.scale_kernel == True:
			self.model_uncertainty_map = np.clip( self.model_uncertainty_map / self.max_std_scale, 0, 1 )		
		else:
			self.model_uncertainty_map = np.clip( self.model_uncertainty_map, 0, 1 )

	def step(self, actions: dict):
		"""Execute all updates for each step"""

		# Update ground truth if dynamic #
		if self.dynamic:
			self.ground_truth.step()

		# Process movement actions, there is actions only for active agents #
		collisions_mask_dict = self.fleet.move_fleet(actions)

		if self.fleet.fleet_collisions > 0 and any(collisions_mask_dict.values()):
			print("Nº collision:" + str(self.fleet.fleet_collisions))
		
		# Update the redundancy mask after movements #
		self.redundancy_mask = np.sum([agent.influence_mask for idx, agent in enumerate(self.fleet.vehicles) if self.active_agents[idx]], axis = 0)
		
		# Take samples and update model #
		self.update_model()

		# Compute reward #
		rewards = self.get_reward(collisions_mask_dict, actions)

		# Update the photograph/state of the agents #
		self.capture_states()

		# Plot graphics if activated #
		if self.activate_plot_graphics:
			self.render()

		# Final conditions #
		self.done = {agent_id: (self.fleet.get_fleet_distances_traveled()[agent_id] > self.max_distance_travelled or self.fleet.fleet_collisions > self.max_collisions) for agent_id in range(self.n_agents)}
		# self.done[0] = True # UNDER TEST
		# self.done[2] = True # UNDER TEST
		self.dones_by_sensors_types = {actual_type: all([is_done for agent_id, is_done in self.done.items() if self.sensors_type[agent_id] == actual_type]) for actual_type in range(self.n_sensors_type)}  
		self.active_agents = {key: not value for key, value in self.done.items()}
		self.n_active_agents = sum(self.active_agents.values())

		return self.states, rewards, self.done

	def capture_states(self):
		""" Update the photograph/state for every vehicle. Every channel will be an input of the Neural Network. """

		states = {}
		# Channel 0 -> Known boundaries/map
		if self.obstacles:
			obstacle_map = self.scenario_map - self.inside_obstacles_map
		else:
			obstacle_map = self.scenario_map

		# Create fleet position map #
		# fleet_position_map = np.zeros_like(self.scenario_map)
		# fleet_position_map[self.fleet.fleet_positions[:,0], self.fleet.fleet_positions[:,1]] = 1.0 # set 1 where there is an agent
		fleet_position_map_like_stds = np.zeros_like(self.scenario_map)
		for agent_id in range(self.n_agents):
			fleet_position_map_like_stds[self.fleet.fleet_positions[agent_id,0], self.fleet.fleet_positions[agent_id,1]] = self.scaled_std_sensormeasure[agent_id] # set its scaled std where there is the agent

		if self.colored_agents == True and self.activate_plot_graphics:
			fleet_position_map_colored = np.zeros_like(self.scenario_map)
			for agent_id in self.get_active_agents_positions_dict().keys():
					fleet_position_map_colored[self.fleet.fleet_positions[agent_id,0], self.fleet.fleet_positions[agent_id,1]] = (1/self.n_colors_agents_render)*(agent_id+2) + 0.01

		first_available_agent = np.argmax(list(self.active_agents.values())) # first True in active_agents
		for agent_id, active in self.active_agents.items():
			if active:
				observing_agent_position = np.zeros_like(self.scenario_map)
				# observing_agent_position[self.fleet.fleet_positions[agent_id,0], self.fleet.fleet_positions[agent_id,1]] = 1.0 # map only with the position of the observing agent
				observing_agent_position[self.fleet.fleet_positions[agent_id,0], self.fleet.fleet_positions[agent_id,1]] = self.scaled_std_sensormeasure[agent_id] # map only with the position with its scaled std of the observing agent
				
				# agent_observation_of_fleet = fleet_position_map.copy()
				agent_observation_of_fleet = fleet_position_map_like_stds.copy()
				agents_to_remove_positions = np.array([pos for idx, pos in enumerate(self.fleet.fleet_positions) if (idx == agent_id) or (not self.active_agents[idx])])  # if observing agent, or not active
				agent_observation_of_fleet[agents_to_remove_positions[:,0], agents_to_remove_positions[:,1]] = 0.0 # agents map without the observing agent

				"""Each key from states dictionary is an agent, all states associated to that agent are concatenated in its value:"""
				states[agent_id] = np.concatenate(( 
					obstacle_map[np.newaxis], # Channel 0 -> Known boundaries/map
					self.model_mean_map[np.newaxis], # Channel 1 -> Model mean map
					self.model_uncertainty_map[np.newaxis], # Channel 2 -> Model uncertainty map
					observing_agent_position[np.newaxis], # Channel 3 -> Observing agent position map
					agent_observation_of_fleet[np.newaxis], # Channel 4 -> Others active agents position map
				))

				if agent_id == first_available_agent and self.activate_plot_graphics:
					if self.colored_agents == True:
						self.state_to_render_first_active_agent = np.concatenate(( 
							self.scenario_map[np.newaxis],
							self.ground_truth.read()[np.newaxis],
							self.model_mean_map[np.newaxis],
							self.model_uncertainty_map[np.newaxis],
							fleet_position_map_colored[np.newaxis],
							self.redundancy_mask[np.newaxis]
						))
					
					else:
						self.state_to_render_first_active_agent = np.concatenate(( 
							self.scenario_map[np.newaxis],
							self.ground_truth.read()[np.newaxis],
							self.model_mean_map[np.newaxis],
							self.model_uncertainty_map[np.newaxis],
							observing_agent_position[np.newaxis],
							agent_observation_of_fleet[np.newaxis],
							self.redundancy_mask[np.newaxis]
						))

		self.states = {agent_id: states[agent_id] for agent_id in range(self.n_agents) if self.active_agents[agent_id]}
		# self.states = {agent_id: np.uint16(states[agent_id]*65535) for agent_id in range(self.n_agents) if self.active_agents[agent_id]} # convert to uin16 to save in buffer
		# self.states = {agent_id: np.float64(self.states[agent_id]/65535) for agent_id in self.states.keys()} # reconvert to float64 to pass to neural network

	def render(self):
		""" Print visual representation of each state of the scenario. """

		if not any(self.active_agents.values()):
			return
		
		if self.render_fig is None: # create first frame of fig, if not already created

			if self.colored_agents == True:
				self.render_fig, self.axs = plt.subplots(1, 6, figsize=(17,5))
			else:
				self.render_fig, self.axs = plt.subplots(1, 7, figsize=(17,5))
			
			# AXIS 0: Print the obstacles map #
			self.im0 = self.axs[0].imshow(self.state_to_render_first_active_agent[0], cmap = 'cet_linear_bgy_10_95_c74')
			self.axs[0].set_title('Navigation map')

			# AXIS 1: Print the Ground Truth #
			self.state_to_render_first_active_agent[1][self.non_water_mask] = np.nan
			self.im1 = self.axs[1].imshow(self.state_to_render_first_active_agent[1], cmap ='cet_linear_bgy_10_95_c74', vmin = 0.0, vmax = 1.0)
			self.axs[1].set_title("Real Importance (GT)")

			# AXIS 2: Print model mean #
			self.state_to_render_first_active_agent[2][self.non_water_mask] = np.nan
			self.im2 = self.axs[2].imshow(self.state_to_render_first_active_agent[2], cmap ='cet_linear_bgy_10_95_c74', vmin = 0.0, vmax = 1.0)
			self.axs[2].set_title("Model Mean")

			# AXIS 3: Print model uncertainty #
			self.state_to_render_first_active_agent[3][self.non_water_mask] = np.nan
			self.im3 = self.axs[3].imshow(self.state_to_render_first_active_agent[3], cmap ='gray', vmin = 0.0, vmax = 1.0)#, norm='log')
			self.axs[3].set_title("Model Uncertainty")

			if self.colored_agents == True:
				# AXIS 4: Active colored agents positions #
				self.state_to_render_first_active_agent[4][self.non_water_mask] = 1/self.n_colors_agents_render + 0.01
				self.im4 = self.axs[4].imshow(self.state_to_render_first_active_agent[4], cmap = self.agents_colormap, vmin = 0.0, vmax = 1.0)
				self.axs[4].set_title("Agents position")

				# AXIS 5: Redundancy mask #
				self.state_to_render_first_active_agent[5][self.non_water_mask] = np.nan
				self.im5 = self.axs[5].imshow(self.state_to_render_first_active_agent[5], cmap = 'cet_linear_bgy_10_95_c74', vmin = 0.0, vmax = 4.0)
				self.axs[5].set_title("Redundancy mask")
			else:
				# AXIS 4: Agent 0 position #
				self.state_to_render_first_active_agent[4][self.non_water_mask] = 0.75
				self.im4 = self.axs[4].imshow(self.state_to_render_first_active_agent[4], cmap = 'gray', vmin = 0.0, vmax = 1.0)
				self.axs[4].set_title("Agent 0 position")

				# AXIS 5: Others-than-Agent 0 positions #
				self.state_to_render_first_active_agent[5][self.non_water_mask] = 0.75
				self.im5 = self.axs[5].imshow(self.state_to_render_first_active_agent[5], cmap = 'gray', vmin = 0.0, vmax = 1.0)
				self.axs[5].set_title("Others agents position")

				# AXIS 6: Redundancy mask #
				self.state_to_render_first_active_agent[6][self.non_water_mask] = np.nan
				self.im6 = self.axs[6].imshow(self.state_to_render_first_active_agent[6], cmap = 'cet_linear_bgy_10_95_c74', vmin = 0.0, vmax = 4.0)
				self.axs[6].set_title("Redundancy mask")

		else:
			# UPDATE FIG INFO/DATA IN EVERY RENDER CALL #
			# AXIS 0: Print the obstacles map #
			self.im0.set_data(self.state_to_render_first_active_agent[0])
			# AXIS 1: Print the Ground Truth #
			self.state_to_render_first_active_agent[1][self.non_water_mask] = np.nan
			self.im1.set_data(self.state_to_render_first_active_agent[1])
			# AXIS 2: Print model mean #
			self.state_to_render_first_active_agent[2][self.non_water_mask] = np.nan
			self.im2.set_data(self.state_to_render_first_active_agent[2])
			# AXIS 3: Print model uncertainty #
			self.state_to_render_first_active_agent[3][self.non_water_mask] = np.nan
			self.im3.set_data(self.state_to_render_first_active_agent[3])
			if self.colored_agents == True:
				# AXIS 4: Active colored agents positions #
				self.state_to_render_first_active_agent[4][self.non_water_mask] = 1/self.n_colors_agents_render + 0.01
				self.im4.set_data(self.state_to_render_first_active_agent[4])

				# AXIS 5: Redundancy mask #
				self.state_to_render_first_active_agent[5][self.non_water_mask] = np.nan
				self.im5.set_data(self.state_to_render_first_active_agent[5])
			else:
				# AXIS 4: Agent 0 position #
				self.state_to_render_first_active_agent[4][self.non_water_mask] = 0.75
				self.im4.set_data(self.state_to_render_first_active_agent[4])
				# AXIS 5: Others-than-Agent 0 positions #
				self.state_to_render_first_active_agent[5][self.non_water_mask] = 0.75
				self.im5.set_data(self.state_to_render_first_active_agent[5])
				# AXIS 6: Redundancy mask #
				self.state_to_render_first_active_agent[6][self.non_water_mask] = np.nan
				self.im6.set_data(self.state_to_render_first_active_agent[6])

		plt.draw()	
		plt.pause(0.01)

	def get_reward(self, collisions_mask_dict, actions):
		""" Reward function

			1) model_changes:
			r(t) = Sum( W(m)/Dr(m) ) + |I_t-1(m) - It(m)|/Dr(m)
		"""

		if self.reward_function == 'Position_changes_model':
			
			changes_in_model_mean = np.abs(self.model_mean_map - self.previous_model_mean_map)
			changes_in_model_uncertainty = np.abs(self.model_uncertainty_map - self.previous_model_uncertainty_map)

			changes_mean = np.array(
				[np.sum(
					changes_in_model_mean[agent.actual_agent_position[0], agent.actual_agent_position[1]]
					) if self.active_agents[idx] else 0 for idx, agent in enumerate(self.fleet.vehicles)
					]
				)			
			
			changes_uncertinty = np.array(
				[np.sum(
					changes_in_model_uncertainty[agent.actual_agent_position[0], agent.actual_agent_position[1]]
					) if self.active_agents[idx] else 0 for idx, agent in enumerate(self.fleet.vehicles)
					]
				)
			
			rewards = self.reward_weights[0] * changes_mean + self.reward_weights[1] * changes_uncertinty

		elif self.reward_function == 'Influence_area_changes_model':

			changes_in_model_mean = np.abs(self.model_mean_map - self.previous_model_mean_map)
			changes_in_model_uncertainty = np.abs(self.model_uncertainty_map - self.previous_model_uncertainty_map)

			changes_mean = np.array(
				[np.sum(
					changes_in_model_mean[agent.influence_mask.astype(bool)] / self.redundancy_mask[agent.influence_mask.astype(bool)]
					) if self.active_agents[idx] else 0 for idx, agent in enumerate(self.fleet.vehicles)
					]
				)
			
			changes_uncertinty = np.array(
				[np.sum(
					changes_in_model_uncertainty[agent.influence_mask.astype(bool)]  / self.redundancy_mask[agent.influence_mask.astype(bool)]
					) if self.active_agents[idx] else 0 for idx, agent in enumerate(self.fleet.vehicles)
					]
				)

			rewards = self.reward_weights[0] * changes_mean + self.reward_weights[1] * changes_uncertinty
		
		elif self.reward_function == 'Error_with_model':

			ponderation_maps = [np.zeros_like(self.scenario_map) for _ in range(self.n_agents)]
			for i, j in np.argwhere(self.redundancy_mask > 0):
					# Check if the influence area of every agent is in the pixel. If true, save its id
					agents_in_pixel = [agent_id for agent_id, agent in enumerate(self.fleet.vehicles) if agent.influence_mask[i,j] == 1 and self.active_agents[agent_id]]
					stds_in_pixel = [1 / self.std_sensormeasure[agent_id] if agent_id in agents_in_pixel else 0 for agent_id in range(self.n_agents)]
					ponderations = stds_in_pixel/(np.sum(stds_in_pixel))
					for agent_id, ponderation in enumerate(ponderations):
						ponderation_maps[agent_id][i,j] = ponderation
					
			error_with_gt = np.abs(self.ground_truth.read() - self.model_mean_map)
			error_improve = self.error_with_gt_backup - error_with_gt
			self.error_with_gt_backup = error_with_gt.copy()

			# Inverse error to obtain higher reward when error is lower
			ponderated_improvement = np.array(
				[np.sum(
					error_improve[agent.influence_mask.astype(bool)]  * ponderation_maps[idx][agent.influence_mask.astype(bool)]
					) if self.active_agents[idx] else 0 for idx, agent in enumerate(self.fleet.vehicles)
					]
				)
			
			rewards = self.reward_weights[0] * ponderated_improvement 

		cost = {agent_id: 1 if action % 2 == 0 else np.sqrt(2) for agent_id, action in actions.items()} # movements cost (difference between horizontal and diagonal)
		rewards = {agent_id: rewards[agent_id]/cost[agent_id] if not collisions_mask_dict[agent_id] else -1.0 for agent_id in actions.keys()} # save calculated agent reward/cost, penalization if collision

		return {agent_id: rewards[agent_id] if self.active_agents[agent_id] else 0 for agent_id in range(self.n_agents)}
	
	
	def get_active_agents_positions_dict(self):

		return {idx: veh.actual_agent_position for idx, veh in enumerate(self.fleet.vehicles) if self.active_agents[idx]}

	def get_gt_in_visitable_locations(self):
		""" Returns the ground truth values only for visitable locations """

		return self.ground_truth.read()[self.visitable_locations[:, 0], self.visitable_locations[:, 1]]

	def get_model_mu_in_visitable_locations(self):
		""" Returns the model mean values only for visitable locations """

		return self.model_mean_map[self.visitable_locations[:, 0], self.visitable_locations[:, 1]]

	def get_model_mu_mean_abs_error(self):
			""" Returns the absolute error """

			return mean_absolute_error(self.get_gt_in_visitable_locations(), self.get_model_mu_in_visitable_locations())

	def get_model_mu_mse_error(self, squared = False):
			""" Returns the MSE error """

			return mean_squared_error(self.get_gt_in_visitable_locations(), self.get_model_mu_in_visitable_locations(), squared = squared)

	def get_model_mu_mse_error_in_peaks(self, squared = False):
			""" Returns the MSE error in peaks of the ground truth """

			gt_in_visitable_locations = self.get_gt_in_visitable_locations()
			model_mu_in_visitable_locations = self.get_model_mu_in_visitable_locations()
			
			peaks_mask = np.where(self.get_gt_in_visitable_locations() > 0.9, True, False)

			gt_in_peaks = gt_in_visitable_locations[peaks_mask]
			model_mu_in_peaks = model_mu_in_visitable_locations[peaks_mask]

			return mean_squared_error(y_true=gt_in_peaks, y_pred=model_mu_in_peaks, sample_weight=gt_in_peaks, squared=squared)

	def get_model_mu_r2_error(self):
			""" Returns the R2 error """

			return r2_score(self.get_gt_in_visitable_locations(), self.get_model_mu_in_visitable_locations())

	def get_uncertainty_mean(self):
			""" Returns the mean of uncertainty """
			
			return np.mean(self.model_uncertainty_map)

	def get_uncertainty_max(self):
			""" Returns the max of uncertainty """
			
			return np.max(self.model_uncertainty_map)

	def get_redundancy_max(self):
			""" Returns the max number of agents that are in overlapping areas. """
			
			return np.max(self.redundancy_mask)
	
	def save_environment_configuration(self, path):
		""" Save the environment configuration in the current directory as a json file"""

		environment_configuration = {

			'scenario_map': self.scenario_map.tolist(),
			'number_of_agents': self.n_agents,
			'fleet_initial_positions': self.backup_fleet_initial_positions_entry if isinstance(self.backup_fleet_initial_positions_entry, str) else self.backup_fleet_initial_positions_entry.tolist(),
			'max_distance_travelled': self.max_distance_travelled,
			'mean_sensormeasure': self.mean_sensormeasure.tolist(),
			'range_std_sensormeasure': self.range_std_sensormeasure,
			'std_sensormeasure': self.backup_std_sensormeasure_entry if isinstance(self.backup_std_sensormeasure_entry, str) else self.backup_std_sensormeasure_entry.tolist(),
			'seed': self.seed,
			'movement_length': self.movement_length,
			'influence_length': self.influence_length,
			'flag_to_check_collisions_within': self.flag_to_check_collisions_within,
			'max_collisions': self.max_collisions,
			'reward_function': self.reward_function,
			'reward_weights': self.reward_weights,
			'ground_truth_type': self.ground_truth_type,
			'dynamic': self.dynamic,
			'obstacles': self.obstacles,
			'regression_library': self.regression_library,
			'scale_kernel': self.scale_kernel,
		}

		with open(path + '/environment_config.json', 'w') as f:
			json.dump(environment_configuration, f)


if __name__ == '__main__':

	from Algorithms.DRL.ActionMasking.ActionMaskingUtils import ConsensusSafeActionMasking
	
	seed = 3
	np.random.seed(seed)
	scenario_map = np.genfromtxt('Environment/Maps/ypacarai_map_low_res.csv', delimiter=',')
	# scenario_map = np.genfromtxt('Environment/Maps/ypacarai_lake_58x41.csv', delimiter=',')

	# Agents info #
	n_agents = 4 # max 4 
	movement_length = 2
	influence_length = 6
	
	# Sensors info #
	mean_sensormeasure = np.array([0, 0, 0, 0])[:n_agents] # mean of the measure of every agent
	range_std_sensormeasure = (1*0.5/100, 1*0.5*100/100) # AML is "the best", from then on 100 times worse
	std_sensormeasure = np.array([0.05, 0.10, 0.20, 0.40])[:n_agents] # std of the measure of every agent
	# std_sensormeasure = 'random'


	# Set initial positions #
	random_initial_positions = False
	if random_initial_positions:
		initial_positions = 'fixed'
	else:
		# initial_positions = np.array([[30, 20], [40, 25], [40, 20], [30, 28]])[:n_agents, :]
		initial_positions = np.array([[46, 28], [46, 31], [49, 28], [49, 31]])[:n_agents, :]

	# Create environment # 
	env = MultiAgentMonitoring(scenario_map = scenario_map,
							   number_of_agents = n_agents,
							   max_distance_travelled = 100,
							   mean_sensormeasure = mean_sensormeasure,
							   range_std_sensormeasure = range_std_sensormeasure,
							   std_sensormeasure = std_sensormeasure, # std array or 'random'
							   fleet_initial_positions = initial_positions, # None, 'area', 'fixed' or positions array
							   seed = seed,
							   movement_length = movement_length,
							   influence_length = influence_length,
							   flag_to_check_collisions_within = True,
							   max_collisions = 1000,
							   reward_function = 'Error_with_model',  # Position_changes_model, Influence_area_changes_model, Error_with_model
							   ground_truth_type = 'shekel',
							   dynamic = False,
							   obstacles = False,
							   regression_library = 'gpytorch', # scikit, gpytorch
							   scale_kernel  = True,
							   reward_weights = (1.0, 0.1),
							   show_plot_graphics = True,
							 )
	
	action_masking_module = ConsensusSafeActionMasking(navigation_map = scenario_map, action_space_dim = 8, movement_length = movement_length)
 
	env.reset_env()
	env.render()

	R = [] # reward
	MEAN_ABS_ERROR = [] 

	actions = {i: np.random.randint(0,8) for i in range(n_agents)} 
	done = {i:False for i in range(n_agents)} 

	while any([not value for value in done.values()]): # while at least 1 active
	
		q = {idx: np.random.rand(8) for idx in range(n_agents) if env.active_agents[idx]} # only generate q values for active agents

		for agent_id, action in actions.items(): 
			if env.active_agents[agent_id]:
				q[agent_id][action] = 1000 # overwrite q of actual action to a very high value, so it will be selected until collision
	
		actions = action_masking_module.query_actions(q, env.get_active_agents_positions_dict()) # only generate actions for active agents
		
		s, r, done = env.step(actions)

		R.append(list(r.values()))
		MEAN_ABS_ERROR.append(env.get_model_mu_mean_abs_error())

		print("Actions: " + str(dict(sorted(actions.items()))))
		print("Rewards: " + str(r))

	env.render()
	plt.show()

	# Reward and Error final graphs #
	final_fig, final_axes = plt.subplots(1, 2, figsize=(15,5))

	final_axes[0].plot(np.cumsum(np.asarray(R),axis=0), '-o')
	final_axes[0].set(title = 'Reward', xlabel = 'Step', ylabel = 'Individual Reward')
	final_axes[0].legend([f'Agent {i}' for i in range(n_agents)])
	final_axes[0].grid()

	final_axes[1].plot(MEAN_ABS_ERROR, '-o')
	final_axes[1].set(title = 'Error', xlabel = 'Step', ylabel = 'Mean Absolute Error')
	final_axes[1].grid()

	plt.show()

	print("Finish")