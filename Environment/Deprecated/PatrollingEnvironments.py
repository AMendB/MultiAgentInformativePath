import sys
sys.path.append('.')
import gym
import numpy as np
import matplotlib.pyplot as plt
from Environment.GroundTruthsModels.ShekelGroundTruth import GroundTruth
from Environment.GroundTruthsModels.AlgaeBloomGroundTruth import algae_bloom, algae_colormap, background_colormap
from Environment.Wrappers.time_stacking_wrapper import MultiAgentTimeStackingMemory
from scipy.spatial import distance_matrix
import matplotlib
import json

background_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["sienna","dodgerblue"])

np.random.seed(0)

class DiscreteVehicle:

	def __init__(self, initial_position, n_actions, movement_length, navigation_map, detection_length):
		
		""" Initial positions of the drones """
		np.random.seed(0)
		self.initial_position = initial_position
		self.position = np.copy(initial_position)

		""" Initialize the waypoints """
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)

		""" Detection radius for the contmaination vision """
		self.detection_length = detection_length
		self.navigation_map = navigation_map
		self.detection_mask = self.compute_detection_mask()

		""" Reset other variables """
		self.distance = 0.0
		self.num_of_collisions = 0
		self.action_space = gym.spaces.Discrete(n_actions)
		self.angle_set = np.linspace(0, 2 * np.pi, n_actions, endpoint=False)
		self.movement_length = movement_length

		

	def move(self, action, valid=True):
		""" Move a vehicle in the direction of the action. If valid is False, the action is not performed. """

		angle = self.angle_set[action]
		movement = np.round(np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])).astype(int)
		next_position = self.position + movement
		self.distance += np.linalg.norm(self.position - next_position)

		if self.check_collision(next_position) or not valid:
			collide = True
			self.num_of_collisions += 1
		else:
			collide = False
			self.position = next_position
			self.waypoints = np.vstack((self.waypoints, [self.position]))

		self.detection_mask = self.compute_detection_mask()

		return collide

	def check_collision(self, next_position):

		if self.navigation_map[int(next_position[0]), int(next_position[1])] == 0:
			return True  # There is a collision

		return False

	def compute_detection_mask(self):
		""" Compute the circular mask """

		known_mask = np.zeros_like(self.navigation_map)

		px, py = self.position.astype(int)

		# State - coverage area #
		x = np.arange(0, self.navigation_map.shape[0])
		y = np.arange(0, self.navigation_map.shape[1])

		# Compute the circular mask (area) of the state 3 #
		mask = (x[np.newaxis, :] - px) ** 2 + (y[:, np.newaxis] - py) ** 2 <= self.detection_length ** 2

		known_mask[mask.T] = 1.0

		return known_mask

	def reset(self, initial_position):
		""" Reset the agent - Position, detection mask, etc. """

		self.initial_position = initial_position
		self.position = np.copy(initial_position)
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)
		self.distance = 0.0
		self.num_of_collisions = 0
		self.detection_mask = self.compute_detection_mask()

	def check_action(self, action):
		""" Return True if the action leads to a collision """

		angle = self.angle_set[action]
		movement = np.round(np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])).astype(int)
		next_position = self.position + movement

		return self.check_collision(next_position)

	def move_to_position(self, goal_position):
		""" Move to the given position """

		assert self.navigation_map[goal_position[0], goal_position[1]] == 1, "Invalid position to move"
		self.distance += np.linalg.norm(goal_position - self.position)
		""" Update the position """
		self.position = goal_position

class DiscreteFleet:

	def __init__(self,
				 number_of_vehicles,
				 n_actions,
				 fleet_initial_positions,
				 movement_length,
				 detection_length,
				 navigation_map,
				 max_connection_distance=10,
				 optimal_connection_distance=5):

		""" Coordinator of the movements of the fleet. Coordinates the common model, the distance between drones, etc. """
		np.random.seed(0)
		self.number_of_vehicles = number_of_vehicles
		self.initial_positions = fleet_initial_positions
		self.n_actions = n_actions
		self.movement_length = movement_length
		self.detection_length = detection_length

		""" Create the vehicles object array """
		self.vehicles = [DiscreteVehicle(initial_position=fleet_initial_positions[k],
										 n_actions=n_actions,
										 movement_length=movement_length,
										 navigation_map=navigation_map,
										 detection_length=detection_length) for k in range(self.number_of_vehicles)]

		self.agent_positions = np.asarray([veh.position for veh in self.vehicles])

		# Get the redundancy mask #
		self.redundancy_mask = np.sum([veh.detection_mask for veh in self.vehicles], axis=0)
		# Get the collective detection mask #
		self.collective_mask = self.redundancy_mask.astype(bool)
		self.historic_visited_mask = self.redundancy_mask.astype(bool)
		# Reset model variables 
		self.measured_values = None
		self.measured_locations = None

		# Reset fleet-communication-restriction variables #
		self.max_connection_distance = max_connection_distance
		self.isolated_mask = None
		self.fleet_collisions = 0
		self.danger_of_isolation = None
		self.distance_between_agents = None
		self.optimal_connection_distance = optimal_connection_distance
		self.number_of_disconnections = 0

	@staticmethod
	def majority(arr: np.ndarray) -> bool:
		return arr.sum() >= len(arr) // 2

	def check_fleet_collision_within(self, veh_actions):
		""" Check if there is any collision between agents """
		
		new_positions = []

		for idx, veh_action in veh_actions.items():

			angle = self.vehicles[idx].angle_set[veh_action]
			movement = np.round(np.array([self.vehicles[idx].movement_length * np.cos(angle), self.vehicles[idx].movement_length * np.sin(angle)])).astype(int)
			new_positions.append(list(self.vehicles[idx].position + movement))

		_, inverse_index, counts = np.unique(np.asarray(new_positions), return_inverse=True, return_counts=True, axis=0)

		# True if repeated #
		not_collision_within = counts[inverse_index] == 1

		return not_collision_within

	def move(self, fleet_actions):

		# Check if there are collisions between vehicles #
		self_colliding_mask = self.check_fleet_collision_within(fleet_actions)
		# Process the fleet actions and move the vehicles #
		collision_array = {k: self.vehicles[k].move(fleet_actions[k], valid=valid) for k, valid in zip(list(fleet_actions.keys()), self_colliding_mask)}
		# Update vector with agent positions #
		self.agent_positions = np.asarray([veh.position for veh in self.vehicles])
		# Sum up the collisions for termination #
		self.fleet_collisions = np.sum([self.vehicles[k].num_of_collisions for k in range(self.number_of_vehicles)])
		# Compute the redundancy mask #
		self.redundancy_mask = np.sum([veh.detection_mask for veh in self.vehicles], axis=0)
		# Update the collective mask #
		self.collective_mask = self.redundancy_mask.astype(bool)
		# Update the historic visited mask #
		self.historic_visited_mask = np.logical_or(self.historic_visited_mask, self.collective_mask)
		# Update the isolation mask (for networked agents) #
		self.update_isolated_mask()

		return collision_array

	def update_isolated_mask(self):
		""" Compute the mask of isolated vehicles. Only for restricted fleets. """

		# Get the distance matrix #
		distance = self.get_distance_matrix()
		# Delete the diagonal (self-distance, always 0) #
		self.distance_between_agents = distance[~np.eye(distance.shape[0], dtype=bool)].reshape(distance.shape[0], -1)
		# True if all agents are further from the danger distance
		danger_of_isolation_mask = self.distance_between_agents > self.optimal_connection_distance
		self.danger_of_isolation = np.asarray([self.majority(value) for value in danger_of_isolation_mask])
		# True if all agents are further from the max connection distance
		isolation_mask = self.distance_between_agents > self.max_connection_distance
		self.isolated_mask = np.asarray([self.majority(value) for value in isolation_mask])
		self.number_of_disconnections += np.sum(self.isolated_mask)

	def measure(self, gt_field):

		"""
		Take a measurement in the given N positions
		:param gt_field:
		:return: An numpy array with dims (N,2)
		"""
		positions = np.array([self.vehicles[k].position for k in range(self.number_of_vehicles)])

		values = []
		for pos in positions:
			values.append([gt_field[int(pos[0]), int(pos[1])]])

		if self.measured_locations is None:
			self.measured_locations = positions
			self.measured_values = values
		else:
			self.measured_locations = np.vstack((self.measured_locations, positions))
			self.measured_values = np.vstack((self.measured_values, values))

		return self.measured_values, self.measured_locations

	def reset(self, initial_positions=None):
		""" Reset the fleet """

		if initial_positions is None:
			initial_positions = self.initial_positions

		for k in range(self.number_of_vehicles):
			self.vehicles[k].reset(initial_position=initial_positions[k])

		self.agent_positions = np.asarray([veh.position for veh in self.vehicles])
		self.measured_values = None
		self.measured_locations = None
		self.fleet_collisions = 0
		self.number_of_disconnections = 0

		# Get the redundancy mask #
		self.redundancy_mask = np.sum([veh.detection_mask for veh in self.vehicles], axis=0)
		# Get the collective detection mask #
		self.collective_mask = self.redundancy_mask.astype(bool)
		self.historic_visited_mask = self.redundancy_mask.astype(bool)

		self.update_isolated_mask()

	def get_distances(self):
		return [self.vehicles[k].distance for k in range(self.number_of_vehicles)]

	def check_collisions(self, test_actions):
		""" Array of bools (True if collision) """
		return [self.vehicles[k].check_action(test_actions[k]) for k in range(self.number_of_vehicles)]

	def move_fleet_to_positions(self, goal_list):
		""" Move the fleet to the given positions.
		 All goal positions must ve valid. """

		goal_list = np.atleast_2d(goal_list)

		for k in range(self.number_of_vehicles):
			self.vehicles[k].move_to_position(goal_position=goal_list[k])

	def get_distance_matrix(self):
		return distance_matrix(self.agent_positions, self.agent_positions)

	def get_positions(self):

		return np.asarray([veh.position for veh in self.vehicles])


class MultiAgentPatrolling(gym.Env):

	def __init__(self, scenario_map,
				 distance_budget,
				 number_of_vehicles,
				 fleet_initial_positions=None,
				 seed=0,
				 miopic=True,
				 detection_length=2,
				 movement_length=2,
				 max_collisions=5,
				 forget_factor=1.0,
				 networked_agents=False,
				 max_connection_distance=10,
				 optimal_connection_distance=5,
				 max_number_of_disconnections=10,
				 attrittion=0.0,
				 obstacles=False,
				 hard_penalization=False,
				 reward_type='weighted_idleness',
				 reward_weights = (10.0, 1.0),
				 ground_truth_type='algae_bloom',
				 frame_stacking = 0,
				 dynamic=False,
				 state_index_stacking = (0,1,2,3,4)):

		""" The gym environment """

		# Load the scenario map
		np.random.seed(seed)
		self.scenario_map = scenario_map
		self.visitable_locations = np.vstack(np.where(self.scenario_map != 0)).T
		self.number_of_agents = number_of_vehicles
		self.dynamic = dynamic

		# Initial positions
		if fleet_initial_positions is None:
			self.random_inititial_positions = True
			random_positions_indx = np.random.choice(np.arange(0, len(self.visitable_locations)), number_of_vehicles, replace=False)
			self.initial_positions = self.visitable_locations[random_positions_indx]
		else:
			self.random_inititial_positions = False
			self.initial_positions = fleet_initial_positions

		self.obstacles = obstacles
		self.miopic = miopic
		self.reward_type = reward_type
	
		# Number of pixels
		self.distance_budget = distance_budget
		self.max_number_of_movements = distance_budget // detection_length
		# Number of agents
		self.seed = seed
		# Detection radius
		self.detection_length = detection_length
		self.forget_factor = forget_factor
		self.attrition = attrittion
		# Fleet of N vehicles
		self.optimal_connection_distance = optimal_connection_distance
		self.max_connection_distance = max_connection_distance
		self.movement_length = movement_length
		self.reward_weights = reward_weights
		
		# Create the fleets 
		self.fleet = DiscreteFleet(number_of_vehicles=self.number_of_agents,
								   n_actions=8,
								   fleet_initial_positions=self.initial_positions,
								   movement_length=movement_length,
								   detection_length=detection_length,
								   navigation_map=self.scenario_map,
								   max_connection_distance=self.max_connection_distance,
								   optimal_connection_distance=self.optimal_connection_distance)

		self.max_collisions = max_collisions
		self.ground_truth_type = ground_truth_type
		if ground_truth_type == 'shekel':
			self.gt = GroundTruth(self.scenario_map, max_number_of_peaks=4, is_bounded=True, seed=self.seed)
		elif ground_truth_type == 'algae_bloom':
			self.gt = algae_bloom(self.scenario_map, seed=self.seed)
		else:
			raise NotImplementedError("This Benchmark is not implemented. Choose one that is.")
		

		""" Model attributes """
		self.actual_known_map = None
		self.idleness_matrix = None
		self.importance_matrix = None
		self.model = None
		self.inside_obstacles_map = None
		self.state = None
		self.fig = None

		self.action_space = gym.spaces.Discrete(8)

		assert frame_stacking >= 0, "frame_stacking must be >= 0"
		self.frame_stacking = frame_stacking
		self.state_index_stacking = state_index_stacking

		if frame_stacking != 0:
			self.frame_stacking = MultiAgentTimeStackingMemory(n_agents = self.number_of_agents,
			 													n_timesteps = frame_stacking - 1, 
																state_indexes = state_index_stacking, 
																n_channels = 5)
			self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5 + len(state_index_stacking)*(frame_stacking - 1), *self.scenario_map.shape), dtype=np.float32)

		else:
			self.frame_stacking = None
			self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5, *self.scenario_map.shape), dtype=np.float32)

		self.state_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4, *self.scenario_map.shape), dtype=np.float32)

		
		self.individual_action_state = gym.spaces.Discrete(8)

		self.networked_agents = networked_agents
		self.hard_networked_penalization = hard_penalization
		self.number_of_disconnections = 0
		self.max_number_of_disconnections = max_number_of_disconnections

		self.reward_normalization_value = self.fleet.vehicles[0].detection_mask

	def reset(self):
		""" Reset the environment """

		# Reset the ground truth #
		self.gt.reset()
		self.importance_matrix = self.gt.read()
		# Create an empty model #
		self.model = np.zeros_like(self.scenario_map) if self.miopic else self.importance_matrix
		self.model_ant = self.model.copy()

		# Get the N random initial positions #
		if self.random_inititial_positions:
			random_positions_indx = np.random.choice(np.arange(0, len(self.visitable_locations)), self.number_of_agents, replace=False)
			self.initial_positions = self.visitable_locations[random_positions_indx]

		# Reset the positions of the fleet #
		self.fleet.reset(initial_positions=self.initial_positions)
		self.active_agents = {agent_id: True for agent_id in range(self.number_of_agents)}

		# New idleness mask (1-> high idleness, 0-> just visited)
		self.idleness_matrix = 1 - np.copy(self.fleet.collective_mask)

		# Randomly generated obstacles #
		if self.obstacles:
			# Generate a inside obstacles map #
			self.inside_obstacles_map = np.zeros_like(self.scenario_map)
			obstacles_pos_indx = np.random.choice(np.arange(0, len(self.visitable_locations)), size=20, replace=False)
			self.inside_obstacles_map[self.visitable_locations[obstacles_pos_indx, 0], self.visitable_locations[obstacles_pos_indx, 1]] = 1.0

			# Update the obstacle map for every agent #
			for i in range(self.number_of_agents):
				self.fleet.vehicles[i].navigation_map = self.scenario_map - self.inside_obstacles_map

		# Update the state of the agents #
		self.update_state()

		return self.state if self.frame_stacking is None else self.frame_stacking.process(self.state)

	def update_temporal_mask(self):

		self.idleness_matrix = self.idleness_matrix + 1.0 / (self.forget_factor * self.max_number_of_movements)
		self.idleness_matrix = self.idleness_matrix - self.fleet.collective_mask
		self.idleness_matrix = np.clip(self.idleness_matrix, 0, 1)

		return self.idleness_matrix

	def update_information_importance(self):
		""" Applied the attrition term """
		self.importance_matrix = np.clip(
			self.importance_matrix - self.attrition * self.gt.read() * self.fleet.collective_mask, 0, 999999)

	def update_state(self):
		""" Update the state for every vehicle """

		state = {}

		# Channel 1 -> Known boundaries
		if self.obstacles:
			obstacle_map = self.scenario_map - np.logical_and(self.inside_obstacles_map, self.fleet.historic_visited_mask)
		else:
			obstacle_map = self.scenario_map

		# Channel 2 -> Known information
		# state[2] = self.importance_matrix * self.fleet.historic_visited_mask if self.miopic else self.importance_matrix
		if self.miopic:
			known_information = -np.ones_like(self.model)
			known_information[np.where(self.fleet.historic_visited_mask)] = self.model[np.where(self.fleet.historic_visited_mask)]
		else:
			known_information = self.gt.read()

		# Create fleet position #
		fleet_position_map = np.zeros_like(self.scenario_map)
		fleet_position_map[self.fleet.agent_positions[:,0], self.fleet.agent_positions[:,1]] = 1.0

		# Channel 3 and 4
		for i in range(self.number_of_agents):
			
			agent_observation_of_fleet = fleet_position_map.copy()
			agent_observation_of_fleet[self.fleet.agent_positions[i,0], self.fleet.agent_positions[i,1]] = 0.0

			agent_observation_of_position = np.zeros_like(self.scenario_map)
			agent_observation_of_position[self.fleet.agent_positions[i,0], self.fleet.agent_positions[i,1]] = 1.0
			
			state[i] = np.concatenate((
				obstacle_map[np.newaxis],
				self.idleness_matrix[np.newaxis],
				known_information[np.newaxis],
				agent_observation_of_fleet[np.newaxis],
				agent_observation_of_position[np.newaxis]
			))

		self.state = {agent_id: state[agent_id] for agent_id in range(self.number_of_agents) if self.active_agents[agent_id]}

	def step(self, action: dict):

		# Process action movement only for active agents #
		action = {action_id: action[action_id] for action_id in range(self.number_of_agents) if self.active_agents[action_id]}
		collision_mask = self.fleet.move(action)

		# Update model #
		if self.miopic:
			self.update_model()
		else:
			self.model = self.gt.read()

		# Compute reward
		reward = self.reward_function(collision_mask, action)

		# Update idleness and attrition
		self.update_temporal_mask()
		self.update_information_importance()

		# Update state
		self.update_state()

		# Final condition #
		done = {agent_id: self.fleet.get_distances()[agent_id] > self.distance_budget or self.fleet.fleet_collisions > self.max_collisions for agent_id in range(self.number_of_agents)}
		self.active_agents = [not d for d in done.values()]

		if self.networked_agents:

			if self.fleet.number_of_disconnections > self.max_number_of_disconnections and self.hard_networked_penalization:
				
				for key in done.keys():
					done[key] = True

		
		# Update ground truth
		if self.dynamic:
			self.gt.step()

		return self.state if self.frame_stacking is None else self.frame_stacking.process(self.state), reward, done, self.info

	def update_model(self):
		""" Update the model using the new positions """

		self.model_ant = self.model.copy()

		gt_ = self.gt.read()
		for vehicle in self.fleet.vehicles:
			self.model[vehicle.detection_mask.astype(bool)] = gt_[vehicle.detection_mask.astype(bool)]

	def render(self, mode='human'):

		import matplotlib.pyplot as plt

		agente_disponible = np.argmax(self.active_agents)

		if not any(self.active_agents):
			return

		if self.fig is None:

			self.fig, self.axs = plt.subplots(1, 6, figsize=(15,5))
			
			# Print the obstacles map
			self.im0 = self.axs[0].imshow(self.state[agente_disponible][0], cmap = background_colormap)
			self.axs[0].set_title('Navigation map')
			# Print the idleness map
			self.im1 = self.axs[1].imshow(self.state[agente_disponible][1],  cmap = 'rainbow_r')
			self.axs[1].set_title('Idleness map (W)')

			# Create a background for unknown places #
			known = np.zeros_like(self.scenario_map) + 0.25
			known[1::2, ::2] = 0.5
			known[::2, 1::2] = 0.5
			self.im2_known = self.axs[2].imshow(known, cmap='gray', vmin=0, vmax=1)

			# Synthesize the model with the background
			model = known*np.nan
			model[np.where(self.fleet.historic_visited_mask)] = self.model[np.where(self.fleet.historic_visited_mask)]

			# Print model (I)  #
			self.im2 = self.axs[2].imshow(model,  cmap=algae_colormap, vmin=0.0, vmax=1.0)
			self.axs[2].set_title("Model / Importance (I)")

			# Print the real GT
			self.im3_known = self.axs[3].imshow(known, cmap='gray', vmin=0, vmax=1)
			real_gt = known*np.nan
			real_gt[self.visitable_locations[:,0], self.visitable_locations[:,1]] = self.gt.read()[self.visitable_locations[:,0], self.visitable_locations[:,1]]
			self.im3 = self.axs[3].imshow(real_gt,  cmap=algae_colormap, vmin=0.0, vmax=1.0)
			self.axs[3].set_title("Real importance GT")

			# Agent 0 position #
			self.im4 = self.axs[4].imshow(self.state[agente_disponible][3], cmap = 'gray')
			self.axs[4].set_title("Agent 0 position")

			# Others-than-Agent 0 position #
			self.im5 = self.axs[5].imshow(self.state[agente_disponible][4], cmap = 'gray')
			self.axs[5].set_title("Others agents position")

		self.im0.set_data(self.state[agente_disponible][0])
		self.im1.set_data(self.state[agente_disponible][1])

		known = np.zeros_like(self.scenario_map)*np.nan
		known[np.where(self.fleet.historic_visited_mask == 1)] = self.state[agente_disponible][2][np.where(self.fleet.historic_visited_mask == 1)]
		self.im2.set_data(known)

		self.im4.set_data(self.state[agente_disponible][3])
		real_gt = known*np.nan
		real_gt[self.visitable_locations[:,0], self.visitable_locations[:,1]] = self.gt.read()[self.visitable_locations[:,0], self.visitable_locations[:,1]]
		self.im3.set_data(real_gt)
		self.im5.set_data(self.state[agente_disponible][4])

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()

		plt.draw()

		plt.pause(0.01)

	def reward_function(self, collision_mask, actions):
		""" Reward function

			1) weighted_idleness:
			r(t) = Sum(I(m)*W(m)/Dr(m)) - Pc - Pn
			2) model_changes:
			r(t) = Sum(W(m)/Dr(m)) + |I_t-1(m) - It(m)|/Dr(m)
		"""

		if self.reward_type == 'weighted_idleness':
			rewards = np.array(
				[np.sum(self.importance_matrix[veh.detection_mask.astype(bool)] * self.idleness_matrix[
					veh.detection_mask.astype(bool)] / (1 * self.detection_length * self.fleet.redundancy_mask[
					veh.detection_mask.astype(bool)])) for veh in self.fleet.vehicles]
			)
		elif self.reward_type == 'model_changes':
			
			changes_in_model = np.abs(self.model - self.model_ant)
			
			changes = np.array(
				[np.sum(
					changes_in_model[veh.detection_mask.astype(bool)] / self.fleet.redundancy_mask[veh.detection_mask.astype(bool)]
					) for veh in self.fleet.vehicles
					]
				)

			idleness = np.array(
				[np.sum(
					self.idleness_matrix[veh.detection_mask.astype(bool)] / self.fleet.redundancy_mask[veh.detection_mask.astype(bool)]
					) for veh in self.fleet.vehicles
					]
				) 

			rewards = self.reward_weights[1] * changes + self.reward_weights[0]*idleness

		self.info = {}

		cost = {agent_id: 1 if action % 2 == 0 else np.sqrt(2) for agent_id, action in actions.items()}
		rewards = {agent_id: rewards[agent_id]/cost[agent_id] if not collision_mask[agent_id] else -1.0 for agent_id in actions.keys()}

		if self.networked_agents:
			# For those agents that are too separated from the others (in danger of disconnection)
			min_distances = np.min(self.fleet.distance_between_agents[self.fleet.danger_of_isolation],
								   axis=1) - self.fleet.optimal_connection_distance
			# Apply a penalization from 0 to -1 depending on the exceeding distance from the optimal

			rewards[self.fleet.danger_of_isolation] -= np.clip(min_distances / (self.max_connection_distance - self.optimal_connection_distance), 0, 1)

			rewards[self.fleet.isolated_mask] = -1.0

		return {agent_id: rewards[agent_id] for agent_id in range(self.number_of_agents) if self.active_agents[agent_id]}

	def get_action_mask(self, ind=0):
		""" Return an array of Bools (True means this action for the agent ind causes a collision) """

		assert 0 <= ind < self.number_of_agents, 'Not enough agents!'

		return np.array(list(map(self.fleet.vehicles[ind].check_action, np.arange(0, 8))))
	
	def start_recording(self, trajectory_length=50):
		""" Start recording the environment trajectories """

		self.recording = True
		self.recording_frames = []

	def stop_recording(self, path, episode):
		""" Stop recording the environment trajectories. Save the data in the current directory. """

		self.recording = False
		self.recording_frames = np.array(self.recording_frames).astype(np.float16)
		# Save the data in the current directory with the given path and name
		np.save(path + f'_{episode}', self.recording_frames)

	def save_environment_configuration(self, path):
		""" Save the environment configuration in the current directory as a json file"""

		environment_configuration = {

			'number_of_agents': self.number_of_agents,
			'miopic': self.miopic,
			'fleet_initial_positions': self.initial_positions.tolist(),
			'distance_budget': self.distance_budget,
			'detection_length': self.detection_length,
			'max_number_of_movements': self.max_number_of_movements,
			'forgetting_factor': self.forget_factor,
			'attrition': self.attrition,
			'reward_type': self.reward_type,
			'networked_agents': self.networked_agents,
			'optimal_connection_distance': self.optimal_connection_distance,
			'max_connection_distance': self.max_connection_distance,
			'ground_truth': self.ground_truth_type,
			'reward_weights': self.reward_weights
		}

		with open(path + '/environment_config.json', 'w') as f:
			json.dump(environment_configuration, f)



if __name__ == '__main__':


	sc_map = np.genfromtxt('Environment/Maps/example_map.csv', delimiter=',')

	N = 4
	initial_positions = np.array([[30, 20], [32, 20], [34, 20], [30, 22]])[:N, :]
	visitable = np.column_stack(np.where(sc_map == 1))
	#initial_positions = visitable[np.random.randint(0,len(visitable), size=N), :]
	

	env = MultiAgentPatrolling(scenario_map=sc_map,
							   fleet_initial_positions=initial_positions,
							   distance_budget=250,
							   number_of_vehicles=N,
							   seed=0,
							   miopic=True,
							   detection_length=2,
							   movement_length=1,
							   max_collisions=500,
							   forget_factor=0.5,
							   attrittion=0.1,
							   networked_agents=False,
							   reward_type='model_changes',
							   ground_truth_type='shekel',
							   obstacles=True,
							   frame_stacking=1,
							   state_index_stacking=(2,3,4),
							   reward_weights=(1.0, 0.1)
							 )

	env.reset()

	done = {i:False for i in range(4)}

	R = []

	action = {i: np.random.randint(0,8) for i in range(N)}

	while not any(list(done.values())):

		for idx, agent in enumerate(env.fleet.vehicles):
		
			agent_mask = np.array([agent.check_action(a) for a in range(8)], dtype=int)

			if agent_mask[action[idx]]:
				action[idx] = np.random.choice(np.arange(8), p=(1-agent_mask)/np.sum((1-agent_mask)))


		s, r, done, _ = env.step(action)

		env.render()

		R.append(list(r.values()))

		print(r)


	env.render()
	plt.show()

	plt.plot(np.cumsum(np.asarray(R),axis=0), '-o')
	plt.xlabel('Step')
	plt.ylabel('Individual Reward')
	plt.legend([f'Agent {i}' for i in range(N)])
	plt.grid()
	plt.show()
