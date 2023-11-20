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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

background_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["sienna","dodgerblue"])

np.random.seed(0)

class DiscreteVehicle: # clase para crear vehículos individuales

	def __init__(self, initial_position, n_actions, movement_length, navigation_map):
		
		""" Initial positions of the drones """
		np.random.seed(0) # semilla para que siempre salga igual
		self.initial_position = initial_position # asigna posición inicial
		self.actual_agent_position = np.copy(initial_position) # copia la posición inicial a la actual

		""" Initialize the waypoints """
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)
		self.navigation_map = navigation_map # cargo el mapa desde el archivo

		""" Reset other variables """
		self.distance_traveled = 0.0 # sumador de distancia recorrida
		self.num_of_collisions = 0 # contador de colisiones
		self.action_space = gym.spaces.Discrete(n_actions)
		self.angle_set = np.linspace(0, 2 * np.pi, n_actions, endpoint=False) #array con los 8 puntos cardinales en RADIANES, divido en 8 direcciones una circunferencia: [0. , 0.78539816, 1.57079633, 2.35619449, 3.14159265, 3.92699082, 4.71238898, 5.49778714]
		self.movement_length = movement_length 
		

	def move_agent(self, action, valid=True):
		""" Move a vehicle in the direction of the action. If valid is False, the action is not performed. """

		next_position = self.calculate_next_position(action)
		self.distance_traveled += np.linalg.norm(self.actual_agent_position - next_position) # sumo la distancia que recorro con el movimiento a la distancia total recorrida

		if self.check_collision_with_obstacle(next_position) or not valid: # si la siguiente posición es colisión (con tierra o colisión entre drones):
			collide = True
			self.num_of_collisions += 1 # sumo una colisión al contador, pero no se mueve el vehículo
		else:
			collide = False
			self.actual_agent_position = next_position # asigno la posición siguiente a la actual
			self.waypoints = np.vstack((self.waypoints, [self.actual_agent_position])) #añado la posición actual al vector de puntos visitados (hace una concatenación de los dos vectores, el de todos los puntos y el array convertido a lista que contiene el punto actual)

		return collide # returns if the agent collide
	
	def calculate_next_position(self, action):
		angle = self.angle_set[action] # tomo como ángulo de movimiento el asociado a la acción tomada, la acción sirve como índice del array de puntos cardinales
		movement = np.round(np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])).astype(int) #convierto el ángulo en movimiento cartesiano (cuántas celdas se mueve en eje x y cuántas eje y)
		next_position = self.actual_agent_position + movement # posición siguiente, le sumo el movimiento a la actual

		return next_position
		
	def check_collision_with_obstacle(self, next_position): #función para comprobar si en la siguiente posición hay un obstáculo: devuelve true o false
		""" Return True if the next position leads to a collision """

		if self.navigation_map[int(next_position[0]), int(next_position[1])] == 0: # Si la posición es cero en el mapa (hay obstáculo)
			return True  # There is a collision

		return False
	
	def check_action(self, action):
		""" Return True if the action leads to a collision """
		
		# Calcula next_position:
		next_position = self.calculate_next_position(action)

		# Devuelve la comprobación de si la siguiente posición es colisión, llamando a la función check_collision_with_obstacle():
		return self.check_collision_with_obstacle(next_position) 
	
	def reset_agent(self, initial_position):
		""" Reset the agent - Position, detection mask, etc. """

		self.initial_position = initial_position
		self.actual_agent_position = np.copy(initial_position)
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)
		self.distance_traveled = 0.0
		self.num_of_collisions = 0


class DiscreteFleet: # clase para crear FLOTAS de la clase de vehículos

	def __init__(self,
				 number_of_vehicles,
				 n_actions,
				 fleet_initial_positions,
				 movement_length,
				 navigation_map,
				 check_collisions_within):

		# """ Coordinator of the movements of the fleet. Coordinates the common model, the distance between drones, etc. """
		np.random.seed(0)
		self.number_of_vehicles = number_of_vehicles # número de vehículos
		self.initial_positions = fleet_initial_positions # posición inicial de la flota
		self.n_actions = n_actions # número de acciones 
		self.movement_length = movement_length # longitud del movimiento
		self.check_collisions_within = check_collisions_within


		""" Create the vehicles object array """
		self.vehicles = [DiscreteVehicle(initial_position=fleet_initial_positions[k],
										 n_actions=n_actions,
										 movement_length=movement_length,
										 navigation_map=navigation_map) for k in range(self.number_of_vehicles)] #list comprehension, crea tantos vehículos como marque number_of_vehicles

		self.agent_positions = np.asarray([veh.actual_agent_position for veh in self.vehicles]) # guarda la posición de todos los vehículos en un array

		# Reset fleet-collisions-restriction variable #
		self.fleet_collisions = 0

	@staticmethod
	def majority(arr: np.ndarray) -> bool:
		return arr.sum() >= len(arr) // 2 # this function checks if the sum of the elements in the array is greater than or equal 
										#to half the length of the array, which can be interpreted as checking if the majority of the elements in the array are "True" or "non-zero".
	def check_collision_within_fleet(self, veh_actions):
		""" Check if there is any collision between agents. Returns boolean array with True to the vehicles with unique new position, i.e., valid actions. """
		
		new_positions = []

		for idx, veh_action in veh_actions.items():
			# Calculate next positions #
			angle = self.vehicles[idx].angle_set[veh_action]
			movement = np.round(np.array([self.vehicles[idx].movement_length * np.cos(angle), self.vehicles[idx].movement_length * np.sin(angle)])).astype(int)
			new_positions.append(list(self.vehicles[idx].actual_agent_position + movement))

		_, inverse_index, counts = np.unique(np.asarray(new_positions), return_inverse=True, return_counts=True, axis=0) # check if unique

		# True if NOT repeated #
		valid_actions_within_fleet = counts[inverse_index] == 1 

		return valid_actions_within_fleet
	
	def reset_fleet(self, initial_positions=None):
		""" Reset the fleet """

		if initial_positions is None:
			initial_positions = self.initial_positions

		# Assign initial positions to each agent #
		for k in range(self.number_of_vehicles):
			self.vehicles[k].reset_agent(initial_position=initial_positions[k])

		self.agent_positions = np.asarray([veh.actual_agent_position for veh in self.vehicles])
		self.fleet_collisions = 0

	def move_fleet(self, fleet_actions):

		if self.check_collisions_within:
			# Check if there are collisions between vehicles #
			valid_actions_within_fleet_mask = self.check_collision_within_fleet(fleet_actions)
			# Process the fleet actions and move the vehicles # 
			collision_array = {k: self.vehicles[k].move_agent(fleet_actions[k], valid=valid) for k, valid in zip(list(fleet_actions.keys()), valid_actions_within_fleet_mask)}
		else: 
			collision_array = {k: self.vehicles[k].move_agent(fleet_actions[k], valid=True) for k in fleet_actions.keys()}

		# Update vector with agent positions #
		self.agent_positions = np.asarray([veh.actual_agent_position for veh in self.vehicles])
		# Sum up the collisions for termination #
		self.fleet_collisions = np.sum([self.vehicles[k].num_of_collisions for k in range(self.number_of_vehicles)]) # suma de todas las colisiones de la flota

		return collision_array # devuelve el diccionario donde se indica el número de colisiones de cada vehículo
		

	def get_agents_distances_traveled(self):
		return [self.vehicles[k].distance_traveled for k in range(self.number_of_vehicles)] # devuelve una lista con las distancias recorridas por cada vehículo

	def get_fleet_positions(self):

		return np.array([veh.actual_agent_position for veh in self.vehicles])

class MultiAgentPatrolling(gym.Env):

	def __init__(self, 
	      		 scenario_map,
				 number_of_vehicles,
				 max_distance_traveled,
				 mean_sensormeasure,
				 std_sensormeasure,
				 variance_sensormeasure,
				 fleet_initial_positions=None,
				 seed=0,
				 movement_length=2,
				 check_collisions_within=False,
				 max_collisions=5,
				 reward_type='weighted_idleness', 
				 ground_truth_type='algae_bloom', 
				 obstacles=False,
				 frame_stacking = 0, 
				 dynamic=False, 
				 state_index_stacking = (0,1,2,3,4), ##
				 reward_weights = (10.0, 1.0),
				 hard_penalization=False ##
				 ):

		""" The gym environment """

		# Load the scenario map
		np.random.seed(seed)
		self.scenario_map = scenario_map
		self.number_of_agents = number_of_vehicles
		self.seed = seed
		self.dynamic = dynamic
		self.obstacles = obstacles
		self.reward_type = reward_type
		self.visitable_locations = np.vstack(np.where(self.scenario_map != 0)).T # coords visitable cells

		# Initial positions #
		if fleet_initial_positions is None:
			self.random_inititial_positions = True
			random_positions_indx = np.random.choice(np.arange(0, len(self.visitable_locations)), number_of_vehicles, replace=False) # se selecciona un índice aleatorio como máximo el número de celdas visitables
			self.initial_positions = self.visitable_locations[random_positions_indx] # con ese índice aleatorio se selecciona la posición inicial aleatoria dentro de las celdas visitables
		else:
			self.random_inititial_positions = False
			self.initial_positions = fleet_initial_positions
	
		# Limits to be declared a death/done agent #
		self.max_distance_traveled = max_distance_traveled
		self.max_collisions = max_collisions

		# Fleet of N vehicles #
		self.movement_length = movement_length
		self.reward_weights = reward_weights
		
		# Create the fleets #
		self.fleet = DiscreteFleet(number_of_vehicles=self.number_of_agents,
								   n_actions=8,
								   fleet_initial_positions=self.initial_positions,
								   movement_length=movement_length,
								   navigation_map=self.scenario_map,
								   check_collisions_within=check_collisions_within)

		# Generate Ground Truth #
		self.ground_truth_type = ground_truth_type
		if ground_truth_type == 'shekel':
			self.ground_truth = GroundTruth(self.scenario_map, max_number_of_peaks=4, is_bounded=True, seed=self.seed)
		elif ground_truth_type == 'algae_bloom':
			self.ground_truth = algae_bloom(self.scenario_map, seed=self.seed)
		else:
			raise NotImplementedError("This Benchmark is not implemented. Choose one that is.")

		# Model maps #
		self.model_mean_map = None
		self.model_uncertainty_map = None

		# Load sensors noise #
		self.mean_sensormeasure = mean_sensormeasure
		self.std_sensormeasure = std_sensormeasure
		self.variance_sensormeasure = variance_sensormeasure

		# Create Gaussian Process #
		self.gaussian_process = GaussianProcess(scenario_map=sc_map)

		# Visualization #
		self.photographs = None
		self.fig = None

		# self.reward_normalization_value = self.fleet.vehicles[0].detection_mask


	def reset_env(self):
		""" Reset the environment """

		# Reset the ground truth #
		self.ground_truth.reset()
		self.importance_matrix = self.ground_truth.read()

		# Create an empty model #
		self.model_mean_map = np.zeros_like(self.scenario_map) 
		self.model_uncertainty_map = np.zeros_like(self.scenario_map) 
		# self.model_ant = self.model.copy()

		# Get the N random initial positions #
		if self.random_inititial_positions:
			random_positions_indx = np.random.choice(np.arange(0, len(self.visitable_locations)), self.number_of_agents, replace=False)
			self.initial_positions = self.visitable_locations[random_positions_indx]

		# Reset the positions of the fleet #
		self.fleet.reset_fleet(initial_positions=self.initial_positions)
		self.active_agents = {agent_id: True for agent_id in range(self.number_of_agents)}

		# Randomly generated obstacles #
		if self.obstacles:
			self.inside_obstacles_map = np.zeros_like(self.scenario_map)
			# Generate a random inside obstacles map #
			obstacles_pos_indx = np.random.choice(np.arange(0, len(self.visitable_locations)), size=20, replace=False)
			self.inside_obstacles_map[self.visitable_locations[obstacles_pos_indx, 0], self.visitable_locations[obstacles_pos_indx, 1]] = 1.0

			# Update the obstacle map for every agent #
			for i in range(self.number_of_agents):
				self.fleet.vehicles[i].navigation_map = self.scenario_map - self.inside_obstacles_map
		else:
			self.inside_obstacles_map = np.zeros_like(self.scenario_map)

		# Update the photograph/state of the agents #
		self.update_photographs()

		return self.photographs #if self.frame_stacking is None else self.frame_stacking.process(self.photographs)

	def take_samples(self):
		""" The active agents take a noisy sample from the ground truth """
		
		# Get the Ground Truth #
		ground_truth = self.ground_truth.read() 

		# Save positions where samples are taken #
		position_measures = [[agent.actual_agent_position[0], agent.actual_agent_position[1]] for idx, agent in enumerate(self.fleet.vehicles) if self.active_agents[idx]]
		
		# Take the sample and add noise #
		noisy_measures = [ground_truth[pose_x, pose_y] + np.random.normal(mean, std) for (pose_x, pose_y), mean, std in zip(position_measures, self.mean_sensormeasure, self.std_sensormeasure)]

		# Delete death drones alphas
		variance_measures = np.array([self.variance_sensormeasure[idx] for idx in self.active_agents if self.active_agents[idx]]) 

		return position_measures, noisy_measures, variance_measures

	def step_env(self, actions: dict):
		"""Ejecuta todas las actualizaciones para cada step"""

		# Sensor samples #
		position_new_measures, new_measures, variance_measures = self.take_samples()
		print("\nNuevo:") # UNDER TEST
		print(position_new_measures) # UNDER TEST
		print(new_measures) # UNDER TEST
		print(variance_measures) # UNDER TEST
		
		# Fit gaussian process with new samples #
		self.gaussian_process.fit_gp(X_new=position_new_measures, y_new=new_measures, variances_new=variance_measures)
		
		# Update model: prediction of ground truth #
		self.model_mean_map, self.model_uncertainty_map = self.gaussian_process.predict_gt()

		# Process actions movement only for active agents #
		actions = {action_id: actions[action_id] for action_id in range(self.number_of_agents) if self.active_agents[action_id]}
		collision_mask = self.fleet.move_fleet(actions) # returns collision_mask dict

		# # Compute reward
		# reward = self.reward_function(collision_mask, action)

		# Update the photograph/state of the agents #
		self.update_photographs()

		# Final condition #
		print("Nº colisiones:" + str(self.fleet.fleet_collisions))
		done = {agent_id: (self.fleet.get_agents_distances_traveled()[agent_id] > self.max_distance_traveled or self.fleet.fleet_collisions > self.max_collisions) for agent_id in range(self.number_of_agents)}
		done[2] = True # UNDER TEST
		done[1] = True # UNDER TEST
		self.active_agents = {key: not value for key, value in done.items()}

		# Update ground truth
		if self.dynamic:
			self.ground_truth.step()

		return self.photographs, done #if self.frame_stacking is None else self.frame_stacking.process(self.photographs), reward, done, self.info

	def update_photographs(self):
		""" Update the photograph/state for every vehicle. Every channel will be an input of the Neural Network. """

		photographs = {}
		# Channel 0 -> Known boundaries/map
		if self.obstacles:
			obstacle_map = self.scenario_map - self.inside_obstacles_map
		else:
			obstacle_map = self.scenario_map

		# Create fleet position map #
		fleet_position_map = np.zeros_like(self.scenario_map)
		fleet_position_map[self.fleet.agent_positions[:,0], self.fleet.agent_positions[:,1]] = 1.0

		# Channels 3 and 4
		for i in range(self.number_of_agents):
			agent_observation_of_position = np.zeros_like(self.scenario_map)
			agent_observation_of_position[self.fleet.agent_positions[i,0], self.fleet.agent_positions[i,1]] = 1.0 # map only with the position of the observing agent
			
			agent_observation_of_fleet = fleet_position_map.copy()
			agent_observation_of_fleet[self.fleet.agent_positions[i,0], self.fleet.agent_positions[i,1]] = 0.0 # agents map without the observing agent

			"""Cada índice del diccionario de estado es un agente, y en ese índice se va a guardar todo lo siguiente:"""
			photographs[i] = np.concatenate(( 
				obstacle_map[np.newaxis], # Channel 0 -> Known boundaries/map
				self.ground_truth.read()[np.newaxis],
				self.model_mean_map[np.newaxis], # Channel 1 -> Model mean map
				self.model_uncertainty_map[np.newaxis], # Channel 2 -> Model uncertainty map
				agent_observation_of_position[np.newaxis], # Channel 3 -> Observing agent position map
				agent_observation_of_fleet[np.newaxis] # Channel 4 -> Others agents position map
			))

		self.photographs = {agent_id: photographs[agent_id] for agent_id in range(self.number_of_agents) if self.active_agents[agent_id]}

	def render(self):

		import matplotlib.pyplot as plt

		#Siempre represento perspectiva agente 0: # UNDER TEST
		first_available_agent =  np.argmax(self.active_agents) # first True in active_agents # UNDER TEST

		if not any(self.active_agents.values()):
			return
		

		if self.fig is None: # create first frame of fig, if not already created

			self.non_water_mask = self.scenario_map !=1 - self.inside_obstacles_map # mask with True where no water
			
			self.fig, self.axs = plt.subplots(1, 6, figsize=(15,5))
			
			# AXIS 0: Print the obstacles map #
			self.im0 = self.axs[0].imshow(self.photographs[first_available_agent][0], cmap = 'jet')
			self.axs[0].set_title('Navigation map')

			# AXIS 1: Print the Ground Truth #
			self.photographs[first_available_agent][1][self.non_water_mask] = np.nan
			self.im1 = self.axs[1].imshow(self.photographs[first_available_agent][1],  cmap='jet', vmin=0.0, vmax=1.0)
			self.axs[1].set_title("Real Importance (GT)")

			# AXIS 2: Print model mean #
			self.photographs[first_available_agent][2][self.non_water_mask] = np.nan
			self.im2 = self.axs[2].imshow(self.photographs[first_available_agent][2],  cmap='jet', vmin=0.0, vmax=1.0)
			self.axs[2].set_title("Model Mean/Importance")

			# AXIS 3: Print model uncertainty #
			self.photographs[first_available_agent][3][self.non_water_mask] = np.nan
			self.im3 = self.axs[3].imshow(self.photographs[first_available_agent][3],  cmap='gray', vmin=0.0, vmax=1.0)#, norm='log')
			self.axs[3].set_title("Model Uncertainty")

			# AXIS 4: Agent 0 position #
			self.photographs[first_available_agent][4][self.non_water_mask] = 0.75
			self.im4 = self.axs[4].imshow(self.photographs[first_available_agent][4], cmap = 'gray')
			self.axs[4].set_title("Agent 0 position")

			# AXIS 5: Others-than-Agent 0 position #
			self.photographs[first_available_agent][5][self.non_water_mask] = 0.75
			self.im5 = self.axs[5].imshow(self.photographs[first_available_agent][5], cmap = 'gray')
			self.axs[5].set_title("Others agents position")

		# UPDATE FIG INFO/DATA IN EVERY RENDER CALL #
		# AXIS 0: Print the obstacles map #
		self.im0.set_data(self.photographs[first_available_agent][0])
		# AXIS 1: Print the Ground Truth #
		self.photographs[first_available_agent][1][self.non_water_mask] = np.nan
		self.im1.set_data(self.photographs[first_available_agent][1])
		# AXIS 2: Print model mean #
		self.photographs[first_available_agent][2][self.non_water_mask] = np.nan
		self.im2.set_data(self.photographs[first_available_agent][2])
		# AXIS 3: Print model uncertainty #
		self.photographs[first_available_agent][3][self.non_water_mask] = np.nan
		self.im3.set_data(self.photographs[first_available_agent][3])
		# AXIS 4: Agent 0 position #
		self.photographs[first_available_agent][4][self.non_water_mask] = 0.75
		self.im4.set_data(self.photographs[first_available_agent][4])
		# AXIS 5: Others-than-Agent 0 position #
		self.photographs[first_available_agent][5][self.non_water_mask] = 0.75
		self.im5.set_data(self.photographs[first_available_agent][5])

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()

		plt.draw()

		plt.pause(0.01)

class GaussianProcess:

	def __init__(self, scenario_map):
		""" Gaussian Process to predict the map """

		# Assign input variables #
		self.scenario_map = scenario_map

		# Define Gaussian Process with Kernel #
		self.kernel = ConstantKernel(1) #* Matern(length_scale=5, length_scale_bounds=(1, 1000)) + WhiteKernel(0.005, noise_level_bounds=(0.00001, 0.1))
		self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.001, n_restarts_optimizer=1)

		# Initialize necessary variables #
		self.X_meas = np.empty((0, 2)) # empty array of dimensions (0, 2) to vstack samples coordinates
		self.y_meas = np.empty((0, 1)) # vertical empty array
		self.alphas = np.empty(0) # empty array vacío 0-dimension

		self.visitable_indexes = np.where(self.scenario_map == 1) # map visitable cells indexes
		self.X_cells = np.vstack(self.visitable_indexes).T # visitable cells

		# Define empty maps to predict the model #
		self.model_map = np.zeros_like(self.scenario_map)
		self.uncertainty_map = np.zeros_like(self.scenario_map)
	    
	def fit_gp(self, X_new, y_new, variances_new):
		""" Fit Gaussian Process to data """

		# Add new positions, measures and alphas/variance to dataset #
		self.X_meas = np.vstack((self.X_meas, X_new))
		self.y_meas = np.vstack((self.y_meas, np.asarray(y_new).reshape(-1,1)))
		self.alphas = np.concatenate((self.alphas, variances_new)) # concat instead of vstack because arrays must be of 1 dimension

		# Overwrite samples if better error
		# indices_existentes = np.isin(self.X_meas, X_new).all(axis=1)
		# indices_nuevos = ~indices_existentes

		# self.y_meas[indices_existentes] = np.where(variances_new < self.alphas[indices_existentes], y_new, self.y_meas[indices_existentes])
		# self.alphas[indices_existentes] = np.where(variances_new < self.alphas[indices_existentes], variances_new, self.alphas[indices_existentes])

		# self.X_meas = np.concatenate((self.X_meas, X_new[indices_nuevos]))
		# self.y_meas = np.concatenate((self.y_meas, y_new[indices_nuevos]))
		# self.alphas = np.concatenate((self.alphas, variances_new[indices_nuevos]))

		# Only one sample per point #
		_, unique_indexes = np.unique(self.X_meas, return_index=True, axis=0)
		self.X_meas = self.X_meas[unique_indexes]
		self.y_meas = self.y_meas[unique_indexes]
		self.alphas = self.alphas[unique_indexes]

		# Update internal gp alpha variable #
		self.gp.alpha = self.alphas

		# FIT #
		self.gp.fit(self.X_meas, self.y_meas)
    
	def predict_gt(self):
		""" Ground truth prediction for visitable cells (X_cells) """
		
		model_out, uncertainty_out = self.gp.predict(self.X_cells, return_std=True)

		# Assign GP predictions to corresponding map cells, "flatten" to match dimensions
		self.model_map[self.visitable_indexes] = model_out.flatten()
		self.uncertainty_map[self.visitable_indexes] = uncertainty_out.flatten()

		return self.model_map, self.uncertainty_map


if __name__ == '__main__':

	from Algorithm.RainbowDQL.ActionMasking.ActionMaskingUtils import ConsensusSafeActionMasking

	sc_map = np.genfromtxt('Environment/Maps/example_map.csv', delimiter=',')

	N_agents = 4 # maximum 4 at the moment

	# visitable = np.column_stack(np.where(sc_map == 1)) #coge las coordenadas de las celdas visitables (1) y las guarda en un array
	#initial_positions = visitable[np.random.randint(0,len(visitable), size=N), :]
	initial_positions = np.array([[30, 20], [40, 25], [40, 20], [30, 28]])[:N_agents, :]
	mean_sensormeasure = np.array([0, 0, 0, 0])[:N_agents] # mean of the measeure of every agent
	std_sensormeasure = np.array([0.03, 0.7, 0.9, 1])[:N_agents] # std of the measure of every agent
	variance_sensormeasure = std_sensormeasure**2 # variance = std^2

	# Create environment # 
	env = MultiAgentPatrolling(scenario_map=sc_map,
							   number_of_vehicles=N_agents,
							   max_distance_traveled=2500,
							   mean_sensormeasure=mean_sensormeasure,
							   std_sensormeasure=std_sensormeasure,
							   variance_sensormeasure=variance_sensormeasure,
							   fleet_initial_positions=initial_positions,
							   seed=0,
							   movement_length=2,
							   check_collisions_within=False, # con el ConsensusSafeActionMasking nunca se van a chocar, así que apago las comprobaciones
							   max_collisions=1000,
							   reward_type='model_changes',
							   ground_truth_type='shekel',
							   dynamic=False,
							   obstacles=False,
							   frame_stacking=1,
							   state_index_stacking=(2,3,4),
							   reward_weights=(1.0, 0.1),
							   hard_penalization=False
							 )
	
	action_masking_module = ConsensusSafeActionMasking(navigation_map = sc_map, action_space_dim = 8, movement_length = 2)
 
	env.reset_env()

	R = [] # reward


	actions = {i: np.random.randint(0,8) for i in range(N_agents)} 
	steps = 0
	done = {i:False for i in range(N_agents)} 
	while any([not value for value in done.values()]): # while 1 active at least
		steps += 1


		q = np.random.rand(N_agents, 8)

		for agent_id, action in actions.items():
			q[agent_id, action] = 1000


		actions = action_masking_module.query_actions(q, env.fleet.get_fleet_positions())


		#for idx, agent in enumerate(env.fleet.vehicles):
		#	if done[idx]== False: # UNDER TEST
				# NO se permite colisión con fronteras/obstáculos		
				# agent_mask = np.array([agent.check_action(a) for a in range(8)], dtype=int) 

				#if agent_mask[actions[idx]] or steps % 10 == 0: # si hay intento de choque o 10 pasos, se cambia la acción, si no se mantiene la misma
				#	actions[idx] = np.random.choice(np.arange(8), p=(1-agent_mask)/np.sum((1-agent_mask))) # prob=0 las acciones que hacen chocar

		# s, r, done, _ = env.step_env(actions)
		s, done = env.step_env(actions)

		env.render()

		# R.append(list(r.values()))

		# print(r)


	env.render()
	plt.show()

	# plt.plot(np.cumsum(np.asarray(R),axis=0), '-o')
	# plt.xlabel('Step')
	# plt.ylabel('Individual Reward')
	# plt.legend([f'Agent {i}' for i in range(N_agents)])
	# plt.grid()
	# plt.show()
