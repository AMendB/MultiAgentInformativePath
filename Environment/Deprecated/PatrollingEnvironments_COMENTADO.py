import sys
sys.path.append('.')
import gym
import numpy as np
import matplotlib.pyplot as plt
from Environment.GroundTruthsModels.ShekelGroundTruth import GroundTruth
from Environment.GroundTruthsModels.AlgaeBloomGroundTruth import algae_bloom, algae_colormap, background_colormap
# from Environment.Wrappers.time_stacking_wrapper import MultiAgentTimeStackingMemory
# from scipy.spatial import distance_matrix
import matplotlib
import json

background_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["sienna","dodgerblue"])

np.random.seed(0)

class DiscreteVehicle: # clase para crear vehículos individuales

	def __init__(self, initial_position, n_actions, movement_length, navigation_map, detection_length):
		
		self.fig1 = None
		""" Initial positions of the drones """
		np.random.seed(0) #semilla para que siempre salga igual
		self.initial_position = initial_position # coge la posición que le entra a la función
		self.position = np.copy(initial_position) # copia la posición inicial a la actual

		""" Initialize the waypoints """
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)

		""" Detection radius for the contamination vision """
		self.detection_length = detection_length # coge la distancia de detección que le entra a la función (es 2 en este caso)
		self.navigation_map = navigation_map # cargo el mapa desde el archivo
		self.detection_mask = self.compute_detection_mask() 

		""" Reset other variables """
		self.distance = 0.0 # sumador de distancia recorrida
		self.num_of_collisions = 0 # contador de colisiones
		self.action_space = gym.spaces.Discrete(n_actions)
		self.angle_set = np.linspace(0, 2 * np.pi, n_actions, endpoint=False) #array con los 8 puntos cardinales en RADIANES, divido en 8 direcciones una circunferencia: [0. , 0.78539816, 1.57079633, 2.35619449, 3.14159265, 3.92699082, 4.71238898, 5.49778714]
		self.movement_length = movement_length #¿longitud del movimiento? Establecido a 2 en el init de MultiAgentPatrolling, es el radio de la circunferencia de cardinales
		

	def move(self, action, valid=True):
		""" Move a vehicle in the direction of the action. If valid is False, the action is not performed. """

		angle = self.angle_set[action] # tomo como ángulo de movimiento el que me diga la acción tomada, la acción sirve como índice del array de puntos cardinales
		movement = np.round(np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])).astype(int) #convierto el ángulo en movimiento cartesiano (cuántas celdas se mueve en eje x y cuántas eje y)
		next_position = self.position + movement # posición siguiente, le sumo el movimiento a la actual
		self.distance += np.linalg.norm(self.position - next_position) # sumo la distancia que recorro con el movimiento a la distancia total recorrida

		if self.check_collision(next_position) or not valid: # si la siguiente posición es colisión con tierra o no es válida (colisión entre drones):
			collide = True
			self.num_of_collisions += 1 # sumo una colisión al contador 
		else:
			collide = False
			self.position = next_position # asigno la posición siguiente a la actual
			self.waypoints = np.vstack((self.waypoints, [self.position])) #añado la posición actual al vector de puntos visitados (hace una concatenación de los dos vectores, el de todos los puntos y el array convertido a lista que contiene el punto actual)

		self.detection_mask = self.compute_detection_mask() # llamada a función que devuelve el área de covertura alrededor de la posición (máscara sobre el mapa general)
		return collide # devuelve si ha habido una colisión

	def check_collision(self, next_position): #función para comprobar simplemente si hay una colisión: devuelve true o false

		if self.navigation_map[int(next_position[0]), int(next_position[1])] == 0: # si la posición es cero en el mapa (obstáculo)
			return True  # There is a collision

		return False

	def compute_detection_mask(self): # función para calcular el área de covertura alrededor de la posición (máscara sobre el mapa general)
		""" Compute the circular mask """

		known_mask = np.zeros_like(self.navigation_map) # obtengo un array del mapa original relleno de ceros

		px, py = self.position.astype(int) # separo posición en eje x e y 

		# State - coverage area #
		x = np.arange(0, self.navigation_map.shape[0]) # posiciones posibles en x
		y = np.arange(0, self.navigation_map.shape[1]) # posiciones posibles en y

		# Compute the circular mask (area) of the state 3 #
		mask = (x[np.newaxis, :] - px) ** 2 + (y[:, np.newaxis] - py) ** 2 <= self.detection_length ** 2 
		# numpy.newaxis is used to increase the dimension of the existing array by one more dimension, when used once (https://stackoverflow.com/questions/29241056/how-do-i-use-np-newaxis)
		#		con eso convierte la x en un vector fila y la y en un vector columna, y le resta la posición actual a todos los componentes

		known_mask[mask.T] = 1.0 # convierto los valores True en 1 y los False en 0
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

		angle = self.angle_set[action] # ángulo de la acción a tomar
		movement = np.round(np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])).astype(int) # calcula el movimiento igual que en la función move
		next_position = self.position + movement

		return self.check_collision(next_position) # devuelve la comprobación de si la siguiente posición es colisión

	def move_to_position(self, goal_position):
		""" Move to the given position """

		assert self.navigation_map[goal_position[0], goal_position[1]] == 1, "Invalid position to move" # si no es válida la posición (0 en lugar de 1) devuelve mensaje notificando
		self.distance += np.linalg.norm(goal_position - self.position) # suma a la distancia recorrida total la norma entre los dos puntos (la distancia entre ellos) con la librería de álgebra linalg
		""" Update the position """
		self.position = goal_position  # actualiza la posición actual a la que era objetivo

class DiscreteFleet: # clase para crear FLOTAS de la clase de vehículos

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
		self.number_of_vehicles = number_of_vehicles # número de vehículos
		self.initial_positions = fleet_initial_positions # posición inicial de la flota
		self.n_actions = n_actions # número de acciones 
		self.movement_length = movement_length # longitud del motimiento
		self.detection_length = detection_length # distancia de detección

		""" Create the vehicles object array """
		self.vehicles = [DiscreteVehicle(initial_position=fleet_initial_positions[k],
										 n_actions=n_actions,
										 movement_length=movement_length,
										 navigation_map=navigation_map,
										 detection_length=detection_length) for k in range(self.number_of_vehicles)] #list comprehension, crea tantos vehículos como marque number_of_vehicles

		self.agent_positions = np.asarray([veh.position for veh in self.vehicles]) # guarda la posición de todos los vehículos en un array

		# Get the redundancy mask #
		self.redundancy_mask = np.sum([veh.detection_mask for veh in self.vehicles], axis=0) # mira qué zona están mirando dos drones a la vez (redundancia innecesaria)
		# Get the collective detection mask #
		self.collective_mask = self.redundancy_mask.astype(bool) 
		self.historic_visited_mask = self.redundancy_mask.astype(bool) # al inicio, la matriz histórica de zonas visitadas será igual a la primera zona visitada (la actual, donde spawnean los drones)
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

	@staticmethod # Un método estático es un método que pertenece a una clase en lugar de una instancia de la clase, y puede ser llamado sin crear una instancia de la clase.
	def majority(arr: np.ndarray) -> bool:
		return arr.sum() >= len(arr) // 2 # this function checks if the sum of the elements in the array is greater than or equal 
										#to half the length of the array, which can be interpreted as checking if the majority of the elements in the array are "True" or "non-zero".

	def check_fleet_collision_within(self, veh_actions):
		""" Check if there is any collision between agents """
		""" En resumen, este método comprueba si alguno de los vehículos de la flota colisiona entre sí en función de sus nuevas posiciones, que 
		vienen determinadas por sus acciones y longitudes de movimiento. El método devuelve una matriz booleana que indica si la nueva posición de cada vehículo es única o no."""
		
		new_positions = []

		for idx, veh_action in veh_actions.items(): # para cada id del vehículo y cada acción que se encuentren en el diccionario veh_actions

			angle = self.vehicles[idx].angle_set[veh_action] # coge el ángulo asociado a la acción para cada vehículo
			movement = np.round(np.array([self.vehicles[idx].movement_length * np.cos(angle), self.vehicles[idx].movement_length * np.sin(angle)])).astype(int)
			new_positions.append(list(self.vehicles[idx].position + movement)) # añade a la lista todas las nuevas posiciones de los drones

		_, inverse_index, counts = np.unique(np.asarray(new_positions), return_inverse=True, return_counts=True, axis=0) #comprueba si las nuevas posiciones son únicas o no, en caso de no serlo es que habría colisión entre los drones en el siguiente movimiento
		"""La función np.unique() devuelve tres matrices: unique_positions, inverse_index y counts. Aquí, unique_positions se descarta, pero es una matriz 2D que contiene las 
		posiciones únicas en la lista new_positions, inverse_index es una matriz del mismo tamaño que new_positions que mapea cada posición en new_positions 
		a su índice en unique_positions, y counts es una matriz del mismo tamaño que unique_positions que contiene el número de veces que aparece cada posición
		única en new_positions. El propósito de calcular la matriz counts es verificar las colisiones entre vehículos. Si una posición aparece más de una vez en 
		la lista new_positions (es decir, si su conteo correspondiente en counts es mayor que 1), entonces significa que dos o más vehículos han chocado y terminaron 
		en la misma posición. Por otro lado, si una posición aparece solo una vez en new_positions, entonces significa que el vehículo correspondiente se ha movido 
		a una posición única sin chocar con ningún otro vehículo. Luego, se crea la matriz booleana not_collision_within en función de esta información."""

		# True if repeated #
		not_collision_within = counts[inverse_index] == 1 # es True para los elementos en los que el recuento correspondiente es igual a 1 (no se chocan), y False en caso contrario.

		return not_collision_within

	def move(self, fleet_actions):

		# Check if there are collisions between vehicles #
		self_colliding_mask = self.check_fleet_collision_within(fleet_actions)
		# Process the fleet actions and move the vehicles # 
		collision_array = {k: self.vehicles[k].move(fleet_actions[k], valid=valid) for k, valid in zip(list(fleet_actions.keys()), self_colliding_mask)}
		""" Se utiliza fleet_actions.keys() para iterar sobre todos los vehículos que tienen acciones en fleet_actions, y 
		así poder llamar al método move() de cada vehículo con su respectiva acción, y este devuelve si ha habido una colisión."""
		"""`zip` se utiliza para unir dos listas de igual longitud y crear una lista de tuplas que contienen elementos de ambas listas en el mismo índice. 
		En este caso, `zip` se utiliza para iterar sobre dos listas de igual longitud (`list(fleet_actions.keys())` y `self_colliding_mask`), y en cada 
		iteración se devuelve k que es el índice del vehículo y el booleano que indica si ese vehículo está en una posición que no entra en colisión con otros vehículos. 	

		Por ejemplo, si `list(fleet_actions.keys())` es `[0, 1, 2]` y `self_colliding_mask` es `[True, False, True]`, `zip` devolverá una secuencia de tres tuplas:
		(0, True)
		(1, False)
		(2, True)
		Esto se utiliza para actualizar el diccionario `collision_array` con el número correcto de colisiones para cada vehículo que se ha movido."""
		"""`collision_array` es un diccionario que almacena información sobre las colisiones que han ocurrido durante el movimiento de los vehículos en la flota. 
		La clave del diccionario es el índice del vehículo en la flota, y el valor es el número de colisiones que ha tenido el vehículo en esta iteración. 
		Por ejemplo, si el diccionario se ve así:
		collision_array = {0: 1, 1: 0, 2: 2}
		Esto significa que el vehículo 0 ha tenido 1 colisión, el vehículo 1 no ha tenido ninguna colisión, y el vehículo 2 ha tenido 2 colisiones. 
		Esta información puede ser útil para tomar decisiones en tiempo real sobre el movimiento de los vehículos, como detener temporalmente un vehículo que 
		ha tenido demasiadas colisiones para evitar daños adicionales."""
		# Update vector with agent positions #
		self.agent_positions = np.asarray([veh.position for veh in self.vehicles])
		# Sum up the collisions for termination #
		self.fleet_collisions = np.sum([self.vehicles[k].num_of_collisions for k in range(self.number_of_vehicles)]) # suma de todas las colisiones de la flota
		# Compute the redundancy mask #
		self.redundancy_mask = np.sum([veh.detection_mask for veh in self.vehicles], axis=0) # se suman las máscaras de detección de todos los vehículos, si hay un punto con un número mayor que 1 es que hay redundancia
		# Update the collective mask #
		self.collective_mask = self.redundancy_mask.astype(bool) # convierte la máscara a booleana, 0 = false, 1 >= true
		# Update the historic visited mask #
		self.historic_visited_mask = np.logical_or(self.historic_visited_mask, self.collective_mask) # añade la zona actual al histórico de zonas visitadas
		# Update the isolation mask (for networked agents) #
		self.update_isolated_mask()

		return collision_array # devuelve el diccionario donde se indica el número de colisiones de cada vehículo

	def update_isolated_mask(self): # La máscara de aislamiento indica si todos los vehículos en la flota están aislados de los demás
		""" Compute the mask of isolated vehicles. Only for restricted fleets. """

		# Get the distance matrix #
		distance = self.get_distance_matrix() # matriz de distancias entre todos los vehículos en la flota 
		# Delete the diagonal (self-distance, always 0) se elimina la diagonal que siempre es cero ya que se refiere a la distancia de cada vehículo consigo mismo. #
		self.distance_between_agents = distance[~np.eye(distance.shape[0], dtype=bool)].reshape(distance.shape[0], -1)
		""" distance es una matriz cuadrada de distancias entre los vehículos. Para obtener la matriz distance_between_agents, que es una 
		matriz unidimensional que contiene todas las distancias entre cada par de vehículos distintos, se utiliza la función np.eye() para crear una 
		matriz de booleanos con True en la diagonal y False en el resto de elementos. Luego, se utiliza el operador ~ (complemento lógico) para invertir 
		esa matriz, de manera que la diagonal tiene valor False y el resto de elementos tienen valor True.
		A continuación, se utiliza el método `reshape` para convertir esa matriz en una matriz unidimensional (`self.distance_between_agents`), que 
		contiene las distancias entre cada par de vehículos distintos. La razón de hacer esto es para no contar las distancias entre un vehículo y sí mismo.

		Por ejemplo, si `distance` es la siguiente matriz:
			array([[ 0.        ,  3.16227766,  4.472136  ],
				[ 3.16227766,  0.        ,  2.23606798],
				[ 4.472136  ,  2.23606798,  0.        ]])
		La matriz resultante `self.distance_between_agents` será:
			array([3.16227766, 4.472136  , 2.23606798])
		que son las distancias entre los vehículos 1 y 2, 1 y 3, y 2 y 3, respectivamente. """
		# True if all agents are further from the danger distance
		danger_of_isolation_mask = self.distance_between_agents > self.optimal_connection_distance # se obtiene una máscara que para los valores True indica que esos agentes están en peligro de aislamiento
		self.danger_of_isolation = np.asarray([self.majority(value) for value in danger_of_isolation_mask])
		"""La función majority se utiliza para determinar si la mayoría de los valores de un array booleano son verdaderos o falsos. 
		En este caso, se utiliza para calcular si la mayoría de los valores de danger_of_isolation_mask son verdaderos o falsos. 
		Si la mayoría de los valores son verdaderos, significa que no hay peligro de aislamiento para los vehículos en la flota, porque no están más lejos que la distancia óptima de conexión. 
		Si la mayoría de los valores son falsos, significa que hay peligro de aislamiento para al menos uno de los vehículos en la flota."""

		# True if all agents are further from the max connection distance
		isolation_mask = self.distance_between_agents > self.max_connection_distance # máscara booleana que indica para cada vehículo si está a una distancia mayor que la distancia máxima de conexión de todos los demás vehículos
		self.isolated_mask = np.asarray([self.majority(value) for value in isolation_mask])
		self.number_of_disconnections += np.sum(self.isolated_mask) # cuenta el número de vehículos que YA SÍ están aislados (que cumplen la condición de self.isolated_mask)
# =============================================================================
	# def measure(self, gt_field): # toma medida con el sensor

	# 	"""
	# 	Take a measurement in the given N positions
	# 	:param gt_field:
	# 	:return: An numpy array with dims (N,2)
	# 	"""
	# 	positions = np.array([self.vehicles[k].position for k in range(self.number_of_vehicles)])

	# 	values = []
	# 	for pos in positions:
	# 		values.append([gt_field[int(pos[0]), int(pos[1])]])

	# 	if self.measured_locations is None:
	# 		self.measured_locations = positions
	# 		self.measured_values = values
	# 	else:
	# 		self.measured_locations = np.vstack((self.measured_locations, positions))
	# 		self.measured_values = np.vstack((self.measured_values, values))

	# 	return self.measured_values, self.measured_locations
# =============================================================================
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
		return [self.vehicles[k].distance for k in range(self.number_of_vehicles)] # devuelve una lista con las distancias recorridas por cada vehículo
# =============================================================================
	# def check_collisions(self, test_actions):
	# 	""" Array of bools (True if collision) """
	# 	return [self.vehicles[k].check_action(test_actions[k]) for k in range(self.number_of_vehicles)] # devuelve una lista booleana comprobando si hay colisión para cada vehículo

	# def move_fleet_to_positions(self, goal_list):
	# 	""" Move the fleet to the given positions.
	# 	 All goal positions must ve valid. """

	# 	goal_list = np.atleast_2d(goal_list)

	# 	for k in range(self.number_of_vehicles):
	# 		self.vehicles[k].move_to_position(goal_position=goal_list[k])
# =============================================================================
	def get_distance_matrix(self): # devuelve matriz de distancias entre todos los vehículos en la flota 
		return distance_matrix(self.agent_positions, self.agent_positions)

	def get_positions(self): 
		return np.asarray([veh.position for veh in self.vehicles]) # devuelve una lista con las posiciones actuales de cada vehículo


class MultiAgentPatrolling(gym.Env):

	def __init__(self, scenario_map,
				 distance_budget, # distancia máxima capaz de recorrer el dron (analogía de batería)
				 number_of_vehicles,
				 fleet_initial_positions=None,
				 seed=0,
				 miopic=True, # ¿?
				 detection_length=2,
				 movement_length=2,
				 max_collisions=5,
				 forget_factor=1.0,
				 networked_agents=False,
				 max_connection_distance=10,
				 optimal_connection_distance=5,
				 max_number_of_disconnections=10,
				 attrittion=0.0, # "desgaste"
				 obstacles=False,
				 hard_penalization=False,
				 reward_type='weighted_idleness', # tipo de recompensa
				 reward_weights = (10.0, 1.0),
				 ground_truth_type='algae_bloom', # tipo de entorno
				 frame_stacking = 0,
				 dynamic=False, # si se quiere que se muevan las algas
				 state_index_stacking = (0,1,2,3,4)):

		""" The gym environment """

		# Load the scenario map
		np.random.seed(seed)
		self.scenario_map = scenario_map
		self.visitable_locations = np.vstack(np.where(self.scenario_map != 0)).T # array con las coordenadas de todas las celdas visitables (las que son != de cero en el mapa)
		self.number_of_agents = number_of_vehicles
		self.dynamic = dynamic

		# Initial positions
		if fleet_initial_positions is None:
			self.random_inititial_positions = True
			random_positions_indx = np.random.choice(np.arange(0, len(self.visitable_locations)), number_of_vehicles, replace=False) # se selecciona un índice aleatorio como máximo el número de celdas visitables
			self.initial_positions = self.visitable_locations[random_positions_indx] # con ese índice aleatorio se selecciona la posición inicial aleatoria dentro de las celdas visitables
		else:
			self.random_inititial_positions = False
			self.initial_positions = fleet_initial_positions

		self.obstacles = obstacles
		self.miopic = miopic
		self.reward_type = reward_type
	
		# Number of pixels
		self.distance_budget = distance_budget
		self.max_number_of_movements = distance_budget // detection_length
		# 
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

		# self.action_space = gym.spaces.Discrete(8) # define el espacio de acciones que puede tomar el agente en el entorno (8 acciones)

		assert frame_stacking >= 0, "frame_stacking must be >= 0"
		self.frame_stacking = frame_stacking
		self.state_index_stacking = state_index_stacking

		""" Se crea el espacio de observaciones de la simulación. frame_stacking indica cuántos frames se apilarán en cada observación. Si frame_stacking es 0, 
		entonces solo se observa el estado actual de la simulación. Si frame_stacking es mayor que 0, entonces se apilarán frame_stacking-1 estados previos 
		para formar una observación más completa."""
		if frame_stacking != 0: # Si se está apilando frames, entonces se utiliza la clase MultiAgentTimeStackingMemory para manejar la memoria de los estados previos.
			self.frame_stacking = MultiAgentTimeStackingMemory(n_agents = self.number_of_agents,
			 													n_timesteps = frame_stacking - 1, # -1 porque la información actual del entorno ya está incluida en la observación actual
																state_indexes = state_index_stacking, #tupla que indica qué elementos del estado se guardarán en la memoria (por ejemplo, (0,1,2,3,4) indica que se guardarán las cinco capas del mapa)
																n_channels = 5) # n_channels indica el número de canales en cada estado (que es 5 para la simulación actual).
			self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5 + len(state_index_stacking)*(frame_stacking - 1), *self.scenario_map.shape), dtype=np.float32)
			"""define el espacio de observaciones para la simulación. El tamaño del espacio de observaciones depende de si se están apilando frames o no, y del número de 
			elementos del estado que se están guardando en la memoria. En ambos casos, el espacio de observaciones es un arreglo tridimensional de tipo float32, donde la 
			primera dimensión corresponde al número de capas en cada observación, y las dos siguientes dimensiones corresponden al tamaño del mapa de la simulación.
			El asterisco "*" se utiliza para desempaquetar los elementos de una tupla en argumentos individuales. En este caso, self.scenario_map.shape es una tupla que 
			contiene la forma de la matriz de mapa de escenario, por ejemplo, (10, 10). Usando el asterisco, *self.scenario_map.shape se desempaqueta para pasar los elementos 
			individuales de la tupla como argumentos a la función gym.spaces.Box."""
		else: # si no se está usando más de un frame
			self.frame_stacking = None
			self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5, *self.scenario_map.shape), dtype=np.float32)

		self.state_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4, *self.scenario_map.shape), dtype=np.float32)

		
		#self.individual_action_state = gym.spaces.Discrete(8) # define el espacio de acciones que puede tomar el agente en el entorno (8 acciones)

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
			# Generate a random inside obstacles map #
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
		"""Esta función actualiza la máscara temporal de inactividad (`idleness_matrix`). En resumen, esta matriz se utiliza para registrar la 
		inactividad de cada celda del mapa en función de la actividad reciente de los drones en esa celda.
		En detalle, la función actualiza la matriz de inactividad sumando 1 al valor de cada celda y dividiendo cada valor por el factor de olvido 
		y el número máximo de movimientos permitidos (`self.max_number_of_movements`) que se pueden realizar en el área de patrullaje. Luego, se 
		resta la matriz de máscaras colectivas de los drones (`self.fleet.collective_mask`) para que las celdas en las que un dron está activo no 
		se consideren inactivas. Finalmente, se ajusta la matriz para que sus valores estén en el rango [0,1] utilizando la función `np.clip`."""
		self.idleness_matrix = self.idleness_matrix + 1.0 / (self.forget_factor * self.max_number_of_movements)
		self.idleness_matrix = self.idleness_matrix - self.fleet.collective_mask
		self.idleness_matrix = np.clip(self.idleness_matrix, 0, 1)

		return self.idleness_matrix

	def update_information_importance(self):
		""" Applied the attrition term """
		"""La función `update_information_importance()` actualiza la matriz `importance_matrix` aplicando el término de "attrition", que es 
		un factor que indica el desgaste o la disminución de la importancia de la información a lo largo del tiempo. La matriz `importance_matrix` 
		contiene información sobre la importancia de cada celda del mapa en función de la recompensa que proporciona. Por lo tanto, esta función 
		se utiliza para disminuir la importancia de las celdas que han sido visitadas con frecuencia por los drones y que ya no proporcionan tanta 
		recompensa, de forma que se incentiva a los drones a explorar nuevas áreas del mapa en busca de nuevas fuentes de recompensa."""
		self.importance_matrix = np.clip(
			self.importance_matrix - self.attrition * self.gt.read() * self.fleet.collective_mask, 0, 999999)

	def update_state(self):
		""" Update the state for every vehicle """
		""" Es una fotografía de todos los aspectos que definen al estado actual. Tendrá varios
		canales, como: mapa de obstáculos, modelo, posición del agente, posición de los demás agentes..."""

		state = {}

		"""Canal 1: este canal es una imagen de las fronteras del mapa conocidas por los agentes. 
		Si los obstáculos están activados, se hace un and entre los obstáculos generados de forma random al inicio 
		(inside_obstacle) y los de historic_visited_mask(??). Se resta al scenario esa máscara, lo que convierte en 0 
		las celdas que antes eran visitables (1)."""
		# Channel 1 -> Known boundaries (fronteras)
		if self.obstacles:
			obstacle_map = self.scenario_map - np.logical_and(self.inside_obstacles_map, self.fleet.historic_visited_mask)
		else:
			obstacle_map = self.scenario_map

		"""Estado 2: Representa la matriz de importancia de la información conocida por los agentes. La importancia 
		de cada celda se calcula en función de la información desconocida que contiene. Si el agente tiene una vista 
		miópica (miopic=True), se utiliza una matriz que tiene valor -1 en las celdas no visitadas y el valor de la 
		información conocida en las celdas visitadas. 
		np.where() es una función de NumPy que se utiliza para encontrar los índices donde se cumple una determinada condición.
		Cuando se aplica np.where() a una condición, devuelve una tupla de dos array: uno que contiene los índices de fila 
		y otro que contiene los índices de columna donde la condición se cumple.
		Aquí, self.fleet.historic_visited_mask es una matriz booleana que representa las celdas visitadas por los agentes. 
		np.where() se utiliza para encontrar los índices (fila y columna) de las celdas donde self.fleet.historic_visited_mask 
		es True, es decir, los índices de las celdas visitadas.
		Estos índices se utilizan posteriormente para seleccionar las celdas correspondientes en la matriz self.model y copiar 
		esos valores en la matriz known_information, que representa la información conocida por los agentes en el estado 2, es decir, 
		el modelo pasa a ser información conocida.
		
		De lo contrario, si no es miopic, se utiliza directamente la matriz de información conocida (gt.read())."""
		# State 2 -> Known information
		# state[2] = self.importance_matrix * self.fleet.historic_visited_mask if self.miopic else self.importance_matrix
		if self.miopic:
			known_information = -np.ones_like(self.model)
			known_information[np.where(self.fleet.historic_visited_mask)] = self.model[np.where(self.fleet.historic_visited_mask)]
		else:
			known_information = self.gt.read()

		# Create fleet position #
		"""fleet_position_map representa la posición de la flota en el mapa. Es una matriz del mismo tamaño que el mapa 
		del escenario, con un valor de 1.0 en las posiciones donde hay un agente y 0.0 en el resto de las celdas."""
		fleet_position_map = np.zeros_like(self.scenario_map)
		fleet_position_map[self.fleet.agent_positions[:,0], self.fleet.agent_positions[:,1]] = 1.0

		# State 3 and 4
		"""Representa una vista del entorno desde la perspectiva de cada agente"""
		for i in range(self.number_of_agents):
			agent_observation_of_fleet = fleet_position_map.copy()
			agent_observation_of_fleet[self.fleet.agent_positions[i,0], self.fleet.agent_positions[i,1]] = 0.0 

			agent_observation_of_position = np.zeros_like(self.scenario_map)
			agent_observation_of_position[self.fleet.agent_positions[i,0], self.fleet.agent_positions[i,1]] = 1.0
			
			"""Cada índice del diccionario de estado es un agente, y en ese índice se va a guardar todo lo siguiente:"""
			state[i] = np.concatenate(( 
				obstacle_map[np.newaxis], # con np.newaxis pasan todos los array de (58,38) a (1,58,38)
				self.idleness_matrix[np.newaxis],
				known_information[np.newaxis],
				agent_observation_of_fleet[np.newaxis],
				agent_observation_of_position[np.newaxis]
			))

		self.state = {agent_id: state[agent_id] for agent_id in range(self.number_of_agents) if self.active_agents[agent_id]}

	def step(self, action: dict):
		"""Ejecuta todas las actualizaciones para cada step"""

		# Process action movement only for active agents #
		action = {action_id: action[action_id] for action_id in range(self.number_of_agents) if self.active_agents[action_id]}
		collision_mask = self.fleet.move(action)
		print(collision_mask)
		print(self.fleet.fleet_collisions)

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
		"""Se crea un diccionario llamado done que indica si cada agente ha terminado su episodio o no. Para determinar si un agente ha terminado, se evalúan dos condiciones:
		    self.fleet.get_distances()[agent_id] > self.distance_budget: Esto verifica si la distancia recorrida por el agente, obtenida mediante self.fleet.get_distances(), 
				es mayor que el límite de distancia establecido en self.distance_budget. Si la distancia recorrida supera el límite, se considera que el agente ha terminado.

		    self.fleet.fleet_collisions > self.max_collisions: Esto verifica si el número de colisiones totales de la flota de agentes, almacenado en self.fleet.fleet_collisions, 
				es mayor que el máximo permitido establecido en self.max_collisions. Si el número de colisiones supera el límite, se considera que el agente ha terminado."""
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
			"""Con la máscara de detección hace que sólo se lea del gt el alrededor del vehículo para ser
			guardado en el modelo."""

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

			# Others-than-Agent 0 position #
			self.im4 = self.axs[4].imshow(self.state[agente_disponible][3], cmap = 'gray')
			self.axs[4].set_title("Others agents position")

			# Agent 0 position #
			self.im5 = self.axs[5].imshow(self.state[agente_disponible][4], cmap = 'gray')
			self.axs[5].set_title("Agent 0 position")

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


	sc_map = np.genfromtxt('Environment/Maps/ypacarai_map_low_res.csv', delimiter=',')

	N = 4
	initial_positions = np.array([[30, 20], [40, 25], [40, 20], [30, 28]])[:N, :]
	visitable = np.column_stack(np.where(sc_map == 1)) #coge las coordenadas de las celdas visitables (1) y las guarda en un array
	#initial_positions = visitable[np.random.randint(0,len(visitable), size=N), :]
	

	env = MultiAgentPatrolling(scenario_map=sc_map,
							   fleet_initial_positions=initial_positions,
							   distance_budget=250,
							   number_of_vehicles=N,
							   seed=0,
							   miopic=True, 
							   detection_length=2,
							   movement_length=2,
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

	action = {i: np.random.randint(0,8) for i in range(N)} #la acción se toma totalmente aleatoria

	while not any(list(done.values())):

		for idx, agent in enumerate(env.fleet.vehicles):
		
			agent_mask = np.array([agent.check_action(a) for a in range(8)], dtype=int) # máscara para cada acción: [0 0 0 0 0 0 0 0] si no hay choque

			if agent_mask[action[idx]]: # si hay choque, se cambia la acción, si no no entra en el if y se mantiene la misma acción (no se actualiza) hasta que choque con un obstáculo
				action[idx] = np.random.choice(np.arange(8), p=(1-agent_mask)/np.sum((1-agent_mask))) # prob=0 las acciones que hacen chocar


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
