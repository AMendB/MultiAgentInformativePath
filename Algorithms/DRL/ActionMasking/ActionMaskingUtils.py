import numpy as np

class SafeActionMasking:

	def __init__(self, action_space_dim: int, movement_length: float) -> None:
		""" Safe Action Masking """

		self.navigation_map = None
		self.position = None
		self.angle_set = np.linspace(0, 2 * np.pi, action_space_dim, endpoint=False)
		self.movement_length = movement_length

	def update_state(self, position: np.ndarray, new_navigation_map: np.ndarray = None):
		""" Update the navigation map """

		if new_navigation_map is not None:
			self.navigation_map = new_navigation_map

		""" Update the position """
		self.position = position

	def mask_action(self, q_values: np.ndarray = None):

		if q_values is None:
			""" Random selection """
			q_values = np.random.rand(8)

		movements = np.array([np.round(np.array([np.cos(angle), np.sin(angle)]) * self.movement_length ).astype(int) for angle in self.angle_set])
		next_positions = self.position + movements

		action_mask = np.array([self.navigation_map[int(next_position[0]), int(next_position[1])] == 0 for next_position in next_positions]).astype(bool)

		q_values[action_mask] = -np.inf

		return q_values, np.argmax(q_values)

class NoGoBackMasking:

	def __init__(self) -> None:
		
		self.previous_action = None

	def mask_action(self, q_values: np.ndarray = None):

		if q_values is None:
			""" Random selection """
			q_values = np.random.rand(8)

		if self.previous_action is None:
			self.previous_action = np.argmax(q_values)
		else:
			return_action = (self.previous_action + len(q_values) // 2) % len(q_values)
			q_values[return_action] = -np.inf

		return q_values, np.argmax(q_values)

	def update_last_action(self, last_action):

		self.previous_action = last_action
	
class NoGoBackFleetMasking:

	def __init__(self) -> None:

		self.reset()
		
	def reset(self):

		self.previous_actions = None

	def mask_actions(self, q_values: dict):

		if self.previous_actions is None:
			self.previous_actions = {idx: np.argmax(q_values[idx]) for idx in q_values.keys()}
		else:
			return_actions = {idx: (self.previous_actions[idx] + len(q_values[idx]) // 2) % len(q_values[idx]) for idx in q_values.keys()}
			for idx in q_values.keys():
				q_values[idx][return_actions[idx]] = -1000 # a very low value instead of -np.inf to not have probability of collide with obstacle in random select in case of no alternative way out

		return q_values

	def update_previous_actions(self, previous_actions):

		self.previous_actions = previous_actions

class ConsensusSafeActionMasking:
	""" The optimists decide first! """

	def __init__(self, navigation_map, action_space_dim: int, movement_length: float) -> None:
		
		self.movement_length = movement_length
		self.angle_set = np.linspace(0, 2 * np.pi, action_space_dim, endpoint=False)
		self.fleet_map = np.zeros_like(navigation_map)
		self.navigation_map = navigation_map

	def update_map(self, new_navigation_map: np.ndarray):

		self.navigation_map = new_navigation_map.copy()

	def query_actions(self, q_values: dict, positions: dict, ):

		# 1) The largest q-value agent decides first
		# 2) If there are multiple agents with the same q-value, the agent is selected randomly
		# 3) Then, compute the next position of the agent and update the fleet map
		# 4) The next agent is selected based on the updated fleet map, etc
		
		self.fleet_map = self.navigation_map.copy()
		q_max = {idx: q_values[idx].max() for idx in q_values.keys()}
		agents_order = sorted(q_max, key=q_max.get)[::-1]
		final_actions = {}

		for agent in agents_order:
			
			#Unpack the agent position
			agent_position = positions[agent]

			# Compute next positions
			movements = np.round([np.round(np.array([np.cos(angle), np.sin(angle)])) * self.movement_length for angle in self.angle_set]).astype(int)
			next_positions = agent_position + movements
			next_positions = np.clip(next_positions, (0,0), np.array(self.fleet_map.shape)-1) # saturate movement if out of indexes values (map edges)
			
			# Check if next positions lead to a collision
			action_mask = np.array([self.fleet_map[int(next_position[0]), int(next_position[1])] == 0 for next_position in next_positions]).astype(bool)

			# Censor the impossible actions in the Q-values
			q_values[agent][action_mask] = -np.inf

			# Select the action
			action = np.argmax(q_values[agent])

			# Update the fleet map
			next_position = next_positions[action]
			self.fleet_map[int(next_position[0]), int(next_position[1])] = 0

			# Store the action
			final_actions[agent] = action.copy()


		return final_actions 
		
