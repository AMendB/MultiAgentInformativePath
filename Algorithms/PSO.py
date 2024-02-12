import numpy as np

class ParticleSwarmOptimizationAgent:

    def __init__(self, world: np.ndarray, number_of_actions: int, movement_length: int, seed=0):

        self.world = world
        self.action = None
        self.number_of_actions = number_of_actions
        self.movement_length = movement_length
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
    
        self.max_local_measure = [0, 0]
        self.max_global_measure = 0
        self.best_local_location = None
        self.best_global_location = None

        # Constant ponderation values [c1, c2, c3, c4] https://arxiv.org/pdf/2211.15217.pdf #
        self.velocities = 0
        self.c_exploration = (2.0187, 0, 3.2697, 0) 
        self.c_explotation = (3.6845, 1.5614, 0, 3.6703) 
        self.actions_directions = np.array([[np.cos(2*np.pi*i/self.number_of_actions), np.sin(2*np.pi*i/self.number_of_actions)] for i in range(self.number_of_actions)]) 

    def move(self, mean_map: np.ndarray,  uncert_map: np.ndarray, 
                 measures: np.ndarray, traveled_distance: int, actual_agent_position: np.ndarray, all_agents_positions: np.ndarray):
        """ Compute the new state """
        self.model_mean = mean_map
        self.model_uncertainty = uncert_map
        self.measures = measures
        self.actual_agent_position = actual_agent_position
        self.agents_positions = all_agents_positions

        if traveled_distance > 50:
            self.c_values = self.c_explotation
        else:
            self.c_values = self.c_exploration


        self.observing_agent_index = np.where(np.all(self.agents_positions == self.actual_agent_position, axis=1))[0][0]

        self.action = self.update_vectors()
        
        return self.action

    def update_vectors(self):
        """ Update the vectors direction of the agent """

        # Update best historic locations #
        if self.measures[self.observing_agent_index] >= self.max_local_measure:
            self.max_local_measure = self.measures[self.observing_agent_index]
            self.best_local_location = self.agents_positions[self.observing_agent_index]
        if np.max(self.measures) >= self.max_global_measure:
            self.max_global_measure = np.max(self.measures)
            self.best_global_location = self.agents_positions[np.argmax(self.measures)]

        # Update maximum of model mean and uncertainty location #
        self.max_mean_location = np.unravel_index(np.argmax(self.model_mean), self.model_mean.shape)
        self.max_uncertainty_location = np.unravel_index(np.argmax(self.model_uncertainty), self.model_uncertainty.shape)

        # Update the vectors #
        self.vector_to_best_local_location = self.best_local_location - self.agents_positions[self.observing_agent_index]
        self.vector_to_best_global_location = self.best_global_location - self.agents_positions[self.observing_agent_index]
        self.vector_to_max_uncertainty = self.max_uncertainty_location - self.agents_positions[self.observing_agent_index]
        self.vector_to_max_mean = self.max_mean_location - self.agents_positions[self.observing_agent_index]

        # Get final ponderated vector c*u #
        self.u_values = np.random.uniform(0, 1, 4) # random ponderation values
        self.vector = self.c_values[0] * self.u_values[0] * self.vector_to_best_local_location + \
                      self.c_values[1] * self.u_values[1] * self.vector_to_best_global_location + \
                      self.c_values[2] * self.u_values[2] * self.vector_to_max_uncertainty + \
                      self.c_values[3] * self.u_values[3] * self.vector_to_max_mean
        
        self.velocities = self.velocities + self.vector
        self.velocities = np.clip(self.velocities, -2, 2)

        # Normalize the vector #
        self.final = self.velocities / np.linalg.norm(self.velocities)

        # Get the nearest valid action #
        # for action in np.argsort(np.linalg.norm(self.actions_directions - self.vector, axis=1)):
        #     new_position = self.agents_positions[self.observing_agent_index] + self.action_to_vector(action) * self.movement_length
        #     new_position = np.round(new_position).astype(int)
        #     if self.world[new_position[0], new_position[1]] == 1:
        #         self.action = action
        #         break
        # return self.action

        # Return Q-values in term of nearness to action #
        return 1/np.linalg.norm(self.actions_directions - self.final, axis=1)
    
    def action_to_vector(self, action):
        """ Convert the action to a vector """

        return self.actions_directions[action]
    
    def reset(self, _):
        """ Reset the state of the agent """
    
        self.max_local_measure = 0
        self.max_global_measure = 0
        self.best_local_location = None
        self.best_global_location = None
