import numpy as np

class LawnMowerAgent:

    def __init__(self, world: np.ndarray, number_of_actions: int, movement_length: int, forward_direction: int, seed=0):

        """ Finite State Machine that represents a lawn mower agent. """
        self.world = world
        self.action = None
        self.number_of_actions = number_of_actions
        self.move_length = movement_length
        self.state = 'FORWARD'
        self.turn_count = 0
        self.initial_action = forward_direction
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

    
    def compute_obstacles(self, position):
        # Compute if there is an obstacle or reached the border #
        OBS = position[0] < 0 or position[0] >= self.world.shape[0] or position[1] < 0 or position[1] >= self.world.shape[1]
        if not OBS:
            OBS = OBS or self.world[position[0], position[1]] == 0

        return OBS


    def move(self, actual_position):
        """ Compute the new state """

        # Compute the new position #
        new_position = actual_position + self.action_to_vector(self.state_to_action(self.state)) * self.move_length 
        # Compute if there is an obstacle or reached the border #
        OBS = self.compute_obstacles(new_position)

        if self.state == 'FORWARD':
            
            if not OBS:
                self.state = 'FORWARD'
            else:

                # Check if with the new direction there is an obstacle #
                new_position = actual_position + self.action_to_vector(self.state_to_action('TURN')) * self.move_length
                OBS = self.compute_obstacles(new_position)

                if not OBS:
                    self.state = 'TURN'
                else:
                    self.state = 'RECEED'

        elif self.state == 'RECEED':
            # Stay in receed state until there is no obstacle #

            # Check if with the new direction there is an obstacle #
            new_position = actual_position + self.action_to_vector(self.state_to_action('TURN')) * self.move_length
            OBS = self.compute_obstacles(new_position)
            if OBS:
                self.state = 'RECEED'
            else:
                self.state = 'TURN'

        elif self.state == 'TURN':

            if self.turn_count == 1 or OBS:
                self.state = 'REVERSE'
                self.turn_count = 0
            else:
                self.state = 'TURN'
                self.turn_count += 1

        elif self.state == 'REVERSE':

            if not OBS:
                self.state = 'REVERSE'
            else:

                # Check if with the new direction there is an obstacle #
                new_position = actual_position + self.action_to_vector(self.state_to_action('TURN2')) * self.move_length
                OBS = self.compute_obstacles(new_position)

                if not OBS:
                    self.state = 'TURN2'
                else:
                    self.state = 'RECEED2'

        elif self.state == 'RECEED2':
            # Stay in receed state until there is no obstacle #
            new_position = actual_position + self.action_to_vector(self.state_to_action('TURN2')) * self.move_length
            OBS = self.compute_obstacles(new_position)
            if OBS:
                self.state = 'RECEED2'
            else:
                self.state = 'TURN2'

        elif self.state == 'TURN2':
                
                if self.turn_count == 1 or OBS:
                    self.state = 'FORWARD'
                    self.turn_count = 0
                else:
                    self.state = 'TURN2'
                    self.turn_count += 1

        # Compute the new position #
        new_position = actual_position + self.action_to_vector(self.state_to_action(self.state)) * self.move_length 
        # Compute if there is an obstacle or reached the border #
        OBS = self.compute_obstacles(new_position)

        if OBS:
            self.initial_action = self.perpendicular_action(self.initial_action)
            self.state = 'FORWARD'
        
        return self.state_to_action(self.state)
    
    def state_to_action(self, state):

        if state == 'FORWARD':
            return self.initial_action
        elif state == 'TURN':
            return self.perpendicular_action(self.initial_action)
        elif state == 'REVERSE':
            return self.opposite_action(self.initial_action)
        elif state == 'TURN2':
            return self.perpendicular_action(self.initial_action)
        elif state == 'RECEED':
            return self.opposite_action(self.initial_action)
        elif state == 'RECEED2':
            return self.initial_action

    def action_to_vector(self, action):
        """ Transform an action to a vector """

        vectors = np.round(np.array([[np.cos(2*np.pi*i/self.number_of_actions), np.sin(2*np.pi*i/self.number_of_actions)] for i in range(self.number_of_actions)]))

        return vectors[action].astype(int)
    
    def perpendicular_action(self, action):
        """ Compute the perpendicular action """
        return (action - self.number_of_actions//4) % self.number_of_actions
    
    def opposite_action(self, action):
        """ Compute the opposite action """
        return (action + self.number_of_actions//2) % self.number_of_actions
    
    def reset(self, initial_action):
        """ Reset the state of the agent """
        self.state = 'FORWARD'
        self.initial_action = initial_action
        self.turn_count = 0
