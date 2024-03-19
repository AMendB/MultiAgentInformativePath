import numpy as np

class EGreedyAgent:
    def __init__(self, env, rw_fn, rw_weights) -> None:
        self.env = env
        self.reward_function = rw_fn
        self.reward_weights = rw_weights
        self.epsilon = 0.1
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

        # Import get_reward function from MonitoringEnvirnoment.py
        from Environment.MonitoringEnvironment.MultiAgentMonitoring import get_reward
        if self.reward_function == 'Influence_area_changes_model':
            from Environment.MonitoringEnvironment import get_reward_influence_area_changes_model as get_reward
        elif self.reward_function == 'Position_changes_model':
            from Environment.MonitoringEnvironment import get_reward_position_changes_model as get_reward
        elif self.reward_function == 'Error_with_model':
            from Environment.MonitoringEnvironment import get_reward_error_with_model as get_reward
        
    def move(self, state):
        pass
        