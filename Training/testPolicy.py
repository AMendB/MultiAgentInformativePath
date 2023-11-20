import sys
sys.path.append('.')

from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np


sc_map = np.genfromtxt('Environment/Maps/example_map.csv', delimiter=',')
initial_positions = np.asarray([[24, 21],
                                [28, 24],
                                [27, 19],
                                [24, 24]])

env = MultiAgentPatrolling(scenario_map=sc_map,
	                           fleet_initial_positions=initial_positions,
	                           distance_budget=150,
	                           number_of_vehicles=len(initial_positions),
	                           seed=10,
							   miopic=True,
	                           detection_length=2,
	                           movement_length=2,
	                           max_collisions=10,
	                           forget_factor=0.5,
	                           attrittion=0.1,
	                           networked_agents=False,
							   reward_type='model_changes',
							   ground_truth_type='algae_bloom',
	                           obstacles=True,
                               frame_stacking = 1,
                               state_index_stacking = (2,3,4))

multiagent = MultiAgentDuelingDQNAgent(env=env,
									   memory_size=int(1E3),
									   batch_size=64,
									   target_update=1000,
									   soft_update=False,
									   tau=0.0001,
									   epsilon_values=[1.0, 0.1],
									   epsilon_interval=[0.0, 0.33],
									   learning_starts=0,
									   gamma=0.99,
									   lr=1e-4,
									   noisy=False,
									   train_every=10000,
									   save_every=5000,
									   distributional=False,
									   masked_actions=True)


multiagent.load_model('Evaluation/Episode_45000_Policy.pth')

results = multiagent.evaluate_env(10, render=True)

print(results)

