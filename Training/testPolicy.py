import sys
import json
sys.path.append('.')

from Environment.MonitoringEnvironment import MultiAgentMonitoring
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np

path_to_training_folder = 'DoneTrainings/runs_4A - Último (ponderado)/Alg_Network_RW_Influence_10_0/'

f = open(path_to_training_folder + 'environment_config.json',)
env_config = json.load(f)
f.close()

scenario_map = np.array(env_config['scenario_map'])
n_agents = env_config['number_of_agents'] # 1 #
reward_function = env_config['reward_function']
reward_weights = tuple(env_config['reward_weights'])

env = MultiAgentMonitoring(scenario_map=scenario_map,
					number_of_agents=n_agents,
					max_distance_travelled=env_config['max_distance_travelled'],
					mean_sensormeasure=np.array(env_config['mean_sensormeasure']),
					range_std_sensormeasure=tuple(env_config['range_std_sensormeasure']),
					std_sensormeasure= np.array([0.05, 0.10, 0.20, 0.40])[:n_agents], #np.array(env_config['std_sensormeasure']),#
					fleet_initial_positions=env_config['fleet_initial_positions'], #np.array(env_config['fleet_initial_positions']), #
					seed=3,
					movement_length=env_config['movement_length'],
					influence_length=env_config['influence_length'],
					flag_to_check_collisions_within=env_config['flag_to_check_collisions_within'],
					max_collisions=env_config['max_collisions'],
					reward_function=reward_function,
					ground_truth_type=env_config['ground_truth_type'],
					peaks_location='Random',
					dynamic=env_config['dynamic'],
					obstacles=env_config['obstacles'],
					regression_library=env_config['regression_library'],
					reward_weights=reward_weights,
					scale_kernel=env_config['scale_kernel'],
					show_plot_graphics=True,
                                        )

network = MultiAgentDuelingDQNAgent(env=env,
						memory_size=int(1E3),  #int(1E6), 1E5
						batch_size=64,
						target_update=1000,
						seed = 3,
						concensus_actions=True,
						device='cuda:0',
						network_with_sensornoises = True,
						independent_networks_by_sensors_type = False,
						)

network.load_model(path_to_training_folder + 'BestPolicy.pth')

results = network.evaluate_env(10, render=True)

print(results)

