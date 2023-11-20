import sys
sys.path.append('.')

from Environment.MonitoringEnvironment import MultiAgentMonitoring
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np

# Selection of PARAMETERS TO TRAIN #
reward_function = 'Influence_area_changes_model' # Position_changes_model, Influence_area_changes_model, Error_with_model
reward_weights = (10, 0) #(1.0, 0.1)
memory_size = int(1E6)
network_type = 'network_with_sensornoises' # network_with_sensornoises, independent_networks_by_sensors_type
device = 'cuda:0'
episodes = 150000
n_agents = 4  # max 4







SHOW_PLOT_GRAPHICS = False
seed = 0

# Agents and sensors info #
movement_length = 2
influence_length = 6
mean_sensormeasure = np.array([0, 0, 0, 0])[:n_agents] # mean of the measure of every agent
range_std_sensormeasure = (1*0.5/100, 1*0.5*100/100) # AML is "the best", from then on 100 times worse
random_std = True
if random_std:
	std_sensormeasure = 'random' # std of the measure of every agent
else:
	std_sensormeasure = np.array([0.1, 0.25, 0.1, 0.25])[:n_agents] # std of the measure of every agent


scenario_map = np.genfromtxt('Environment/Maps/ypacarai_map_low_res.csv', delimiter=',')

# Set initial positions #
random_initial_positions = True
if random_initial_positions:
	initial_positions = 'fixed'
else:
	initial_positions = np.array([[46, 28], [46, 31], [49, 28], [49, 31]])[:n_agents, :]
	
# Create environment # 
env = MultiAgentMonitoring(scenario_map=scenario_map,
							number_of_agents=n_agents,
							max_distance_travelled=100,
							mean_sensormeasure=mean_sensormeasure,
							range_std_sensormeasure=range_std_sensormeasure,
							std_sensormeasure=std_sensormeasure,
							fleet_initial_positions=initial_positions,
							seed=seed,
							movement_length=movement_length,
							influence_length=influence_length,
							flag_to_check_collisions_within=True,
							max_collisions=10,
							reward_function=reward_function,
							ground_truth_type='shekel',
							dynamic=False,
							obstacles=False,
							regression_library='gpytorch',  # scikit, gpytorch
				 			scale_kernel = True,
							reward_weights=reward_weights, 
							show_plot_graphics=SHOW_PLOT_GRAPHICS,
							)

# Network config:
if network_type == 'network_with_sensornoises':
	network_with_sensornoises = True
	independent_networks_by_sensors_type = False
elif network_type == 'independent_networks_by_sensors_type':
	network_with_sensornoises = False
	independent_networks_by_sensors_type = True
if memory_size == int(1E3):
	logdir = f'testing/Training_{network_type.split("_")[0]}_RW_{reward_function.split("_")[0]}_' + '_'.join(map(str, reward_weights))
else:
	logdir = f'Training/runs_{n_agents}A/Alg_{network_type.split("_")[0].capitalize()}_RW_{reward_function.split("_")[0]}_' + '_'.join(map(str, reward_weights))
network = MultiAgentDuelingDQNAgent(env=env,
									memory_size=memory_size,  #int(1E6), 1E5
									batch_size=128,
									target_update=1000,
									soft_update=True,
									tau=0.001,
									epsilon_values=[1.0, 0.05],
									epsilon_interval=[0.0, 0.5], #0.33
									learning_starts=100, 
									gamma=0.99,
									lr=1e-4,
									save_every=5000, # 5000
									train_every=15,
									masked_actions=False,
									concensus_actions=True,
									device=device,
									logdir=logdir,
									eval_episodes=50, # 10
									eval_every=1000, #1000
									noisy=False,
									distributional=False,
									network_with_sensornoises = network_with_sensornoises,
									independent_networks_by_sensors_type = independent_networks_by_sensors_type,
)

network.train(episodes=episodes) #10000