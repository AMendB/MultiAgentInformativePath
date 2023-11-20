from typing import Dict, List, Tuple
from Environment.MonitoringEnvironment import MultiAgentMonitoring
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Algorithms.DRL.ReplayBuffers.ReplayBuffers import PrioritizedReplayBuffer, ReplayBuffer
from Algorithms.DRL.Networks.network import DuelingVisualNetwork, NoisyDuelingVisualNetwork, DistributionalVisualNetwork, DuelingVisualNetworkWithSensorsNoises
import torch.nn.functional as F
from tqdm import trange
from Algorithms.DRL.ActionMasking.ActionMaskingUtils import NoGoBackMasking, SafeActionMasking, ConsensusSafeActionMasking, NoGoBackFleetMasking
import time
import json
import os

class MultiAgentDuelingDQNAgent:

	def __init__(
			self,
			env: MultiAgentMonitoring, 
			memory_size: int,
			batch_size: int,
			target_update: int,
			soft_update: bool = False,
			tau: float = 0.0001,
			epsilon_values: List[float] = [1.0, 0.0],
			epsilon_interval: List[float] = [0.0, 1.0],
			learning_starts: int = 10,
			gamma: float = 0.99,
			lr: float = 1e-4,
			logdir=None,
			log_name="Experiment",
			save_every=None,
			train_every=1,
			masked_actions= False,
			concensus_actions= False,
			device='cpu',
			seed = 0,
			eval_every = None,
			eval_episodes = 1000,
			# PER parameters
			alpha: float = 0.2,
			beta: float = 0.6,
			prior_eps: float = 1e-6,
			# NN parameters #
			number_of_features: int = 1024,
			noisy: bool = False,
			# Distributional parameters #
			distributional: bool = False,
			num_atoms: int = 51,
			v_interval: Tuple[float, float] = (0.0, 100.0),
			# SensorNoisesNetwork parameters #
			network_with_sensornoises: bool = False,
			# n independent networks defined by sensors type parameters #
			independent_networks_by_sensors_type: bool = False,
	):
		"""

		:param env: Environment to optimize
		:param memory_size: Size of the experience replay
		:param batch_size: Mini-batch size for SGD steps
		:param target_update: Number of episodes between updates of the target
		:param soft_update: Flag to activate the Polyak update of the target
		:param tau: Polyak update constant
		:param gamma: Discount Factor
		:param lr: Learning Rate
		:param alpha: Randomness of the sample in the PER
		:param beta: Bias compensating constant in the PER weights
		:param prior_eps: Minimal probability for every experience to be samples
		:param number_of_features: Number of features after the visual extractor
		:param logdir: Directory to save the tensorboard log
		:param log_name: Name of the tb log
		"""

		""" Logging parameters """
		np.random.seed(seed)
		self.logdir = logdir
		self.experiment_name = log_name
		self.writer = None
		self.save_every = save_every
		self.eval_every = eval_every
		self.eval_episodes = eval_episodes

		""" Observation space dimensions """
		obs_dim = env.observation_space_shape
		action_dim = env.n_actions

		""" Agent embeds the environment """
		self.env = env
		self.batch_size = batch_size
		self.target_update = target_update
		self.soft_update = soft_update
		self.tau = tau
		self.gamma = gamma
		self.learning_rate = lr
		self.epsilon_values = epsilon_values
		self.epsilon_interval = epsilon_interval
		self.epsilon = self.epsilon_values[0]
		self.learning_starts = learning_starts
		self.train_every = train_every
		self.masked_actions = masked_actions
		self.concensus_actions = concensus_actions
		self.noisy = noisy
		self.distributional = distributional
		self.num_atoms = num_atoms
		self.v_interval = v_interval
		self.network_with_sensornoises = network_with_sensornoises
		self.independent_networks_by_sensors_type = independent_networks_by_sensors_type

		""" Automatic selection of the device """
		self.device = device
		torch.cuda.set_device(int(device[-1]))

		print("Selected device: ", self.device)

		""" Prioritized Experience Replay """
		if self.independent_networks_by_sensors_type:
			self.memory = [PrioritizedReplayBuffer(obs_dim, memory_size, batch_size, alpha=alpha) for _ in range(self.env.n_sensors_type)]
		else:
			self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, batch_size, alpha=alpha)
		self.beta = beta
		self.prior_eps = prior_eps

		""" Create the DQN and the DQN-Target (noisy if selected) """
		if self.network_with_sensornoises:
			self.dqn = DuelingVisualNetworkWithSensorsNoises(obs_dim, action_dim, number_of_features).to(self.device)
			self.dqn_target = DuelingVisualNetworkWithSensorsNoises(obs_dim, action_dim, number_of_features).to(self.device)
		elif self.independent_networks_by_sensors_type:
			self.dqn = [DuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device) for _ in range(self.env.n_sensors_type)]
			self.dqn_target = [DuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device) for _ in range(self.env.n_sensors_type)]
		elif self.noisy:
			self.dqn = NoisyDuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device)
			self.dqn_target = NoisyDuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device)
		elif self.distributional:
			self.support = torch.linspace(self.v_interval[0], self.v_interval[1], self.num_atoms).to(self.device)
			self.dqn = DistributionalVisualNetwork(obs_dim, action_dim, number_of_features, num_atoms, self.support).to(self.device)
			self.dqn_target = DistributionalVisualNetwork(obs_dim, action_dim, number_of_features, num_atoms, self.support).to(self.device)
		else:
			self.dqn = DuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device)
			self.dqn_target = DuelingVisualNetwork(obs_dim, action_dim, number_of_features).to(self.device)

		if self.independent_networks_by_sensors_type:
			# Load weights from dqn to dqn_target to be equal at the begining, and eval #
			for sensor_type in range(self.env.n_sensors_type):
				self.dqn_target[sensor_type].load_state_dict(self.dqn[sensor_type].state_dict())
				self.dqn_target[sensor_type].eval()
			""" Optimizers """
			self.optimizer = [optim.Adam(self.dqn[sensor_type].parameters(), lr=self.learning_rate) for sensor_type in range(self.env.n_sensors_type)]
		else:
			# Load weights from dqn to dqn_target to be equal at the begining, and eval #
			self.dqn_target.load_state_dict(self.dqn.state_dict())
			self.dqn_target.eval()
			""" Optimizer """
			self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)

		""" Actual list of transitions """
		self.transition = list()

		""" Evaluation flag """
		# self.is_eval = False

		""" Data for logging """
		self.episodic_reward = []
		self.episodic_loss = []
		self.episodic_length = []
		self.episode = 0

		# Sample new noisy parameters
		if self.noisy:
			self.dqn.reset_noise()
			self.dqn_target.reset_noise()

		# Masking utilities #
		if self.concensus_actions:
			self.consensus_safe_masking_module = ConsensusSafeActionMasking(navigation_map = self.env.scenario_map, action_space_dim = action_dim, movement_length = self.env.movement_length)
			self.nogobackfleet_masking_module = NoGoBackFleetMasking()
		elif self.masked_actions:
			self.safe_masking_module = SafeActionMasking(action_space_dim = action_dim, movement_length = self.env.movement_length)
			self.nogoback_masking_modules = {i: NoGoBackMasking() for i in range(self.env.n_agents)}

	# TODO: Implement an annealed Learning Rate (see:
	#  https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)

	def predict_action(self, state: np.ndarray, deterministic: bool = False):

		"""Select an action from the input state. If deterministic, no noise is applied. """

		if self.epsilon > np.random.rand() and not self.noisy and not deterministic:
			selected_action = self.env.action_space.sample()

		else:
			q_values = self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy()
			selected_action = np.argmax(q_values)

		return selected_action
	
	def select_actions(self, states: dict, deterministic: bool = False) -> dict:

		actions = {agent_id: self.predict_action(state, deterministic) for agent_id, state in states.items() }

		return actions

	def predict_masked_action(self, state: np.ndarray, agent_id: int, position: np.ndarray,  deterministic: bool = False):
		""" Select an action masked to avoid collisions and so """

		# Update the state of the safety module #
		self.safe_masking_module.update_state(position = position, new_navigation_map = state[0])

		if self.epsilon > np.random.rand() and not self.noisy and not deterministic:
			
			# Compute randomly the action #
			q_values, _ = self.safe_masking_module.mask_action(q_values = None)
			q_values, selected_action = self.nogoback_masking_modules[agent_id].mask_action(q_values = q_values)
			self.nogoback_masking_modules[agent_id].update_last_action(selected_action)

		else:
			q_values = self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy()
			q_values, _ = self.safe_masking_module.mask_action(q_values = q_values.flatten())
			q_values, selected_action = self.nogoback_masking_modules[agent_id].mask_action(q_values = q_values)
			self.nogoback_masking_modules[agent_id].update_last_action(selected_action)
		
		return selected_action

	def select_masked_actions(self, states: dict, positions: np.ndarray, deterministic: bool = False):

		actions = {agent_id: self.predict_masked_action(state=state, agent_id=agent_id, position=positions[agent_id], deterministic=deterministic) for agent_id, state in states.items()}

		return actions

	def select_concensus_actions(self, states: dict, norm_variance: np.ndarray, positions: np.ndarray, n_actions: int, done: dict, deterministic: bool = False):
		""" Select an action masked to avoid collisions and so """
		
		# Update map if dynamic #
		if self.env.dynamic:
			self.consensus_safe_masking_module.update_map(states[list(states.keys())[0]][0])

		if self.epsilon > np.random.rand() and not self.noisy and not deterministic:
			# Compute randomly the q's #
			q_values = {agent_id: np.random.rand(n_actions) for agent_id in states.keys() if not done[agent_id]}
		else:
			# The network compute the q's #
			if self.network_with_sensornoises:
				q_values = {agent_id: self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device), torch.FloatTensor([norm_variance[agent_id]]).unsqueeze(0).to(self.device)).detach().cpu().numpy().flatten() for agent_id, state in states.items() if not done[agent_id]}
			elif self.independent_networks_by_sensors_type:
				q_values = {agent_id: self.dqn[self.env.sensors_type[agent_id]](torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy().flatten() for agent_id, state in states.items() if not done[agent_id]}
			else:
				q_values = {agent_id: self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy().flatten() for agent_id, state in states.items() if not done[agent_id]}

		# Masking q's and take actions #
		q_values = self.nogobackfleet_masking_module.mask_actions(q_values=q_values)
		permanent_actions = self.consensus_safe_masking_module.query_actions(q_values=q_values, positions=positions)
		self.nogobackfleet_masking_module.update_previous_actions(permanent_actions)
		
		return permanent_actions

	def step(self, action: dict) -> Tuple[np.ndarray, np.float64, bool]:
		"""Take an action and return the response of the env."""

		next_state, reward, done = self.env.step(action)

		return next_state, reward, done

	def update_model(self, sensor_type_index = None) -> torch.Tensor:
		"""Update the model by gradient descent."""

		# PER needs beta to calculate weights
		if self.independent_networks_by_sensors_type:
			samples = self.memory[sensor_type_index].sample_batch(self.beta)
			weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
			indices = samples["indices"]

			# PER: importance sampling before average
			elementwise_loss = self._compute_dqn_loss(samples, sensor_type_index)
			loss = torch.mean(elementwise_loss * weights)

			# Compute gradients and apply them
			self.optimizer[sensor_type_index].zero_grad()
			loss.backward()
			self.optimizer[sensor_type_index].step()

			# PER: update priorities
			loss_for_prior = elementwise_loss.detach().cpu().numpy()
			new_priorities = loss_for_prior + self.prior_eps
			self.memory[sensor_type_index].update_priorities(indices, new_priorities)

			# Sample new noisy distribution
			if self.noisy:
				self.dqn[sensor_type_index].reset_noise()
				self.dqn_target[sensor_type_index].reset_noise()

			return loss.item()
		
		else:
			samples = self.memory.sample_batch(self.beta)
			weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
			indices = samples["indices"]

			# PER: importance sampling before average
			elementwise_loss = self._compute_dqn_loss(samples)
			loss = torch.mean(elementwise_loss * weights)

			# Compute gradients and apply them
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			# PER: update priorities
			loss_for_prior = elementwise_loss.detach().cpu().numpy()
			new_priorities = loss_for_prior + self.prior_eps
			self.memory.update_priorities(indices, new_priorities)

			# Sample new noisy distribution
			if self.noisy:
				self.dqn.reset_noise()
				self.dqn_target.reset_noise()

			return loss.item()

	@staticmethod
	def anneal_epsilon(p, p_init=0.1, p_fin=0.9, e_init=1.0, e_fin=0.0):

		if p < p_init:
			return e_init
		elif p > p_fin:
			return e_fin
		else:
			return (e_fin - e_init) / (p_fin - p_init) * (p - p_init) + 1.0

	@staticmethod
	def anneal_beta(p, p_init=0.1, p_fin=0.9, b_init=0.4, b_end=1.0):

		if p < p_init:
			return b_init
		elif p > p_fin:
			return b_end
		else:
			return (b_end - b_init) / (p_fin - p_init) * (p - p_init) + b_init

	def train(self, episodes):
		# TODO: Change for multiagent

		""" Train the agent. """

		if self.independent_networks_by_sensors_type:

			# Optimization steps #
			steps = 0
			
			# Create train logger and save configs #
			if self.writer is None:
				assert not os.path.exists(self.logdir), "El directorio ya existe. He evitado que se sobrescriba"
				if os.path.exists(self.logdir):
					self.logdir = self.logdir
				self.writer = [SummaryWriter(log_dir=self.logdir+f"/log{sensor_type}/", filename_suffix=f"_network{sensor_type}") for sensor_type in range(self.env.n_sensors_type)]
				self.write_experiment_config()
				self.env.save_environment_configuration(self.logdir if self.logdir is not None else './')

			# Agent in training mode #
			# self.is_eval = False
			# Reset episode count #
			self.episode = [1]*self.env.n_sensors_type
			# Reset metrics #
			episodic_reward_vector = [[]]*self.env.n_sensors_type
			record = [-np.inf]*self.env.n_sensors_type

			for episode in trange(1, int(episodes) + 1):

				done = {i:False for i in range(self.env.n_agents)}
				states = self.env.reset_env()
				score = [0]*self.env.n_sensors_type
				length = [0]*self.env.n_sensors_type
				losses = [[]]*self.env.n_sensors_type
				finished_episode_by_types = {i:False for i in range(self.env.n_sensors_type)}

				# Initially sample noisy policy #
				if self.noisy:
					for sensor_type in range(self.env.n_sensors_type):
						self.dqn[sensor_type].reset_noise()
						self.dqn_target[sensor_type].reset_noise()

				# PER: Increase beta temperature
				self.beta = self.anneal_beta(p=episode / episodes, p_init=0, p_fin=0.9, b_init=0.4, b_end=1.0)

				# Epsilon greedy annealing
				self.epsilon = self.anneal_epsilon(p=episode / episodes,
												p_init=self.epsilon_interval[0],
												p_fin=self.epsilon_interval[1],
												e_init=self.epsilon_values[0],
												e_fin=self.epsilon_values[1])

				# Run an episode #
				while not all(done.values()):

					# Increase the played steps #
					steps += 1

					# Select the action using the current policy #
					actions = self.select_concensus_actions(states=states, norm_variance=self.env.normalized_variance_sensormeasure, positions=self.env.get_active_agents_positions_dict(), n_actions=self.env.n_actions, done = done)

					# Process the agent step #
					next_states, reward, done = self.step(actions)

					# Store every observation for every agent #
					for agent_id in next_states.keys():
						self.transition = [states[agent_id],
											actions[agent_id],
											reward[agent_id],
											next_states[agent_id],
											done[agent_id],
											{}]

						self.memory[self.env.sensors_type[agent_id]].store(*self.transition)

					# Update the state
					states = next_states

					reward_array = np.array([*reward.values()])

					for sensor_type in range(self.env.n_sensors_type):
						if not(finished_episode_by_types[sensor_type]):
							# Accumulate indicators
							sensor_type_reward_array = reward_array[self.env.masks_by_type[sensor_type]]
							score[sensor_type] += np.mean(sensor_type_reward_array)  # The mean reward among types
							length[sensor_type] += 1

						
							# If episode ends for each agent type
							if self.env.dones_by_sensors_types[sensor_type] == True:
								self.episodic_reward = score[sensor_type]
								self.episodic_length = length[sensor_type]
								self.episode[sensor_type] += 1

								# Append loss metric #
								if losses[sensor_type]:
									self.episodic_loss = np.mean(losses[sensor_type])

								# Compute average metrics #
								episodic_reward_vector[sensor_type].append(self.episodic_reward)

								# Log progress
								self.log_data(sensor_type_index=sensor_type)

								# Save policy if is better on average
								mean_episodic_reward = np.mean(episodic_reward_vector[sensor_type][-50:])
								if mean_episodic_reward > record[sensor_type]:
									print(f"\nNew best policy with mean reward of {mean_episodic_reward} for network nº {sensor_type}")
									print("Saving model in " + self.logdir)
									record[sensor_type] = mean_episodic_reward
									self.save_model(name=f'BestPolicy_network{sensor_type}.pth', sensor_type_index=sensor_type)
								
								# Set episode for actual type as finished to not enter again #
								finished_episode_by_types[sensor_type] = True
			
							# If training is ready
							if len(self.memory[sensor_type]) >= self.batch_size and episode >= self.learning_starts:

								# Update model parameters by backprop-bootstrapping #
								if steps % self.train_every == 0:

									loss = self.update_model(sensor_type_index=sensor_type)
									# Append loss #
									losses[sensor_type].append(loss)

								# Update target soft/hard #
								if self.soft_update:
									self._target_soft_update(sensor_type_index=sensor_type)
								elif episode % self.target_update == 0 and all(done.values()):
									self._target_hard_update(sensor_type_index=sensor_type)

				if self.save_every is not None:
					if episode % self.save_every == 0:
						for sensor_type in range(self.env.n_sensors_type):
							self.save_model(name=f'Episode_{episode}_Policy_network{sensor_type}.pth', sensor_type_index=sensor_type)

				# Reset previous actions of NoGoBack #
				self.nogobackfleet_masking_module.reset()

				# Evaluation #
				if self.eval_every is not None:
					if episode % self.eval_every == 0:
						for sensor_type in range(self.env.n_sensors_type):
							mean_reward, mean_length = self.evaluate_env(self.eval_episodes)
							self.writer[sensor_type].add_scalar('test/accumulated_reward', mean_reward[sensor_type], self.episode[sensor_type])
							self.writer[sensor_type].add_scalar('test/accumulated_length', mean_length[sensor_type], self.episode[sensor_type])

			# Save the final policys #
			for sensor_type in range(self.env.n_sensors_type):
				self.save_model(name=f'Final_Policy_network{sensor_type}.pth', sensor_type_index=sensor_type)

		else:

			# Optimization steps #
			steps = 0
			
			# Create train logger and save configs #
			if self.writer is None:
				assert not os.path.exists(self.logdir), "El directorio ya existe. He evitado que se sobrescriba"
				if os.path.exists(self.logdir):
					self.logdir = self.logdir
				self.writer = SummaryWriter(log_dir=self.logdir, filename_suffix=self.experiment_name)
				self.write_experiment_config()
				self.env.save_environment_configuration(self.logdir if self.logdir is not None else './')

			# Agent in training mode #
			# self.is_eval = False
			# Reset episode count #
			self.episode = 1
			# Reset metrics #
			episodic_reward_vector = []
			record = -np.inf

			for episode in trange(1, int(episodes) + 1):

				done = {i:False for i in range(self.env.n_agents)}
				states = self.env.reset_env()
				score = 0
				length = 0
				losses = []

				# Initially sample noisy policy #
				if self.noisy:
					self.dqn.reset_noise()
					self.dqn_target.reset_noise()

				# PER: Increase beta temperature
				self.beta = self.anneal_beta(p=episode / episodes, p_init=0, p_fin=0.9, b_init=0.4, b_end=1.0)

				# Epsilon greedy annealing
				self.epsilon = self.anneal_epsilon(p=episode / episodes,
												p_init=self.epsilon_interval[0],
												p_fin=self.epsilon_interval[1],
												e_init=self.epsilon_values[0],
												e_fin=self.epsilon_values[1])

				# Run an episode #
				while not all(done.values()):

					# Increase the played steps #
					steps += 1

					# Select the action using the current policy #
					if self.concensus_actions:
						actions = self.select_concensus_actions(states=states, norm_variance=self.env.normalized_variance_sensormeasure, positions=self.env.get_active_agents_positions_dict(), n_actions=self.env.n_actions, done = done)
					elif self.masked_actions:
						actions = self.select_masked_actions(states=states, positions=self.env.fleet.get_positions())
					else:
						actions = self.select_actions(states)
						actions = {agent_id: action for agent_id, action in actions.items() if not done[agent_id]} # only active agents

					# Process the agent step #
					next_states, reward, done = self.step(actions)

					# Store every observation for every agent #
					for agent_id in next_states.keys():
						self.transition = [states[agent_id],
											actions[agent_id],
											reward[agent_id],
											next_states[agent_id],
											done[agent_id],
											{"norm_variance": self.env.normalized_variance_sensormeasure[agent_id]} if self.network_with_sensornoises else {}]

						self.memory.store(*self.transition)

					# Update the state
					states = next_states
					# Accumulate indicators
					score += np.mean(list(reward.values()))  # The mean reward among the agents
					length += 1

					# if episode ends
					if all(done.values()):

						# Append loss metric #
						if losses:
							self.episodic_loss = np.mean(losses)

						# Compute average metrics #
						self.episodic_reward = score
						self.episodic_length = length
						episodic_reward_vector.append(self.episodic_reward)
						self.episode += 1

						# Log progress
						self.log_data()

						# Save policy if is better on average
						mean_episodic_reward = np.mean(episodic_reward_vector[-50:])
						if mean_episodic_reward > record:
							print(f"\nNew best policy with mean reward of {mean_episodic_reward}")
							print("Saving model in " + self.writer.log_dir)
							record = mean_episodic_reward
							self.save_model(name='BestPolicy.pth')

					# If training is ready
					if len(self.memory) >= self.batch_size and episode >= self.learning_starts:

						# Update model parameters by backprop-bootstrapping #
						if steps % self.train_every == 0:

							loss = self.update_model()
							# Append loss #
							losses.append(loss)

						# Update target soft/hard #
						if self.soft_update:
							self._target_soft_update()
						elif episode % self.target_update == 0 and all(done.values()):
							self._target_hard_update()

				if self.save_every is not None:
					if episode % self.save_every == 0:
						self.save_model(name=f'Episode_{episode}_Policy.pth')

				# Reset previous actions of NoGoBack #
				self.nogobackfleet_masking_module.reset()

				# Evaluation #
				if self.eval_every is not None:
					if episode % self.eval_every == 0:
						mean_reward, mean_length = self.evaluate_env(self.eval_episodes)
						self.writer.add_scalar('test/accumulated_reward', mean_reward, self.episode)
						self.writer.add_scalar('test/accumulated_length', mean_length, self.episode)

			# Save the final policy #
			self.save_model(name='Final_Policy.pth')

	def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], sensor_type_index = None) -> torch.Tensor:

		"""Return dqn loss."""
		device = self.device  # for shortening the following lines
		states = torch.FloatTensor(samples["obs"]).to(device)
		next_states = torch.FloatTensor(samples["next_obs"]).to(device)
		actions = torch.LongTensor(samples["acts"]).to(device)
		rewards = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
		dones = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

		# G_t   = r + gamma * v(s_{t+1})  if state != Terminal
		#       = r                       otherwise

		if self.network_with_sensornoises:

			norm_variances = torch.FloatTensor([dict['norm_variance'] for dict in samples['info']]).to(device)

			actions = actions.reshape(-1, 1)
			curr_q_value = self.dqn(states, norm_variances).gather(1, actions)
			dones_mask = 1 - dones

			with torch.no_grad():
				next_q_value = self.dqn_target(next_states, norm_variances).gather(1, self.dqn(next_states, norm_variances).argmax(dim=1, keepdim=True))
				target = (rewards + self.gamma * next_q_value * dones_mask).to(self.device)

			# calculate element-wise dqn loss
			elementwise_loss = F.mse_loss(curr_q_value, target, reduction="none")

		elif self.independent_networks_by_sensors_type:
			actions = actions.reshape(-1, 1)
			curr_q_value = self.dqn[sensor_type_index](states).gather(1, actions)
			dones_mask = 1 - dones

			with torch.no_grad():
				next_q_value = self.dqn_target[sensor_type_index](next_states).gather(1, self.dqn[sensor_type_index](next_states).argmax(dim=1, keepdim=True))
				target = (rewards + self.gamma * next_q_value * dones_mask).to(self.device)

			# calculate element-wise dqn loss
			elementwise_loss = F.mse_loss(curr_q_value, target, reduction="none")

		elif self.distributional:
			# Distributional Q-Learning - Here is where the fun begins #
			delta_z = float(self.v_interval[1] - self.v_interval[0]) / (self.num_atoms - 1)

			with torch.no_grad():

				# max_a = argmax_a' Q'(s',a')
				next_action = self.dqn_target(next_states).argmax(1)
				# V'(s', max_a)
				next_dist = self.dqn_target.dist(next_states)
				next_dist = next_dist[range(self.batch_size), next_action]

				# Compute the target distribution by adding the
				t_z = rewards + (1 - dones) * self.gamma * self.support
				t_z = t_z.clamp(min=self.v_interval[0], max=self.v_interval[1])
				b = (t_z - self.v_interval[0]) / delta_z
				lower_bound = b.floor().long()
				upper_bound = b.ceil().long()

				offset = (torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size
					).long()
					.unsqueeze(1)
					.expand(self.batch_size, self.num_atoms)
					.to(self.device)
				)

				proj_dist = torch.zeros(next_dist.size(), device=self.device)
				proj_dist.view(-1).index_add_(
					0, (lower_bound + offset).view(-1), (next_dist * (upper_bound.float() - b)).view(-1)
				)
				proj_dist.view(-1).index_add_(
					0, (upper_bound + offset).view(-1), (next_dist * (b - lower_bound.float())).view(-1)
				)

			dist = self.dqn.dist(states)
			log_p = torch.log(dist[range(self.batch_size), actions])

			elementwise_loss = -(proj_dist * log_p).sum(1)

		else:
			actions = actions.reshape(-1, 1)
			curr_q_value = self.dqn(states).gather(1, actions)
			dones_mask = 1 - dones

			with torch.no_grad():
				next_q_value = self.dqn_target(next_states).gather(1, self.dqn(next_states).argmax(dim=1, keepdim=True))
				target = (rewards + self.gamma * next_q_value * dones_mask).to(self.device)

			# calculate element-wise dqn loss
			elementwise_loss = F.mse_loss(curr_q_value, target, reduction="none")

		return elementwise_loss

	def _target_hard_update(self, sensor_type_index=None):
		"""Hard update: target <- local."""

		if self.independent_networks_by_sensors_type:
			print(f"Hard update performed at episode {self.episode[sensor_type_index]}!")
			self.dqn_target[sensor_type_index].load_state_dict(self.dqn[sensor_type_index].state_dict())
		else:
			print(f"Hard update performed at episode {self.episode}!")
			self.dqn_target.load_state_dict(self.dqn.state_dict())

	def _target_soft_update(self, sensor_type_index=None):
		"""Soft update: target_{t+1} <- local * tau + target_{t} * (1-tau)."""

		if self.independent_networks_by_sensors_type:
			for target_param, local_param in zip(self.dqn_target[sensor_type_index].parameters(), self.dqn_target[sensor_type_index].parameters()):
				target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
		else:
			for target_param, local_param in zip(self.dqn_target.parameters(), self.dqn_target.parameters()):
				target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

	def log_data(self, sensor_type_index=None):
		""" Logs for tensorboard """

		if self.independent_networks_by_sensors_type:
			if self.episodic_loss:
				self.writer[sensor_type_index].add_scalar('train/loss', self.episodic_loss, self.episode[sensor_type_index])

			self.writer[sensor_type_index].add_scalar('train/epsilon', self.epsilon, self.episode[sensor_type_index])
			self.writer[sensor_type_index].add_scalar('train/beta', self.beta, self.episode[sensor_type_index])

			self.writer[sensor_type_index].add_scalar('train/accumulated_reward', self.episodic_reward, self.episode[sensor_type_index])
			self.writer[sensor_type_index].add_scalar('train/accumulated_length', self.episodic_length, self.episode[sensor_type_index])

			self.writer[sensor_type_index].flush()

		else:
			if self.episodic_loss:
				self.writer.add_scalar('train/loss', self.episodic_loss, self.episode)

			self.writer.add_scalar('train/epsilon', self.epsilon, self.episode)
			self.writer.add_scalar('train/beta', self.beta, self.episode)

			self.writer.add_scalar('train/accumulated_reward', self.episodic_reward, self.episode)
			self.writer.add_scalar('train/accumulated_length', self.episodic_length, self.episode)

			self.writer.flush()

	def load_model(self, path_to_file):
		
		if self.independent_networks_by_sensors_type:
			for sensor_type in range(self.env.n_sensors_type):
				self.dqn[sensor_type].load_state_dict(torch.load(path_to_file[:-4] + f'_network{sensor_type}.pth', map_location=self.device))
		else:
			self.dqn.load_state_dict(torch.load(path_to_file, map_location=self.device))

	def save_model(self, name='experiment.pth', sensor_type_index=None):
		
		if self.independent_networks_by_sensors_type:
			try:
				torch.save(self.dqn[sensor_type_index].state_dict(), self.logdir + '/' + name)
			except:
				print(f"Model {name} saving failed!")
				try:
					time.sleep(3)
					torch.save(self.dqn[sensor_type_index].state_dict(), self.logdir + '/' + name)
					print("Done!")
				except:
					print('Definitely not saved :(')
		else:
			try:
				torch.save(self.dqn.state_dict(), self.writer.log_dir + '/' + name)
			except:
				print("Model saving failed!")
				try:
					time.sleep(3)
					torch.save(self.dqn.state_dict(), self.writer.log_dir + '/' + name)
					print("Done!")
				except:
					print('Definitely not saved :(')

	def evaluate_env(self, eval_episodes, render=False):
		""" Evaluate the agent on the environment for a given number of episodes with a deterministic policy """
		if self.independent_networks_by_sensors_type:
			total_reward = np.array([0]*self.env.n_sensors_type)
			total_length = np.array([0]*self.env.n_sensors_type)
			
			# Set networks to eval #
			for sensor_type in range(self.env.n_sensors_type):
				self.dqn[sensor_type].eval()

			for _ in trange(eval_episodes):

				# Reset the environment #
				states = self.env.reset_env()
				if render:
					self.env.render()
				done = {agent_id: False for agent_id in range(self.env.n_agents)}
				

				while not all(done.values()):


					# Select the action using the current policy
					if self.concensus_actions:
						actions = self.select_concensus_actions(states=states, norm_variance=self.env.normalized_variance_sensormeasure, positions=self.env.get_active_agents_positions_dict(), n_actions=self.env.n_actions, done = done)
					elif self.masked_actions:
						actions = self.select_masked_actions(states=states, positions=self.env.fleet.get_positions())
					else:
						actions = self.select_actions(states)
						actions = {agent_id: action for agent_id, action in actions.items() if not done[agent_id]} # only active agents

					# Process the agent step #
					next_state, reward, done = self.step(actions)

					if render:
						self.env.render()

					# Update the state #
					state = next_state
					
					reward_array = np.array([*reward.values()])

					for sensor_type in range(self.env.n_sensors_type):
						if not(self.env.dones_by_sensors_types[sensor_type]):
							total_length[sensor_type] += 1
							sensor_type_reward_array = reward_array[self.env.masks_by_type[sensor_type]]
							total_reward[sensor_type] += np.sum(sensor_type_reward_array)
				
				# Reset previous actions of NoGoBack #
				self.nogobackfleet_masking_module.reset()

			# Set networks to train #
			for sensor_type in range(self.env.n_sensors_type):
				self.dqn[sensor_type].train()

		else:
			self.dqn.eval()
			total_reward = 0
			total_length = 0

			for _ in trange(eval_episodes):

				# Reset the environment #
				states = self.env.reset_env()
				if render:
					self.env.render()
				done = {agent_id: False for agent_id in range(self.env.n_agents)}
				

				while not all(done.values()):

					total_length += 1

					# Select the action using the current policy
					if self.concensus_actions:
						actions = self.select_concensus_actions(states=states, norm_variance=self.env.normalized_variance_sensormeasure, positions=self.env.get_active_agents_positions_dict(), n_actions=self.env.n_actions, done = done)
					elif self.masked_actions:
						actions = self.select_masked_actions(states=states, positions=self.env.fleet.get_positions())
					else:
						actions = self.select_actions(states)
						actions = {agent_id: action for agent_id, action in actions.items() if not done[agent_id]} # only active agents

					# Process the agent step #
					next_state, reward, done = self.step(actions)

					if render:
						self.env.render()

					# Update the state #
					state = next_state
					
					total_reward += np.sum(list(reward.values()))
				
				# Reset previous actions of NoGoBack #
				self.nogobackfleet_masking_module.reset()

			self.dqn.train()

		# Return the average reward, average length
		return total_reward / eval_episodes, total_length / eval_episodes

	def write_experiment_config(self):
		""" Write experiment and environment variables in a json file """

		self.experiment_config = {
			"save_every": self.save_every,
			"eval_every": self.eval_every,
			"eval_episodes": self.eval_episodes,
			"batch_size": self.batch_size,
			"gamma": self.gamma,
			"tau": self.tau,
			"lr": self.learning_rate,
			"epsilon": self.epsilon,
			"epsilon_values": self.epsilon_values,
			"epsilon_interval": self.epsilon_interval,
			"beta": self.beta,
			"num_atoms": self.num_atoms,
			"masked_actions": self.masked_actions,
			"concensus_actions": self.concensus_actions,
			"network_with_sensornoises": self.network_with_sensornoises,
			"independent_networks_by_sensors_type": self.independent_networks_by_sensors_type
		}

		with open(self.logdir + '/experiment_config.json', 'w') as f:

			json.dump(self.experiment_config, f, indent=4)
