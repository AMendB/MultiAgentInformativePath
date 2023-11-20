import torch
from torch import nn
from gym.spaces import Box
import numpy as np
from Algorithms.DRL.Networks.FeatureExtractors import FeatureExtractor

"""" Actor critic network """

class ActorCriticNetwork(nn.Module):

	""" Convolutional network for the critic. First, a 3 convolutional layers are applied to the input images, 
	then the output is flattened and passed through a 3 linear layer. The activation is ReLU for all layers.
	"""
	
	def __init__(self, obs_dim, n_actions: int, number_of_features = 1024):
		super(ActorCriticNetwork, self).__init__()

		# Create a Feature Extractor 
		self.feature_extractor = FeatureExtractor(obs_dim, number_of_features)

		# Create 3 sequential linear layer for the actor, with relu and softmax in the end
		self.actor = nn.Sequential(
			nn.Linear(number_of_features, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, n_actions),
			nn.Softmax(dim=1)
		)

		# Create 3 sequential linear layer for the critic, with relu and no activation in the end
		self.critic = nn.Sequential(
			nn.Linear(number_of_features, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 1)
		)

	def forward(self, x):
		# Apply the feature extractor
		x = self.feature_extractor(x)
		# Apply the actor
		pi = self.actor(x)
		# Apply the critic
		v = self.critic(x)
		return pi, v
	
	def get_action(self, x, deterministic=False, mask=None):
		""" Process action using a Categorical distribution. Return the action and the log probability of the action """

		# Get the action probabilities
		logits, value = self.forward(x)
		# Create a categorical distribution
		distribution = torch.distributions.Categorical(logits)
		# Sample an action from the distribution
		action = distribution.sample() if not deterministic else distribution.mode()
		
		# Return the action and the log probability of the action
		return action, distribution.log_prob(action), value, distribution.entropy()
	


	







