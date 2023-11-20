import pandas as pd

class MetricsDataCreator:

	def __init__(self, metrics_names, algorithm_name, experiment_name = "default_experiment", directory='./'):

		self.metrics_names = metrics_names
		self.algorithm_name = algorithm_name
		self.experiment_name = experiment_name
		if directory[-1] == '/':
			self.directory = directory
		else:
			self.directory = directory + '/'

		self.base_df = None
		self.data = []

	def save_step(self, run_num, step, metrics, algorithm_name = None):

		if algorithm_name is None:
			algorithm_name = self.algorithm_name

		""" Append the next step value of metrics """
		self.data.append([algorithm_name, run_num, step, *metrics]) #poe_x, pose_y, n_collisions, accumulated_reward, next_action):

	def save_data_as_csv(self):

		df = pd.DataFrame(data = self.data, columns=['Algorithm', 'Run', 'Step', *self.metrics_names])

		if self.base_df is None:
			df.to_csv(self.directory + self.experiment_name + '.csv', sep = ',')
			return df
		else:
			self.base_df = pd.concat((self.base_df, df), ignore_index = True)
			self.base_df.to_csv(self.directory + self.experiment_name + '.csv', sep = ',')
			return self.base_df
		
	def load_csv_to_add_info(self, path):

		self.base_df = pd.read_csv(path, sep=',')

	@staticmethod 
	def load_csv_as_df(path):

		return pd.read_csv(path, sep=',')







