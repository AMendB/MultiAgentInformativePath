import sys
sys.path.append('.')
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt

# for _ in range(100):
#     scenario_map = np.genfromtxt('Environment/Maps/ypacarai_map_low_res.csv', delimiter=',')
#     deployment_positions = np.zeros_like(scenario_map)
#     deployment_positions[slice(45,50), slice(27,32)] = 1
#     n_agents = 4

#     initial_positions = np.argwhere(deployment_positions == 1)[np.random.choice(len(np.argwhere(deployment_positions == 1)), n_agents, replace=False)]
    
#     scenario_map[initial_positions[:,0], initial_positions[:,1]] = 0
    
#     plt.imshow(scenario_map)
#     plt.show()







# from PIL import Image
# import os
# import cv2

# # Function to crop an image
# def crop_image(image, x, y, width, high):
#     return image[y:y+high, x:x+width]

# saving_paths = ['Experiments/Results/INBST.2.Influence_area_changes_model_5_5', 
#                 'Experiments/Results/INBST.2.Influence_area_changes_model_10_0',
#                 'Experiments/Results/NWSN.2.Influence_area_changes_model_5_5',
#                 'Experiments/Results/NWSN.2.Influence_area_changes_model_10_0',
#                 ]
# networks = set([path.split('/')[-1].split('.')[0] for path in saving_paths])
# collage = []
# for net in networks:
#     paths = [path for path in saving_paths if net in path]
#     images_paths = [next(os.path.join(path, file) for file in os.listdir(path) if file.startswith('AverageMetrics') and file.endswith('png')) for path in paths]
#     collage.append(np.hstack([crop_image(cv2.imread(img), 100, 0, 1580, 880) for img in images_paths]))

# collage = np.vstack(collage)
# cv2.imwrite('Experiments/Results/Collage.png', collage)

# cv2.imshow("Collage", collage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import os
import cv2
images = []
peaks_positions = ['Random', 'Upper', 'MiddleLeft', 'MiddleRight', 'Middle', 'Bottom']
path = 'Experiments/A/[0.005 0.005 0.05 0.05]'
name = 'Network.4.Influence_10_0'
# path = 'Experiments/A/0.005 vs 0.05 vs 0.5'
# name = '.png'
saving_paths = [f'{path} {positions} {name}' for positions in peaks_positions]

images_paths = [next(os.path.join(path, file) for file in os.listdir(path) if 'Heatmaps' in file) for path in saving_paths]
images = [cv2.imread(img) for img in images_paths]
collage = np.vstack(images)
cv2.imwrite(path + ' ALL_POSITIONS HeatmapsAverage100eps_4A.png', collage)







# from Evaluation.Utils.metrics_wrapper import MetricsDataCreator

# metrics_path = 'Experiments/Results/2oTrainingInfluence2A/INBST.2.Influence_area_changes_model_5_5/metrics.csv'
# metrics_df = MetricsDataCreator.load_csv_as_df(metrics_path)
# run = np.unique(metrics_df['Run'])

# numeric_columns = metrics_df.select_dtypes(include=[np.number])








# from Environment.GroundTruthsModels.ShekelGroundTruth import GroundTruth
# from GaussianProcess.GPModels import GaussianProcessScikit, GaussianProcessGPyTorch
# import torch
# from sklearn.metrics import mean_squared_error
# mean = 0
# std = 0.005
# variance = std**2

# scenario_map = np.genfromtxt('Environment/Maps/ypacarai_map_low_res.csv', delimiter=',')
# non_water_mask = scenario_map != 1
# visitable_locations = np.vstack(np.where(scenario_map != 0)).T 

# ground_truth = GroundTruth(scenario_map, max_number_of_peaks = 4, is_bounded = True, seed = 3, peaks_location='Random')
# gaussian_process = GaussianProcessGPyTorch(scenario_map = scenario_map, initial_lengthscale = 5.0, kernel_bounds = (0.1, 20), training_iterations = 50, scale_kernel=True, device = 'cuda' if torch.cuda.is_available() else 'cpu')
# max_std_scale = 0.19 # empirically selectioned to scale uncertainty map to obtain more contrast and higher rewards differences
# mse = [[], [], [], [], []]
# for _ in range(100):
#     ground_truth.reset()
#     for grid_size in [2, 3, 4, 5, 6]:
#         gaussian_process.reset()
#         grid_map = np.zeros_like(scenario_map)
#         grid_map[1::grid_size,1::grid_size] = 1
#         grid_map[non_water_mask] = 0 
#         position_measures = np.vstack(np.where(grid_map != 0)).T 
        
#         new_measures = np.clip([ground_truth.read()[pose_x, pose_y] + np.random.normal(mean, std) for (pose_x, pose_y) in position_measures], 0, 1)
#         variance_of_measures = [variance for _ in range(len(position_measures))]
#         gaussian_process.fit_gp(X_new=position_measures, y_new=new_measures, variances_new=variance_of_measures)
#         model_mean_map, model_uncertainty_map = gaussian_process.predict_gt()
#         model_mean_map = np.clip( model_mean_map, 0, 1 )
#         model_uncertainty_map = np.clip( model_uncertainty_map / max_std_scale, 0, 1 )
#         mse[grid_size-2].append(mean_squared_error(ground_truth.read()[visitable_locations[:, 0], visitable_locations[:, 1]], model_mean_map[visitable_locations[:, 0], visitable_locations[:, 1]], squared = False))

# print(f'Mean para sensor {std}: {np.mean(mse, axis=1)}')
# print(f'STD para sensor {std}: {np.std(mse, axis=1)}')
    