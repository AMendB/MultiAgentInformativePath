# EJECUTAR EN SPYDER

import numpy as np 
import matplotlib.pyplot as plt
position = np.array([10,30])
scenario_map = np.genfromtxt('Environment/Maps/example_map.csv', delimiter=',')  
visitable_locations = np.vstack(np.where(scenario_map != 0)).T # celdas visitables las distintas de cero en el mapa  
print(visitable_locations)

known_mask = np.zeros_like(scenario_map) # obtengo un array del mapa original relleno de ceros

px, py = position.astype(int) # separo posición en eje x e y 

# State - coverage area #
x = np.arange(0, scenario_map.shape[0]) # posiciones posibles en x
y = np.arange(0, scenario_map.shape[1]) # posiciones posibles en y

# Compute the circular mask (area) of the state 3 #
detection_length = 2
mask = (x[np.newaxis, :] - px) ** 2 + (y[:, np.newaxis] - py) ** 2 <= detection_length ** 2 

plt.figure(1)
plt.imshow(known_mask)
plt.figure(2)
plt.imshow(mask.T)
# =============================================================================
# plt.draw()
# plt.pause(0.01)
# =============================================================================


known_mask[mask.T] = 1.0
plt.figure(3)
plt.imshow(known_mask)


#%%
import numpy as np 
waypoints = np.expand_dims(np.copy(np.array([10, 30])), 0)
print(waypoints)
#%%
distance = 0
goal_position = np.array([12,31])
position = np.array([10,30])
distance += np.linalg.norm(goal_position - position) 
print(distance)
#%%
new_positions = np.array([[17, 7], [45, 23], [36, 11], [30, 9]])
uniques, inverse_index, counts = np.unique(np.asarray(new_positions), return_inverse=True, return_counts=True, axis=0)
not_collision_within = counts[inverse_index] == 1





