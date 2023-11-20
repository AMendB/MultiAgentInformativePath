import sys
sys.path.append('.')

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from Environment.GroundTruthsModels.ShekelGroundTruth import GroundTruth


kernel = RBF(length_scale=5, length_scale_bounds=(0.001, 10))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.001, n_restarts_optimizer=10)

mapa = np.ones((50,50))
ground_truth = GroundTruth(mapa, max_number_of_peaks=4, is_bounded=True, seed=0)
ground_truth.reset()

X = np.vstack(np.where(mapa == 1)).T # coordenas de las celdas donde el mapa es 1, es decir, visitables
modelo_inicial = np.zeros_like(mapa) # modelo al inicio, un mapa como el original todo a cero

fig, ax = plt.subplots(1,3)
d1 = ax[0].imshow(modelo_inicial, cmap = 'gray', vmin =0, vmax=1) # subplot 1/3
d2 = ax[1].imshow(modelo_inicial, cmap = 'gray', vmin =0, vmax=1) # subplot 2/3
d5 = ax[2].imshow(ground_truth.read(), cmap = 'gray', vmin =0, vmax=1) # subplot 3/3
# Para mostrar la cruz donde se marca con el ratón en los subplots 1/3 y 2/3 (índices 0 y 1): #
d3, = ax[0].plot([],[], 'xr')
d4, = ax[1].plot([],[], 'xr')


y = []
X_meas = []
print("X_meas:" + str(X_meas))
print("Shape X_meas:" +  str(np.asarray(X_meas).shape))

alphas = []

i = 0

def onclick(event):

    global i


    print(event.xdata, event.ydata)
    

    # Tomamos una muestra #
    X_meas.append([int(event.ydata), int(event.xdata)])
    y.append(ground_truth.read(np.asarray([int(event.ydata), int(event.xdata)])))

    # Only one sample per point #
    # _, index = np.unique(X_meas, return_index=True, axis=0)
    # X_meas_unique = np.asarray(X_meas)[index]
    # y_unique = np.asarray(y)[index]

    alphas.append([0.001, 0.5][i%2]) #selecciona alternativamente un alpha nuevo dependiendo de si el ínidice de la nueva muestra es par o impar
    i = i+1
    gp.alpha = np.asarray(alphas)
    gp.fit(np.asarray(X_meas), np.asarray(y).reshape(-1,1))
    print("X_meas:" + str(X_meas))
    print("Shape X_meas:" +  str(np.asarray(X_meas).shape))


    model_out, uncertainty_out = gp.predict(X, return_std=True)

    d1.set_data(model_out.reshape((50,50))) # actualizo subplot 1/3 con la predicción del modelo
    d2.set_data(uncertainty_out.reshape((50,50))) # actualizo subplot 2/3 con la incertidumbre del modelo
    # Para mostrar la cruz donde se marca con el ratón: #
    d3.set_data(np.asarray(X_meas)[:,1], np.asarray(X_meas)[:,0])
    d4.set_data(np.asarray(X_meas)[:,1], np.asarray(X_meas)[:,0])

    fig.canvas.draw()
    fig.canvas.flush_events()

    print("Update:" + str(gp.kernel_.get_params()))


cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

