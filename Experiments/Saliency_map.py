import torch
import numpy as np
import matplotlib.pyplot as plt

# Cargar la red
# Cargar estados
# Hacer bien con la función V(s) y no sólo con la media de los valores Q

saliencies = []

for i in range(8):
    X = torch.tensor(states[0], requires_grad=True).unsqueeze(0).to('cuda:0')
    X.retain_grad()
    out = network.dqn(X.float(), torch.FloatTensor([0.025]).to('cuda:0'))
    out[0,i].backward()
    saliency = X.grad.data.abs()
    saliencies.append(saliency.cpu().numpy())

saliencies = np.array(saliencies)
saliencies = saliencies.squeeze(1)
saliencies_mean = saliencies.mean(0)

for state, sal in zip(states[0], saliencies_mean ):
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(state, cmap='gray')
    axs[1].imshow(sal, cmap='hot', alpha= 1-(sal-sal.min())/(sal.max() -sal.min()))
    plt.show()