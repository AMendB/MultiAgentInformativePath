import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('.')
import os

# Read the data
folder = 'Experiments/Results/'
paths = [f'{folder}/[0.007 0.02 0.056 0.091] Random Network.4.Influence_10_0_100/metrics.csv', 
         f'{folder}/[0.213 0.381 0.13 0.197] Random Network.4.Influence_10_0_100/metrics.csv', 
         f'{folder}/[0.007 0.02 0.213 0.13] Random Network.4.Influence_10_0_100/metrics.csv', 
         f'{folder}/[0.007 0.02 0.056 0.091] Random WanderingAgent.4.Influence_10_0_100/metrics.csv',
         f'{folder}/[0.213 0.381 0.13 0.197] Random WanderingAgent.4.Influence_10_0_100/metrics.csv',
         f'{folder}/[0.007 0.02 0.213 0.13] Random WanderingAgent.4.Influence_10_0_100/metrics.csv',
         f'{folder}/[0.007 0.02 0.056 0.091] Random LawnMower.4.Influence_10_0_100/metrics.csv', 
         f'{folder}/[0.213 0.381 0.13 0.197] Random LawnMower.4.Influence_10_0_100/metrics.csv', 
         f'{folder}/[0.007 0.02 0.213 0.13] Random LawnMower.4.Influence_10_0_100/metrics.csv', 
         f'{folder}/[0.007 0.02 0.056 0.091] Random PSO.4.Influence_10_0_100/metrics.csv',
         f'{folder}/[0.213 0.381 0.13 0.197] Random PSO.4.Influence_10_0_100/metrics.csv',
         f'{folder}/[0.007 0.02 0.213 0.13] Random PSO.4.Influence_10_0_100/metrics.csv']

dfs = [pd.read_csv(path) for path in paths]

# Take the last MSE column value for each Run
errors = np.array([df.groupby('Run')['MSE'].last() for df in dfs])
# Create a dataframe with the data
algorithms = ['D-DQL', 'Random Walker', 'Lawn Mower', 'GP-based PSO']
agents_combinations = 3
data = pd.DataFrame({
    'Algorithm': np.repeat(algorithms, agents_combinations * (dfs[0]['Run'].iloc[-1]+1)),  # 4 algorithsm x 3 agents combinations x 100 episodes
    'Agents combinations': np.tile(['1', '2', '3'][:agents_combinations], 400),  # 3 agents combinations x 400 episodes
    'Value': errors.flatten()
})


# Plot the boxplot with seaborn
sns.set(style="darkgrid")
plt.figure(figsize=(15, 10))
ax = sns.boxplot(x='Algorithm', y='Value', hue='Agents combinations', data=data, palette='flare', showmeans=True) # hue='Agents combinations' to difference
ax.legend(title=ax.get_legend().get_title().get_text(), loc='upper left', fontsize=16, title_fontsize=16)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
plt.ylabel('Mean Squared Error (MSE)', fontsize=20)
plt.xlabel('')
# plt.title('Boxplots of the MSE for each algorithm and agents combination')
plt.tight_layout()
plt.savefig(fname=f"{folder}/Boxplots.svg")
plt.show()



