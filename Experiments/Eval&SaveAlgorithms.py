import sys
sys.path.append('.')
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import trange
import pandas as pd
from cycler import cycler

# from Environment.MonitoringEnvironmentPenultimate import MultiAgentMonitoring
from Environment.MonitoringEnvironment import MultiAgentMonitoring
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator


class AlgorithmRecorderAndAnalizer:
    def __init__(self, env:MultiAgentMonitoring, scenario_map, n_agents,  relative_path, algorithm, reward_funct, reward_weights, runs) -> None:
        self.env = env
        self.scenario_map = scenario_map
        self.n_agents = n_agents
        self.relative_path = relative_path
        self.algorithm = algorithm
        self.reward_funct = reward_funct
        self.reward_weights = reward_weights
        self.runs = runs

    def plot_all_figs(self, run):
        self.env.render()
        plt.show()
        self.plot_paths(run)
        self.plot_metrics(show_plot=True, run=run)

    def get_heatmap(self, heatmaps = None, only_save = False):

        if only_save:
            # Save all heatmaps in one figure #
            if len(heatmaps) > 1:
                gridspec = dict(hspace=0.1, width_ratios=[1 + 0.25 * len(heatmaps)]+[1 + 0.25 * len(heatmaps)]*len(heatmaps), height_ratios=[1, 0.3])
                fig, axs = plt.subplots(2, len(heatmaps)+1, figsize=(5* len(heatmaps),6), gridspec_kw=gridspec)

                heatmap_total = np.zeros_like(self.scenario_map)
                for heatmap in heatmaps:
                    heatmap_total += heatmap  
                heatmap_total[self.env.non_water_mask] = np.nan
                axis0 = axs[0][0].imshow(heatmap_total, cmap='YlOrRd', interpolation='nearest')
                axs[0][0].set_title(f"Total")
                fig.colorbar(axis0,ax=axs[0][0], shrink=0.8)
                fig.suptitle(f"{self.algorithm.split('_')[0]} | {self.reward_funct.split('_')[0]}_{'_'.join(map(str, self.reward_weights))}", fontsize=12)
                
                # Percetange visited cells #
                visited = np.count_nonzero(heatmap_total>0)
                visited_5percent = np.count_nonzero(heatmap_total>=self.runs*0.05)
                visited_20percent = np.count_nonzero(heatmap_total>=self.runs*0.20)
                visited_50percent = np.count_nonzero(heatmap_total>=self.runs*0.50)
                visited_80percent = np.count_nonzero(heatmap_total>=self.runs*0.80)
                visitables = len(self.env.visitable_locations)
                axs[1][0].text(0, 0, f"Percentage visited: {round(100*visited/visitables, 2)}%\n"\
                            f"Percentage visited 5% of eps: {round(100*visited_5percent/visitables, 2)}%\n"\
                            f"Percentage visited 20% of eps: {round(100*visited_20percent/visitables, 2)}%\n"\
                            f"Percentage visited 50% of eps: {round(100*visited_50percent/visitables, 2)}%\n"\
                            f"Percentage visited 80% of eps: {round(100*visited_80percent/visitables, 2)}%", 
                            transform = axs[1][0].transAxes, fontsize='small')
                axs[1][0].axis('off')

                for index, heatmap in enumerate(heatmaps):
                    heatmap[self.env.non_water_mask] = np.nan
                    cax = axs[0][index+1].imshow(heatmap, cmap='YlOrRd', interpolation='nearest')
                    axs[0][index+1].set_title(f"Sensors: {index}")
                    fig.colorbar(cax,ax=axs[0][index+1], shrink=0.8)
                    axs[1][index+1].axis('off')
                    visited = np.count_nonzero(heatmap>0)
                    visited_5percent = np.count_nonzero(heatmap>=self.runs*0.05)
                    visited_20percent = np.count_nonzero(heatmap>=self.runs*0.20)
                    visited_50percent = np.count_nonzero(heatmap>=self.runs*0.50)
                    visited_80percent = np.count_nonzero(heatmap>=self.runs*0.80)
                    axs[1][index+1].text(0, 0, f"Percentage visited: {round(100*visited/visitables, 2)}%\n"\
                                f"Percentage visited 5% of eps: {round(100*visited_5percent/visitables, 2)}%\n"\
                                f"Percentage visited 20% of eps: {round(100*visited_20percent/visitables, 2)}%\n"\
                                f"Percentage visited 50% of eps: {round(100*visited_50percent/visitables, 2)}%\n"\
                                f"Percentage visited 80% of eps: {round(100*visited_80percent/visitables, 2)}%", 
                                transform = axs[1][index+1].transAxes, fontsize='small')
                
            else:
                gridspec = dict(hspace=0.0, height_ratios=[1, 0.3])
                fig, axs = plt.subplots(2, 1, figsize=(5,6), gridspec_kw=gridspec)
                heatmap_total = heatmaps[0]
                heatmap_total[self.env.non_water_mask] = np.nan
                axis0 = axs[0].imshow(heatmap_total, cmap='YlOrRd', interpolation='nearest')
                axs[0].set_title(f"Total")
                fig.colorbar(axis0,ax=axs[0])
                fig.suptitle(f"{self.algorithm.split('_')[0]} | {self.reward_funct.split('_')[0]}_{'_'.join(map(str, self.reward_weights))}", fontsize=11)
                
                # Percetange visited cells #
                visited = np.count_nonzero(heatmap_total>0)
                visited_5percent = np.count_nonzero(heatmap_total>=self.runs*0.05)
                visited_20percent = np.count_nonzero(heatmap_total>=self.runs*0.20)
                visited_50percent = np.count_nonzero(heatmap_total>=self.runs*0.50)
                visited_80percent = np.count_nonzero(heatmap_total>=self.runs*0.80)
                visitables = len(self.env.visitable_locations)
                axs[1].text(0.25, 0, f"Percentage visited: {round(100*visited/visitables, 2)}%\n"\
                            f"Percentage visited 5% of eps: {round(100*visited_5percent/visitables, 2)}%\n"\
                            f"Percentage visited 20% of eps: {round(100*visited_20percent/visitables, 2)}%\n"\
                            f"Percentage visited 50% of eps: {round(100*visited_50percent/visitables, 2)}%\n"\
                            f"Percentage visited 80% of eps: {round(100*visited_80percent/visitables, 2)}%", 
                            transform = axs[1].transAxes, fontsize='small')
                axs[1].axis('off')
                
            plt.savefig(fname=f"{self.relative_path}/Heatmaps.png")
            plt.close()
            

        else:
            if heatmaps is None:
                heatmaps = [np.zeros_like(self.scenario_map) for _ in range(self.env.n_sensors_type)]
            
            # Heatmap by sensors type #
            for sens_type in range(self.env.n_sensors_type):           
                visited_locations = np.vstack([self.env.fleet.vehicles[agent].waypoints for agent in range(self.n_agents) if self.env.sensors_type[agent] == sens_type])
                heatmaps[sens_type][visited_locations[:,0], visited_locations[:,1]] += 1

        return(heatmaps)

    def save_gp_model(self, run):
        gridspec = dict(hspace=0.0, width_ratios=[1, 0.4, 1, 1])
        fig, axs = plt.subplots(1, 4, figsize=(12,4.5), gridspec_kw=gridspec)

        # AXIS 0: Print model mean #
        ground_truth = self.env.ground_truth.read()
        ground_truth[env.non_water_mask] = np.nan
        axs[0].imshow(ground_truth, cmap ='cet_linear_bgy_10_95_c74', vmin = 0.0, vmax = 1.0)
        axs[0].set_title("Ground Truth")

        axs[1].set_visible(False)

        # AXIS 1: Print model mean #
        model_mean = env.model_mean_map.copy()
        model_mean[env.non_water_mask] = np.nan
        axs[2].imshow(model_mean, cmap ='cet_linear_bgy_10_95_c74', vmin = 0.0, vmax = 1.0)
        axs[2].set_title("Model Mean")

        # AXIS 1: Print model uncertainty #
        model_uncertainty = env.model_uncertainty_map.copy()
        model_uncertainty[env.non_water_mask] = np.nan
        axs[3].imshow(model_uncertainty, cmap ='gray', vmin = 0.0, vmax = 1.0)#, norm='log')
        axs[3].set_title("Model Uncertainty")

        # Save figure #
        plt.suptitle(f"{self.algorithm.split('_')[0]} | {self.reward_funct.split('_')[0]}_{'_'.join(map(str, self.reward_weights))} | EP{run}", ha='left')
        plt.savefig(fname=f"{self.relative_path}/Models/Ep{run}.png")
        plt.close()

    def plot_paths(self, run = None, save_plot = False):
        # Agents path plot over ground truth #
        plt.figure(figsize=(7,5))

        waypoints = [self.env.fleet.vehicles[i].waypoints for i in range(self.n_agents)]
        gt = self.env.ground_truth.read()
        gt[self.env.non_water_mask] = np.nan

        plt.imshow(gt,  cmap ='cet_linear_bgy_10_95_c74', vmin = 0.0, vmax = 1.0, alpha = 0.3)

        for agent, agents_waypoints in enumerate(waypoints):
            y = [point[0] for point in agents_waypoints]
            x = [point[1] for point in agents_waypoints]
            plt.plot(x, y, color=self.env.colors_agents[agent+2])


        if save_plot:
            plt.title(f"{self.algorithm.split('_')[0]} | {self.reward_funct.split('_')[0]}_{'_'.join(map(str, self.reward_weights))} | EP{run}", fontsize='medium')
            plt.savefig(fname=f"{self.relative_path}/Paths/Ep{run}.png")
            plt.close()
        else:
            plt.title(f"Real Importance (GT) with agents path, EP {run}")
            plt.show()  # Mostrar el gráfico resultante

    def plot_metrics(self, show_plot=False, run = None, save_plot = False, plot_std=False):
            # Reward and Error final graphs #
            fig, ax = plt.subplots(2, 3, figsize=(17,9))

            ax[0][0].set_prop_cycle(cycler(color=self.env.colors_agents[-self.n_agents:]))
            ax[0][0].plot(self.reward_agents_acc, '-')
            ax[0][0].set(title = 'Reward', xlabel = 'Step', ylabel = 'Individual Reward')
            ax[0][0].legend([f'Agent {i}' for i in range(self.n_agents)])
            ax[0][0].plot(self.reward_acc, 'b-', linewidth=4)
            ax[0][0].grid()
            if plot_std:
                for agent in range(self.n_agents):
                    ax[0][0].fill_between(self.results_std.index, self.results_mean[f'AccRw{agent}'] - self.results_std[f'AccRw{agent}'], self.results_mean[f'AccRw{agent}'] + self.results_std[f'AccRw{agent}'], alpha=0.2, label='Std')
                ax[0][0].fill_between(self.results_std.index, self.results_mean['R_acc'] - self.results_std['R_acc'], self.results_mean['R_acc'] + self.results_std['R_acc'], color='b', alpha=0.2, label='Std')

            ax[0][1].plot(self.mse, '-', label='Media')
            # ax[0][1].plot(self.mse_peaks, '-')
            ax[0][1].set(title = 'MSE', xlabel = 'Step')
            # ax[0][1].legend(['Total', 'In Peaks'])
            ax[0][1].grid()
            if plot_std:
                ax[0][1].fill_between(self.results_std.index, self.results_mean['MSE'] - self.results_std['MSE'], self.results_mean['MSE'] + self.results_std['MSE'], alpha=0.2, label='Std')

            ax[0][2].plot(self.error_r2, '-')
            ax[0][2].set(title = 'R2_error', xlabel = 'Step')
            ax[0][2].grid()
            if plot_std:
                ax[0][2].fill_between(self.results_std.index, self.results_mean['R2_error'] - self.results_std['R2_error'], self.results_mean['R2_error'] + self.results_std['R2_error'], alpha=0.2, label='Std')

            ax[1][0].plot(self.uncert_mean, '-')
            ax[1][0].plot(self.uncert_max, '-')
            ax[1][0].set(title = 'Uncertainty', xlabel = 'Step')
            ax[1][0].legend(['Mean', 'Max'])
            ax[1][0].grid()
            if plot_std:
                ax[1][0].fill_between(self.results_std.index, self.results_mean['Uncert_mean'] - self.results_std['Uncert_mean'], self.results_mean['Uncert_mean'] + self.results_std['Uncert_mean'], alpha=0.2, label='Std')
                ax[1][0].fill_between(self.results_std.index, self.results_mean['Uncert_max'] - self.results_std['Uncert_max'], self.results_mean['Uncert_max'] + self.results_std['Uncert_max'], alpha=0.2, label='Std')

            plot_traveled_distance = False
            if plot_traveled_distance == True or self.n_agents == 1:
                ax[1][1].bar(list(map(str, [*range(self.n_agents)])) ,self.traveled_distance_agents[-1], width=0.4, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
                ax[1][1].set(title = 'Traveled_distance', xlabel = 'Agent')
                ax[1][1].set_ylim(90, 110)
                ax[1][1].grid()
            else:
                mean_distance = np.mean(self.distances_between_agents, axis=1)
                ax[1][1].plot(mean_distance, '-')
                ax[1][1].set(title = 'Mean distance between agents', xlabel = 'Step', ylabel = 'Distance')
                # ax[1][1].legend([*env.fleet.get_distances_between_agents().keys()])
                ax[1][1].grid()

            ax[1][2].plot(self.max_redundancy, '-')
            ax[1][2].set(title = 'Max_Redundancy', xlabel = 'Step')
            ax[1][2].grid()
            if plot_std:
                ax[1][2].fill_between(self.results_std.index, self.results_mean['Max_Redundancy'] - self.results_std['Max_Redundancy'], self.results_mean['Max_Redundancy'] + self.results_std['Max_Redundancy'], alpha=0.2, label='Std')

            if not(run is None):
                fig.suptitle(f'Run nº: {run}. Alg: {self.algorithm} Rw: {self.reward_funct}_' + '_'.join(map(str, reward_weights)), fontsize=16)
            else:
                fig.suptitle(f'{len(self.runs)} episodes. {self.algorithm} | {self.reward_funct}_' + '_'.join(map(str, reward_weights)), fontsize=16)

            if save_plot:
                fig.savefig(fname=f"{self.relative_path}/AverageMetrics_{len(self.runs)}eps.png")

            if show_plot:
                plt.show()

    def plot_r_and_e(self, run):
        # Reward and Error final graphs #
        final_fig, final_axes = plt.subplots(1, 2, figsize=(15,5))

        final_axes[0].plot(self.reward_agents_acc , '-')
        final_axes[0].set(title = 'Accumulated Reward', xlabel = 'Step', ylabel = 'Reward')
        final_axes[0].legend([f'Agent {i}' for i in range(self.n_agents)])
        final_axes[0].plot(self.reward_acc, 'b-', linewidth=4)
        final_axes[0].grid()

        final_axes[1].plot(self.mse, '-')
        final_axes[1].set(title = 'Error', xlabel = 'Step', ylabel = 'Absolute Error')
        final_axes[1].grid()
        final_fig.suptitle('Run nº: ' + str(run), fontsize=16)

        plt.show()

    def save_ground_truths(self, ground_truths_to_save):

        output_filename = '/GroundTruths.npy'

        np.save( self.relative_path + output_filename , ground_truths_to_save)

    def save_registers(self, new_reward=None, reset=False):

        # Get data # 
        if reset == True:
            self.reward_steps_agents = [[0 for _ in range(self.n_agents)]]
            self.reward_steps = [0]
            self.mse = [self.env.get_model_mu_mse_error()]
            self.mse_peaks = [self.env.get_model_mu_mse_error_in_peaks()]
            self.error_r2 = [self.env.get_model_mu_r2_error()]
            self.uncert_mean = [self.env.get_uncertainty_mean()]
            self.uncert_max = [self.env.get_uncertainty_max()]
            self.traveled_distance_agents = [self.env.fleet.get_fleet_distances_traveled()]
            self.traveled_distance = [0]
            self.max_redundancy = [self.env.get_redundancy_max()]
            if self.n_agents > 1:
                self.distances_between_agents = []
        else:
            # Add new metrics data #
            self.reward_steps_agents.append(list(new_reward.values()))
            self.reward_steps.append(np.sum(list(new_reward.values())))
            self.mse.append(self.env.get_model_mu_mse_error())
            self.mse_peaks.append(self.env.get_model_mu_mse_error_in_peaks())
            self.error_r2.append(self.env.get_model_mu_r2_error())
            self.uncert_mean.append(self.env.get_uncertainty_mean())
            self.uncert_max.append(self.env.get_uncertainty_max())
            self.traveled_distance_agents.append(self.env.fleet.get_fleet_distances_traveled())
            self.traveled_distance.append(np.sum(self.env.fleet.get_fleet_distances_traveled()))
            self.max_redundancy.append(self.env.get_redundancy_max())

        self.reward_acc = np.cumsum(self.reward_steps)
        self.reward_agents_acc = np.cumsum(self.reward_steps_agents, axis=0)
        if self.n_agents > 1:
            self.distances_between_agents.append([*self.env.fleet.get_distances_between_agents().values()])

        # Save metrics #
        if self.n_agents > 1:
            data = [*self.reward_agents_acc[-1], self.reward_acc[-1], self.mse[-1], self.mse_peaks[-1], self.error_r2[-1], self.uncert_mean[-1], self.uncert_max[-1], self.traveled_distance[-1], self.max_redundancy[-1], *self.traveled_distance_agents[-1], *self.distances_between_agents[-1]]
        else:
            data = [*self.reward_agents_acc[-1], self.reward_acc[-1], self.mse[-1], self.mse_peaks[-1], self.error_r2[-1], self.uncert_mean[-1], self.uncert_max[-1], self.traveled_distance[-1], self.max_redundancy[-1], *self.traveled_distance_agents[-1]]
        metrics.save_step(run_num=run, step=step, metrics=data)

        # Save waypoints #
        for veh_id, veh in enumerate(self.env.fleet.vehicles):
            waypoints.save_step(run_num=run, step=step, metrics=[veh_id, veh.actual_agent_position[0], veh.actual_agent_position[1], done[veh_id]])

    def save_scenario_map(self):

        output_filename = '/ScenarioMap.npy'

        np.save( self.relative_path + output_filename , self.scenario_map)
    
    def plot_and_tables_metrics_average(self, metrics_path, table, wilcoxon_dict, show_plot = True , save_plot = False):
        
        metrics_df = MetricsDataCreator.load_csv_as_df(metrics_path)
        self.runs = metrics_df['Run'].unique()

        numeric_columns = metrics_df.select_dtypes(include=[np.number])
        self.results_mean = numeric_columns.groupby('Step').agg('mean')
        self.results_std = numeric_columns.groupby('Step').agg('std')

        # Num steps limited so all episodes have the same lenght #
        min_steps = 48
        self.results_mean = self.results_mean.head(min_steps)  
        self.results_std = self.results_std.head(min_steps)  

        first_accreward_agent_index = self.results_mean.columns.get_loc('AccRw0')
        self.reward_agents_acc = self.results_mean.iloc[:, first_accreward_agent_index:first_accreward_agent_index + self.n_agents].values.tolist()
        self.reward_acc = self.results_mean['R_acc'].values.tolist()
        self.mse = self.results_mean['MSE'].values.tolist()
        self.mse_std = self.results_std['MSE'].values.tolist()
        self.mse_peaks = self.results_mean['MSE_peaks'].values.tolist()
        self.error_r2 = self.results_mean['R2_error'].values.tolist()
        self.uncert_mean = self.results_mean['Uncert_mean'].values.tolist()
        self.uncert_max = self.results_mean['Uncert_max'].values.tolist()
        self.traveled_distance = self.results_mean['Traveled_distance'].values.tolist()
        self.max_redundancy = self.results_mean['Max_Redundancy'].values.tolist()
        first_traveldist_agent_index = self.results_mean.columns.get_loc('TravelDist0')
        self.traveled_distance_agents = self.results_mean.iloc[:, first_traveldist_agent_index:first_traveldist_agent_index + self.n_agents].values.tolist()
        if self.n_agents > 1:
            first_distbetween_index = self.results_mean.columns.get_loc([*env.fleet.get_distances_between_agents().keys()][0])
            self.distances_between_agents = self.results_mean.iloc[:, first_distbetween_index:first_distbetween_index + int((self.n_agents*(self.n_agents-1))/2)].values.tolist()

        if show_plot or save_plot:
            self.plot_metrics(show_plot=show_plot, save_plot=save_plot, plot_std=True)
            plt.close('all')

        # LATEX TABLE #
        import warnings
        warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

        name_alg = self.algorithm.split('_')[0].capitalize()
        name_rw = self.reward_funct.split('_')[0].capitalize() + '-' + '-'.join(map(str, reward_weights))

        if not name_alg.capitalize() in table:
            new_df = pd.DataFrame(
                    columns=pd.MultiIndex.from_product([[name_alg], ["Mean33", "Std33", "Mean66", "Std66", "Mean100", "Std100"]]),
                    index=table.index)
            table = pd.concat([table, new_df], axis=1)

        # Calculate the real mse at 33%, 66% and 100% of each episode, without aproximation at step min_steps, and add to table #
        mse_33 = []
        mse_66 = []
        mse_100 = []
        for episode in self.runs:
            mse_episode = np.array(numeric_columns[numeric_columns['Run']==episode]['MSE'])
            mse_33.append(mse_episode[round(len(mse_episode)*0.33)])
            mse_66.append(mse_episode[round(len(mse_episode)*0.66)])
            mse_100.append(mse_episode[-1])
        table.loc[name_rw, name_alg] = [np.mean(mse_33), np.std(mse_33), np.mean(mse_66), np.std(mse_66), np.mean(mse_100), np.std(mse_100)]


        # To do WILCOXON TEST, extract the MSE vector of len(vector)=runs from df at steps 33%, 66% and 100% of min_steps=48 for each episode #
        series_33 = numeric_columns[numeric_columns['Step']==round(min_steps*0.33)]['MSE'].reset_index(drop='True').rename('33')
        series_66 = numeric_columns[numeric_columns['Step']==round(min_steps*0.66)]['MSE'].reset_index(drop='True').rename('66')
        series_100 = numeric_columns[numeric_columns['Step']==round(min_steps*1)]['MSE'].reset_index(drop='True').rename('100')
        wilcoxon_dict[f'{name_alg.capitalize()} - {name_rw.capitalize()}'] = pd.concat([series_33, series_66, series_100], axis=1)
        
        return table, wilcoxon_dict


def wilcoxon_test(wilcoxon_dict):
        from itertools import combinations, product
        from scipy.stats import wilcoxon

        results = {}
        
        metrics = ["33", "66", "100"]
        
        for metric in metrics:
            for alg1, alg2 in combinations(wilcoxon_dict.keys(), 2):
                data1 = wilcoxon_dict[alg1]
                data2 = wilcoxon_dict[alg2]

                for metric in metrics:
                    
                    statistic, p_value = wilcoxon(data1[f'{metric}'], data2[f'{metric}'])
                    
                    key = f"{alg1} vs {alg2} - {metric}"
                    results[key] = {
                        "Statistic": statistic,
                        "P-Value": p_value,
                        "Significant": p_value < 0.05  # Puedes ajustar el nivel de significancia aquí
                    }
        
        return results
        

if __name__ == '__main__':

    import time
    from Algorithms.LawnMower import LawnMowerAgent
    from Algorithms.NRRA import WanderingAgent
    from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent

    algorithms = [
        # 'WanderingAgent', 
        # 'LawnMower', 
        # 'DoneTrainings/runs_2A/Alg_Network_RW_Influence_5_5/', 
        # 'DoneTrainings/runs_2A/Alg_Network_RW_Influence_10_0/', ##
        # 'DoneTrainings/runs_4A/Alg_Network_RW_Influence_5_5/',  
        # 'DoneTrainings/runs_4A/Alg_Network_RW_Influence_10_0/',  ##
        # 'DoneTrainings/Penultimate/runs_4A/Alg_Network_RW_Influence_10_0/',  ##
        # 'DoneTrainings/runs_4A - Último (ponderado)/Alg_Network_RW_Influence_10_0/',  ##
        # 'DoneTrainings/runs_4A - Último (ponderado)/Alg_Network_RW_Error_10_0/',  ##
        # 'DoneTrainings/entry_not_normalized_var_runs4A/Alg_Network_RW_Influence_10_0/',  ##
        'DoneTrainings/entry_std_sensor_runs_4A/Alg_Network_RW_Influence_10_0/',  ##
        ]

    SHOW_FINAL_PLOT_GRAPHICS = False
    SHOW_PLOT_GRAPHICS = False
    SAVE_PLOTS = True
    SAVE_COLLAGES = True
    RUNS = 100
    SEED = 3
    # STDs_SENSORS = [np.array([0.005,0.005]), np.array([0.05,0.05]), np.array([0.5,0.5])]
    # STDs_SENSORS = [np.array([0.005,0.005,0.005,0.005]), np.array([0.05,0.05,0.05,0.05]), np.array([0.5,0.5,0.5,0.5])]
    STDs_SENSORS = [np.array([0.005,0.05,0.5,0.5])]
    # STD_SENSORS = np.array([0.005,0.005,0.005,0.005]) #np.array([0.005,0.05,0.075,0.5]) #'random' #np.array([0.1, 0.25, 0.1, 0.25])[:n_agents]
    # STD_SENSORS = np.array([0.05,0.05,0.05,0.05]) #np.array([0.005,0.05,0.075,0.5]) #'random' #np.array([0.1, 0.25, 0.1, 0.25])[:n_agents]
    # STD_SENSORS = np.array([0.5,0.5,0.5,0.5]) #np.array([0.005,0.05,0.075,0.5]) #'random' #np.array([0.1, 0.25, 0.1, 0.25])[:n_agents]

    EXTRA_NAME = ''





    range_std_sensormeasure = (1*0.5/100, 1*0.5*100/100) # AML is "the best", from then on 100 times worse
    for peaks_location in ['Random']:
    # for peaks_location in ['Random', 'Upper', 'MiddleLeft', 'MiddleRight', 'Middle', 'Bottom']:
        saving_paths = []
        data_table_average = pd.DataFrame() 
        wilcoxon_dict = {}              

        for STD_SENSORS in STDs_SENSORS:
            if len(STDs_SENSORS) > 1:
                EXTRA_NAME = f'{str(STD_SENSORS[-1])} {peaks_location} '
            elif len(STDs_SENSORS) == 1:
                EXTRA_NAME = f'[{" ".join(map(str, STD_SENSORS))}] {peaks_location} '
            for path_to_training_folder in algorithms:

                if path_to_training_folder == 'WanderingAgent' or path_to_training_folder == 'LawnMower':
                    selected_algorithm = path_to_training_folder

                    # Set config #
                    n_agents = len(STD_SENSORS) if isinstance(STD_SENSORS, np.ndarray) else 4  # max 4
                    movement_length = 2
                    influence_length = 6
                    mean_sensormeasure = np.array([0, 0, 0, 0])[:n_agents] # mean of the measure of every agent
                    std_sensormeasure = STD_SENSORS  # std of the measure of every agent
                    reward_function = 'Influence_area_changes_model' # Position_changes_model, Influence_area_changes_model, Error_with_model
                    scenario_map = np.genfromtxt('Environment/Maps/ypacarai_map_low_res.csv', delimiter=',')
                    reward_weights=(10, 0)

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
                                            seed=SEED,
                                            movement_length=movement_length,
                                            influence_length=influence_length,
                                            flag_to_check_collisions_within=False,
                                            max_collisions=1000,
                                            reward_function=reward_function,
                                            ground_truth_type='shekel',
                                            peaks_location=peaks_location,
                                            dynamic=False,
                                            obstacles=False,
                                            regression_library='gpytorch',  # scikit, gpytorch or botorch
                                            scale_kernel = True,
                                            reward_weights=reward_weights,
                                            show_plot_graphics=SHOW_PLOT_GRAPHICS,
                                            )
                    
                    if selected_algorithm == "LawnMower":
                        selected_algorithm_agents = [LawnMowerAgent(world=scenario_map, number_of_actions=8, movement_length=movement_length, forward_direction=0, seed=SEED) for _ in range(n_agents)]
                    elif selected_algorithm == "WanderingAgent":
                        selected_algorithm_agents = [WanderingAgent(world=scenario_map, number_of_actions=8, movement_length=movement_length, seed=SEED) for _ in range(n_agents)]

                else:
                    # Load env config #
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
                                        range_std_sensormeasure=range_std_sensormeasure, #tuple(env_config['range_std_sensormeasure']),
                                        std_sensormeasure= STD_SENSORS, #np.array(env_config['std_sensormeasure']),#
                                        fleet_initial_positions=env_config['fleet_initial_positions'], #np.array(env_config['fleet_initial_positions']), #
                                        seed=SEED,
                                        movement_length=env_config['movement_length'],
                                        influence_length=env_config['influence_length'],
                                        flag_to_check_collisions_within=env_config['flag_to_check_collisions_within'],
                                        max_collisions=env_config['max_collisions'],
                                        reward_function=reward_function,
                                        ground_truth_type=env_config['ground_truth_type'],
                                        peaks_location=peaks_location,
                                        dynamic=env_config['dynamic'],
                                        obstacles=env_config['obstacles'],
                                        regression_library=env_config['regression_library'],
                                        reward_weights=reward_weights,
                                        scale_kernel=env_config['scale_kernel'],
                                        show_plot_graphics=SHOW_PLOT_GRAPHICS,
                                        )
                    
                    # Load exp config #
                    f = open(path_to_training_folder + 'experiment_config.json',)
                    exp_config = json.load(f)
                    f.close()

                    network_with_sensornoises = exp_config['network_with_sensornoises']
                    independent_networks_by_sensors_type = exp_config['independent_networks_by_sensors_type']

                    if network_with_sensornoises and not(independent_networks_by_sensors_type):
                        selected_algorithm = "Network_With_SensorNoises"
                    elif not(network_with_sensornoises) and independent_networks_by_sensors_type:
                        selected_algorithm = "Independent_Networks_By_Sensors_Type"
                    else:
                        raise NotImplementedError("This algorithm is not implemented. Choose one that is.")

                    network = MultiAgentDuelingDQNAgent(env=env,
                                            memory_size=int(1E3),  #int(1E6), 1E5
                                            batch_size=exp_config['batch_size'],
                                            target_update=1000,
                                            seed = SEED,
                                            concensus_actions=exp_config['concensus_actions'],
                                            device='cuda:0',
                                            network_with_sensornoises = network_with_sensornoises,
                                            independent_networks_by_sensors_type = independent_networks_by_sensors_type,
                                            )
                    network.load_model(path_to_training_folder + 'BestPolicy.pth')
                    network.epsilon = 0.05


                # Reward function and create path to save #
                relative_path = f'Experiments/Results/{EXTRA_NAME}' + selected_algorithm.split('_')[0] + '.' + str(n_agents) + '.' + reward_function.split('_')[0] + '_' + '_'.join(map(str, reward_weights))
                if not(os.path.exists(relative_path)): # create the directory if not exists
                    os.mkdir(relative_path)
                    os.mkdir(f'{relative_path}/Paths')
                    os.mkdir(f'{relative_path}/Models')
                saving_paths.append(relative_path)

                # algorithm_analizer = AlgorithmRecorderAndAnalizer(env, scenario_map, n_agents, relative_path, selected_algorithm, reward_function, reward_weights)
                algorithm_analizer = AlgorithmRecorderAndAnalizer(env, scenario_map, n_agents, relative_path, selected_algorithm, f'{EXTRA_NAME}{reward_function}', reward_weights, RUNS)
                algorithm_analizer.save_scenario_map()
                env.save_environment_configuration(relative_path)

                # Initialize metrics saving class #
                if n_agents > 1:
                    metrics_names = [*[f'AccRw{id}' for id in range(n_agents)], 'R_acc', 'MSE', 'MSE_peaks', 'R2_error', 'Uncert_mean', 'Uncert_max',
                                    'Traveled_distance', 'Max_Redundancy', *[f'TravelDist{id}' for id in range(n_agents)], *env.fleet.get_distances_between_agents().keys()]
                else:
                    metrics_names = [*[f'AccRw{id}' for id in range(n_agents)], 'R_acc', 'MSE', 'MSE_peaks', 'R2_error', 'Uncert_mean', 'Uncert_max',
                                    'Traveled_distance', 'Max_Redundancy', *[f'TravelDist{id}' for id in range(n_agents)]]
                metrics = MetricsDataCreator(metrics_names=metrics_names,
                                            algorithm_name=selected_algorithm + '.' + str(n_agents) + '.' + reward_function + '.' + '_'.join(map(str, reward_weights)),
                                            experiment_name= 'metrics',
                                            directory=relative_path )

                waypoints = MetricsDataCreator(metrics_names=['vehicle', 'x', 'y', 'Done'],
                                            algorithm_name=selected_algorithm + '.' + str(n_agents) + '.' + reward_function + '.' + '_'.join(map(str, reward_weights)),
                                            experiment_name= 'waypoints',
                                            directory=relative_path)
                
                ground_truths_to_save = []
                episodes_reward_acc = []
                heatmaps = None
                
                # Start episodes #
                for run in trange(RUNS):
                    
                    done = {i: False for i in range(n_agents)}
                    states = env.reset_env()
                    if SHOW_PLOT_GRAPHICS:
                        env.render()

                    # runtime = 0
                    step = 0

                    # Save data #
                    algorithm_analizer.save_registers(reset=True)
                    ground_truths_to_save.append(env.ground_truth.read())
                    
                    if selected_algorithm == "LawnMower":
                        for i in range(n_agents):
                            selected_algorithm_agents[i].reset(0)

                    # Take first actions #
                    if selected_algorithm == 'Network_With_SensorNoises' or selected_algorithm == 'Independent_Networks_By_Sensors_Type':
                        network.nogobackfleet_masking_module.reset()
                        actions = network.select_concensus_actions(states=states, sensor_error=env.std_sensormeasure, positions=env.get_active_agents_positions_dict(), n_actions=env.n_actions, done = done)
                    else:
                        actions = {i: selected_algorithm_agents[i].move(env.fleet.vehicles[i].actual_agent_position) for i in range(n_agents)}

                    while any([not value for value in done.values()]):  # while at least 1 active
                        
                        step += 1

                        # t0 = time.time()
                        states, new_reward, done = env.step(actions)
                        # t1 = time.time()
                        # runtime += t1-t0

                        # print("Actions: " + str(actions))
                        # print("Rewards: " + str(new_reward))

                        # Save data #
                        algorithm_analizer.save_registers(new_reward, reset=False)

                        # Take new actions #
                        if selected_algorithm == 'Network_With_SensorNoises' or selected_algorithm == 'Independent_Networks_By_Sensors_Type':
                            actions = network.select_concensus_actions(states=states, sensor_error=env.std_sensormeasure, positions=env.get_active_agents_positions_dict(), n_actions=env.n_actions, done = done)
                        else:
                            actions = {i: selected_algorithm_agents[i].move(env.fleet.vehicles[i].actual_agent_position) for i in range(n_agents)}

                        #print(env.gaussian_process.model.covar_module.base_kernel.lengthscale.item())

                    # print('Total runtime: ', runtime)
                    

                    if SHOW_PLOT_GRAPHICS:
                        algorithm_analizer.plot_all_figs(run)
                    
                    if SAVE_PLOTS and run%1 == 0:
                        algorithm_analizer.plot_paths(run, save_plot=SAVE_PLOTS)
                        algorithm_analizer.save_gp_model(run)

                    heatmaps = algorithm_analizer.get_heatmap(heatmaps)

                    episodes_reward_acc.append(algorithm_analizer.reward_acc[-1])

                print(f'Average reward {RUNS} episodes: {np.mean(episodes_reward_acc)}')
                
                algorithm_analizer.save_ground_truths(ground_truths_to_save)
                algorithm_analizer.get_heatmap(heatmaps, only_save=True)
                metrics.save_data_as_csv()
                waypoints.save_data_as_csv()

                data_table_average, wilcoxon_dict = algorithm_analizer.plot_and_tables_metrics_average(metrics_path=relative_path + '/metrics.csv', table=data_table_average, wilcoxon_dict=wilcoxon_dict, show_plot=SHOW_FINAL_PLOT_GRAPHICS,save_plot=SAVE_PLOTS)

        if EXTRA_NAME != '':
            if len(STDs_SENSORS) > 1:
                EXTRA_NAME = ' vs '.join([str(std[-1]) for std in STDs_SENSORS]) + f' {peaks_location} '
            elif len(STDs_SENSORS) == 1:
                EXTRA_NAME = f'[{" ".join(map(str, STD_SENSORS))}] {peaks_location} '

        file = open(f'Experiments/Results/{EXTRA_NAME}LatexTableAverage{RUNS}eps_{n_agents}A.txt', "w")
        file.write(data_table_average.style.to_latex())
        file.close()

        # Test de Wilcoxon # 
        results = wilcoxon_test(wilcoxon_dict)
        file = open(f'Experiments/Results/{EXTRA_NAME}Wilcoxon{RUNS}eps_{n_agents}A.txt', "w")
        for key, result in results.items():
            info = f"Test de Wilcoxon para {key}: Estadístico = {result['Statistic']}, Valor p = {result['P-Value']}, Significativo = {result['Significant']}"
            print(info)
            file.write(info + '\n')
        file.close()
        

        if SAVE_COLLAGES:
            from shutil import rmtree
            import cv2

            # Function to crop an image
            def crop_image(image, x, y, width, high):
                return image[y:y+high, x:x+width]
            
            # Collage agents paths #
            images_paths = [sorted([os.path.join(f'{path}/Paths/', file) for file in os.listdir(f'{path}/Paths/')], key=lambda x: int(x.split('/')[-1].replace('Ep', '').replace('.png', ''))) for path in saving_paths]
            collage = np.hstack([np.vstack([crop_image(cv2.imread(img), 110, 20, 490, 485) for img in algorithm]) for algorithm in images_paths])
            cv2.imwrite(f'Experiments/Results/{EXTRA_NAME}Paths{RUNS}eps_{n_agents}A.png', collage)
            
            # collage GP models #
            images_paths = [sorted([os.path.join(f'{path}/Models/', file) for file in os.listdir(f'{path}/Models/')], key=lambda x: int(x.split('/')[-1].replace('Ep', '').replace('.png', ''))) for path in saving_paths]
            gt_collage = np.vstack([crop_image(cv2.imread(img), 0, 0, 500, 2000) for img in images_paths[-1]])
            collage = np.hstack([np.vstack([crop_image(cv2.imread(img), 500, 0, 700, 2000) for img in algorithm]) for algorithm in images_paths])
            collage = np.hstack([gt_collage, collage])
            cv2.imwrite(f'Experiments/Results/{EXTRA_NAME}Models{RUNS}eps_{n_agents}A.png', collage)

            # Collage average metrics #
            networks = set([path.split('/')[-2].split('.')[0] for path in saving_paths])
            collage = []
            for net in networks:
                algorithms_paths = [path for path in saving_paths if net in path]
                images_paths = [next(os.path.join(path, file) for file in os.listdir(path) if file.startswith('AverageMetrics') and file.endswith('png')) for path in algorithms_paths]
                collage.append(np.hstack([crop_image(cv2.imread(img), 100, 0, 1580, 880) for img in images_paths]))
            collage = np.vstack(collage)
            cv2.imwrite(f'Experiments/Results/{EXTRA_NAME}MetricsAverage{RUNS}eps_{n_agents}A.png', collage)

            # Collage average heatmaps #
            collage = []
            for net in networks:
                algorithms_paths = [path for path in saving_paths if net in path]
                images_paths = [next(os.path.join(path, file) for file in os.listdir(path) if file.startswith('Heatmap') and file.endswith('png')) for path in algorithms_paths]
                collage.append(np.hstack([crop_image(cv2.imread(img), 0, 0, 2000, 2000) for img in images_paths]))
            collage = np.vstack(collage)
            cv2.imwrite(f'Experiments/Results/{EXTRA_NAME}HeatmapsAverage{RUNS}eps_{n_agents}A.png', collage)

        # Remove Paths and Models folders #
        for path in saving_paths:
            rmtree(f'{path}/Paths')
            rmtree(f'{path}/Models')