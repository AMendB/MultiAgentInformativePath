from Environment.MonitoringEnvironment import MultiAgentMonitoring
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np
import matplotlib.pyplot as plt
import torch
from Utils.metrics_wrapper import MetricsDataCreator

N = 4
sc_map = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
visitable_locations = np.vstack(np.where(sc_map != 0)).T
random_index = np.random.choice(np.arange(0,len(visitable_locations)), N, replace=False)

initial_positions = np.asarray([[24, 21],[28,24],[27,19],[24,24]])

env = MultiAgentMonitoring(scenario_map=sc_map,
                           fleet_initial_positions=initial_positions,
                           distance_budget=200,
                           number_of_agents=N,
                           seed=0,
                           detection_length=1,
                           movement_length=1,
                           max_collisions=5000,
                           forget_factor=0.5,
                           attrittion=0.1,
                           networked_agents=False,
                           obstacles=False)


multiagent = MultiAgentDuelingDQNAgent(env=env,
                                       memory_size=int(1E4),
                                       batch_size=64,
                                       target_update=1000,
                                       soft_update=False,
                                       tau=0.0001,
                                       epsilon_values=[1.0, 0.05],
                                       epsilon_interval=[0.0, 0.33],
                                       learning_starts=10,
                                       gamma=0.99,
                                       lr=1e-4,
                                       noisy=True,
                                       train_every=20,
                                       save_every=5000)

multiagent.load_model('Learning/runs/Greedy_baseline_no_networked/Episode_45000_Policy.pth')

metrics = MetricsDataCreator(metrics_names=['Accumulated Reward', 'Disconnections'],
                             algorithm_name='DRL',
                             experiment_name='DRLResults',
                             directory='./')

paths = MetricsDataCreator(metrics_names=['vehicle', 'x', 'y'],
                           algorithm_name='DRL',
                           experiment_name='DRL_paths',
                           directory='./')

multiagent.epsilon = 0.05

for run in range(10):
    done = False
    s = env.reset()
    R = 0
    step = 0

    # Initial register #
    metrics.save_step(run_num=run, step=step, metrics=[R, env.fleet.number_of_disconnections])
    for veh_id, veh in enumerate(env.fleet.vehicles):
        paths.save_step(run_num=run, step=step, metrics=[veh_id, veh.position[0], veh.position[1]])

    while not done:

        step += 1

        selected_action = []
        for i in range(env.n_agents):
            individual_state = env.individual_agent_observation(state=s, agent_num=i)
            q_values = multiagent.dqn(torch.FloatTensor(individual_state).unsqueeze(0).to(multiagent.device)).detach().cpu().numpy().flatten()
            mask = np.asarray([env.fleet.vehicles[i].check_action(a) for a in range(0, 8)])
            q_values[mask] = -np.inf
            selected_action.append(np.argmax(q_values))

        s, r, done, i = env.step(selected_action)

        R = np.mean(r) + R

        # Register positions and metrics #
        metrics.register_step(run_num=run, step=step, metrics=[R, env.fleet.number_of_disconnections])
        for veh_id, veh in enumerate(env.fleet.vehicles):
            paths.register_step(run_num=run, step=step, metrics=[veh_id, veh.position[0], veh.position[1]])

        # env.render()

metrics.register_experiment()
paths.register_experiment()
