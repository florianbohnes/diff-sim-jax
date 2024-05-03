import numpy as np
from tqdm import tqdm


class RolloutRecording:
    """List like object that stores a history of recorded episodes.

    Attributes:
        map_name (str): Name of the environment the episodes are recorded in
        history (list): Data storage of the recordings
    """

    def __init__(self, map_name):
        self.map_name = map_name
        self.history = [{'obs': [], 'action': [], 'reward': []}]

    def append(self, new_episode: bool, obs, action, reward):
        """Operator to append the recordings to the data storage."""
        self.history[-1]['obs'].append(obs)
        self.history[-1]['action'].append(action)
        self.history[-1]['reward'].append(reward)

        # The next set added will be the start of a new eps
        if new_episode:
            self.history.append({'obs': [], 'action': [], 'reward': []})


def vec_eval(envs, agent, n_eval=1, map_name=None):
    avg_finish_time = []
    n_crash = 0
    n_overtaking = 0
    records_all = []
    history = [RolloutRecording(n) for n in map_name]

    bar = tqdm(total=n_eval)
    print('Evaluating agent...')

    obs_wrapped, infos = envs.reset()
    new_observations = infos['obs_']  # Access the non-normalized observations through infos

    while True:
        action = agent.get_action(obs_wrapped)
        old_observations = new_observations
        obs_wrapped, rewards, terminateds, truncateds, infos = envs.step(action)
        new_observations = infos['obs_']

        for i in range(len(map_name)):
            history[i].append(
                False,
                old_observations[i],
                action[i, :],
                rewards[i]
            )

        if 'final_info' in infos:
            for i in range(len(map_name)):
                if infos['_final_info'][i]:
                    history[i].append(
                        True,
                        infos['final_info'][i]['obs_'],
                        np.array([np.nan, np.nan]),
                        np.nan
                    )

        if all([len(h.history) > bar.n + 1 for h in history]):
            bar.update(1)
            if all([len(h.history) > n_eval for h in history]):
                for h in history:
                    h.history = h.history[:n_eval]
                break

    print('Done evaluating agent.')
    print(
        f'\t\tfinish_time'
        f'\ta_r_vel_mean'
        f'\ta_r_vel_std'
        f'\ta_r_steer_mean'
        f'\ta_r_steer_std'
        f'\ttotal_r'
        f'\t\tcrash'
        f'\tovertaking'
    )


    # Create records for all episodes and maps in the history
    for map_name_, h in zip(map_name, history):
        history_ = h.history
        for episode, history_eps in enumerate(history_):
            obs_all = dict()
            for key in history_eps['obs'][0].keys():
                # Flatten list into dicts
                obs_all[key] = np.array([o[key] for o in history_eps['obs']])
            obs_all['reward'] = np.array(history_eps['reward'])[:-1, ...]  # Remove the np.nan in the last entry
            obs_all['action_applied'] = np.array(
                history_eps['action'])[:-1, ...]  # Remove the np.nan in the last entry
            obs_all['action_planner'] = obs_all['action_planner'][:-1, ...]  # Remove the np.nan in the last entry
            # obs_all['action_applied'] = np.array([o['prev_action'] for o in history_eps['obs']])[1:, ...]
            obs_all['action_residual'] = obs_all['action_applied'] - obs_all['action_planner']

            records = get_records_episode(obs_all)

            if not records["metrics"]['collisions']:
                avg_finish_time.append(records["metrics"]["best_finish_time"])
            n_crash += records["metrics"]['collisions']
            n_overtaking += records["metrics"]['overtaking_success_sum']
            records_all.append(records)

            # crash type
            if not records["metrics"]['collisions']:
                crash_type = '-'
            elif records["metrics"]['collisions_env']:
                crash_type = 'e'
            elif records["metrics"]['collisions_agents']:
                crash_type = 'a'
            else:
                crash_type = 'u'

            print(
                f'- {str(map_name_[:10])}:'
                f'\t{np.round(records["metrics"]["best_finish_time"], 2):0.2f}'
                f'\t\t{records["metrics"]["action_residual_vel_mean"]:0.2f}'
                f'\t\t{records["metrics"]["action_residual_vel_std"]:0.2f}'
                f'\t\t{records["metrics"]["action_residual_steer_mean"]:0.4f}'
                f'\t\t{records["metrics"]["action_residual_steer_std"]:0.4f}'
                f'\t\t{records["metrics"]["total_return"].round(2):.2f}'
                f'\t\t{crash_type}'
                f'\t\t{records["metrics"]["overtaking_success_sum"]}'
                f'\n'
            )
    print(f'Summary: crashes: {n_crash}.')
    return  records_all, avg_finish_time, n_crash, n_overtaking


def get_records_episode(obs_all):
    """Process the recorded observations to obtain statistics of them."""
    records = dict()

    # History of states
    history = dict()
    history['poses_x'] = obs_all['poses_x'].squeeze()
    history['poses_y'] = obs_all['poses_y'].squeeze()
    history['linear_vels_x'] = obs_all['linear_vels_x'].squeeze()
    history['linear_vels_y'] = obs_all['linear_vels_y'].squeeze()
    history['ang_vels'] = obs_all['ang_vels_z'].squeeze()
    history['slip_angle'] = obs_all['slip_angle'].squeeze()
    history['acc_x'] = obs_all['acc_x'].squeeze()
    history['acc_y'] = obs_all['acc_y'].squeeze()
    history['rewards'] = obs_all['reward'].squeeze()
    history['collisions'] = obs_all['collisions'].squeeze()
    history['collisions_env'] = obs_all['collisions_env'].squeeze()
    history['collisions_agents'] = obs_all['collisions_agents'].squeeze()
    history['lap_times'] = obs_all['lap_times'].squeeze()
    history['lap_counts'] = obs_all['lap_counts'].squeeze()
    history['overtaking_success'] = obs_all['overtaking_success'].squeeze()
    history['finish_times'] = [history['lap_times'][history['lap_counts'] == i + 1].min() for i in
                               range(0, history['lap_counts'].max()) if
                               not any(history['collisions'][history['lap_counts'] == i + 1])]

    # # History of actions
    # Note: Len of action is 1 less since obs include the final 'done' state which doesn't require an action
    history['action_residual_steer'] = obs_all['action_residual'].squeeze()[:, 0]
    history['action_residual_vel'] = obs_all['action_residual'].squeeze()[:, 1]
    history['action_planner_steer'] = obs_all['action_planner'][:obs_all['action_residual'].shape[0], 0]
    history['action_planner_vel'] = obs_all['action_planner'][:obs_all['action_residual'].shape[0], 1]
    # Applied action is really the action thas been used in the env -> actions may be clipped!
    history['action_applied_steer'] = obs_all['action_applied'][:, 0]
    history['action_applied_vel'] = obs_all['action_applied'][:, 1]
    # Add to list
    records['history'] = history

    # History of performance
    metrics = dict()
    # Analysis of actions and stats
    for metrics_name in ['action_residual_vel', 'action_applied_vel', 'action_residual_steer', 'action_applied_steer',
                         'linear_vels_x']:
        metrics[f'{metrics_name}_mean'] = history[f'{metrics_name}'].mean(0)
        metrics[f'{metrics_name}_median'] = np.median(history[f'{metrics_name}'])
        metrics[f'{metrics_name}_std'] = history[f'{metrics_name}'].std(0)
        metrics[f'{metrics_name}_max'] = history[f'{metrics_name}'].max(0)
        metrics[f'{metrics_name}_min'] = history[f'{metrics_name}'].min(0)

    for metrics_name in ['action_residual_steer', 'action_applied_steer', 'linear_vels_y', 'slip_angle']:
        metrics[f'{metrics_name}_abs_mean'] = np.abs(history[f'{metrics_name}']).mean(0)
        metrics[f'{metrics_name}_abs_median'] = np.median(np.abs(history[f'{metrics_name}']))
        metrics[f'{metrics_name}_abs_std'] = np.abs(history[f'{metrics_name}']).std(0)
        metrics[f'{metrics_name}_abs_max'] = np.abs(history[f'{metrics_name}']).max(0)
        metrics[f'{metrics_name}_abs_min'] = np.abs(history[f'{metrics_name}']).min(0)

    # Performance metrics
    metrics['overtaking_success_sum'] = history['overtaking_success'].sum()
    metrics['rewards_mean'] = history['rewards'].mean()
    metrics['rewards_std'] = history['rewards'].std()
    metrics['total_return'] = history['rewards'].sum()
    metrics['collisions'] = history['collisions'].sum(0)
    metrics['collisions_env'] = history['collisions_env'].sum(0)
    metrics['collisions_agents'] = history['collisions_agents'].sum(0)
    metrics['steps'] = history['rewards'].shape[0]  # 1 Step less than the num of observed states due to 'done'
    metrics['full_laps'] = history['lap_counts'].max()
    metrics['full_laps_sum'] = history['lap_counts'].sum()

    # Lap times are recorded for a maximum of 2 laps
    if len(history['finish_times']) < 2:
        metrics['best_finish_time'] = 0.0
    else:
        # Finish time is the lap times of the second lap (running start)
        metrics['best_finish_time'] = history['finish_times'][1] - history['finish_times'][0]

    # Add to list
    records['metrics'] = metrics

    return records
