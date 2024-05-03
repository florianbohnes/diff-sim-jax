import logging
from copy import deepcopy
import gymnasium as gym
import numpy as np
import time

from f110_gym.envs.base_classes import Simulator, Integrator
from f110_gym.envs.trackline import TrackLine

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800


class F110Env(gym.Env):
    """OpenAI gym environment for F1TENTH. Should be initialized by calling gym.make('f110_gym:f110-v1', **kwargs)."""

    metadata = {'render_modes': ['human', 'human_fast'], 'render_fps': 100}

    def __init__(self, config, map_names, render_mode=None):
        self.config = config
        self._render_mode = render_mode

        self.timestep = self.config.sim.dt
        self.control_to_sim_ratio = int(self.config.sim['controller_dt'] / self.config.sim['dt'])
        self.num_agents = self.config.sim.n_agents
        self.ego_idx = 0
        self.v_params = self.config.sim.vehicle_params

        self.random_start_pose = self.config.sim.random_start_pose
        self.integrator = Integrator.RK4

        self.renderer = None
        self.render_callbacks = []

        # Initialize simulator
        self.sim = Simulator(self.v_params, self.num_agents, 42, time_step=self.timestep, integrator=self.integrator)

        # Map stuff
        self.current_map_name = None
        if len(map_names) > 1:
            self.random_map = True
            print('Random map mode')
        elif len(map_names) == 1:
            self.random_map = False
        else:
            raise Exception('There must the something wrong with the maps.')

        self.map_names: list = map_names
        self.map_ext = self.config.maps.map_ext
        self.map_paths = dict()
        self.wpt_paths = dict()

        self.track = TrackLine(self.map_names[0], config_map=self.config.maps)

        for map_name in self.map_names:
            self.map_paths[map_name] = self.config.maps.map_path + f'{map_name}/{map_name}_map'
            self.wpt_paths[map_name] = self.config.maps.map_path + f'{map_name}/{map_name}_raceline.csv'
            # self.wpt_paths[map_name] = self.config.maps.map_path + f'{map_name}/{map_name}_centerline_vel_newconv.csv'

        self._check_init_render()
        self._shuffle_map()

        # States
        self.current_obs = None
        self.prev_obs = None

        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents,))
        self.start_ys = np.zeros((self.num_agents,))
        self.start_thetas = np.zeros((self.num_agents,))
        self.start_rot = np.eye(2)

        # Race info
        self.lap_times = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,), dtype=np.int8)
        self.current_time = 0.0

        # Action space
        self.action_space = gym.spaces.Box(
            low=np.array([self.v_params['s_min'], self.v_params['v_min']]),
            high=np.array([self.v_params['s_max'], self.v_params['v_max']]),
            shape=(2,),
            dtype=np.float64
        )

        # Observation space
        self.observation_space = gym.spaces.Dict(
            {
                'aaa_scans': gym.spaces.Box(low=0, high=31, shape=(self.num_agents, 1080), dtype=np.float32),
                'poses_x': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'poses_y': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'poses_theta': gym.spaces.Box(low=-2 * np.pi, high=2 * np.pi, shape=(self.num_agents,),
                                              dtype=np.float64),
                'linear_vels_x': gym.spaces.Box(
                    low=self.v_params['v_min'], high=self.v_params['v_max'], shape=(self.num_agents,), dtype=np.float64
                ),
                'linear_vels_y': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'ang_vels_z': gym.spaces.Box(low=-100, high=100, shape=(self.num_agents,), dtype=np.float64),
                'collisions': gym.spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=bool),
                'collisions_env': gym.spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=bool),
                'collisions_agents': gym.spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=bool),
                'overtaking_success': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=bool),
                'lap_times': gym.spaces.Box(low=0, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'lap_counts': gym.spaces.Box(low=0, high=np.inf, shape=(self.num_agents,), dtype=np.int8),
                'prev_action': gym.spaces.Box(
                    # repeat array num_agents times
                    low=np.repeat([[self.v_params['s_min'], self.v_params['v_min']]], self.num_agents, axis=0),
                    high=np.repeat([[self.v_params['s_max'], self.v_params['v_max']]], self.num_agents, axis=0),
                    shape=(self.num_agents, 2),
                    dtype=np.float64
                ),
                'slip_angle': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'yaw_rate': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'acc_x': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'acc_y': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'steering_angle': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float64),
                'progress': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
                'safe_velocity': gym.spaces.Box(low=0.0, high=self.v_params['v_max'], shape=(1,), dtype=np.float64),
            }
        )

    @property
    def render_mode(self):
        """Get the rendering mode."""
        return self._render_mode

    @render_mode.setter
    def render_mode(self, render_mode):
        """Set the rendering mode."""
        self._render_mode = render_mode
        self._check_init_render()

    def step(self, action):
        """ Step function for the gym env."""
        if len(action.shape) < 2:
            action = np.expand_dims(action, 0)

        self.prev_obs = deepcopy(self.current_obs)

        for _ in range(self.control_to_sim_ratio):
            # steer, vel
            obs = self.sim.step(action)
            # obs['aaa_scans'] = obs['aaa_scans'][0][np.newaxis, :].astype(np.float32)
            obs['aaa_scans'] = obs['aaa_scans'].astype(np.float32)
            self.current_time = self.current_time + self.timestep
            terminated, truncated = self._check_done(obs)
            if terminated or truncated:
                break

        self.current_obs = obs
        self.current_obs['lap_times'] = self.lap_times
        self.current_obs['lap_counts'] = self.lap_counts
        self.current_obs['prev_action'] = action
        # lap progress
        current_progress = self.track.calculate_progress(
            np.array([self.current_obs['poses_x'][0], self.current_obs['poses_y'][0]]))
        prev_progress = self.track.calculate_progress(
            np.array([self.prev_obs['poses_x'][0], self.prev_obs['poses_y'][0]]))
        progress = current_progress - prev_progress
        progress = np.clip(progress, 0.0, 0.2)
        # if nan
        if np.isnan(progress):
            progress = 0.0
            logging.warning('progress is nan')

        self.current_obs['progress'] = np.array([progress], dtype=np.float64)
        safe_velocity = np.sqrt(0.8 * 9.81 / (np.tan(np.abs(self.current_obs['prev_action'][0][0]) + 1e-5) / 0.33))
        self.current_obs['safe_velocity'] = np.clip(np.array([safe_velocity]), 0.0, 8.0)

        if self.render_mode == "human" or self.render_mode == "human_fast":
            self.render()

        reward, _ = self._calc_reward(self.current_obs, self.prev_obs)
        info = {}

        # convert everything to numpy from jax
        obs = {k: np.asarray(v) for k, v in obs.items()}
        return obs, reward, terminated, truncated, info

    def reset(self, options=None, seed=None, map_name=None):
        """Reset the environment."""
        if self.random_map:
            map_name = options['map_name'] if options is not None and 'map_name' in options else None
            self._shuffle_map(map_name)  # this
        # reset counters and data members
        if self.random_start_pose:
            while True:
                n_positions = len(self.start_positions)
                idx = np.random.randint(0, n_positions)
                n_split = n_positions // self.num_agents
                # get idx range when racing with opponent(s)
                if self.num_agents > 1:
                    idx_range = np.arange(idx, idx + self.num_agents * n_split, n_split)
                    idx_range[0] += n_split // 4.0 * 3.0
                    idx_range[idx_range >= n_positions] -= n_positions
                    poses = self.start_positions[idx_range]
                else:
                    poses = self.start_positions[idx, np.newaxis]
                if poses[0, -1] < 0.01:
                    break
        else:
            n_positions = len(self.start_positions)
            idx = 0
            n_split = n_positions // self.num_agents
            # get idx range when racing with opponent(s)
            if self.num_agents > 1:
                idx_range = np.arange(idx, idx + self.num_agents * n_split, n_split)
                idx_range[0] += n_split // 4.0 * 3.0
                idx_range[idx_range >= n_positions] -= n_positions
                poses = self.start_positions[idx_range]
            else:
                poses = self.start_positions[idx, np.newaxis]

        # Initialize attributes
        self.current_time = 0.0
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array([[np.cos(-self.start_thetas[self.ego_idx]),
                                    -np.sin(-self.start_thetas[self.ego_idx])],
                                   [np.sin(-self.start_thetas[self.ego_idx]),
                                    np.cos(-self.start_thetas[self.ego_idx])]])

        # call reset to simulator
        obs = self.sim.reset(poses)
        obs['aaa_scans'] = obs['aaa_scans'].astype(np.float32)

        self.current_obs = obs
        self.prev_obs = deepcopy(self.current_obs)

        self.current_time = 0.0
        self.lap_times = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,), dtype=np.int8)

        self.current_obs['lap_times'] = self.lap_times
        self.current_obs['lap_counts'] = self.lap_counts
        self.current_obs['prev_action'] = np.zeros((self.num_agents, 2))
        self.current_obs['progress'] = np.array([0.0], dtype=np.float64)
        safe_velocity = np.sqrt(0.8 * 9.81 / (np.tan(np.abs(self.current_obs['prev_action'][0][0]) + 1e-5) / 0.33))
        self.current_obs['safe_velocity'] = np.clip(np.array([safe_velocity]), 0.0, 8.0)

        if self.render_mode == "human" or self.render_mode == "human_fast":
            self.render()

        info = {}
        # convert everything to numpy from jax
        obs = {k: np.asarray(v) for k, v in obs.items()}
        return obs, info

    def _calc_reward(self, current_obs, prev_obs):
        """Calculate the reward."""
        coefficient = self.config.sim.reward
        distance_threshold = self.config.sim.reward.distance_threshold
        reward_info = dict()

        # Taking a step
        reward_info['reward_step'] = 1 * coefficient.step

        # Distance to the closest obstacle
        distance = np.min(np.abs(current_obs['aaa_scans'][0]))
        reward_info['reward_distance'] = (distance_threshold - distance) * coefficient.safe_distance_to_obstacles \
            if distance < distance_threshold else 0.0

        # Lap progress
        reward_info['reward_progress'] = current_obs['progress'][0] * coefficient.progress

        # Track safe max physical speed (0.8 is the friction coefficient, 0.33 the wheelbase)
        safe_velocity = self.current_obs['safe_velocity']

        reward_info['reward_safe_velocity'] = \
            (2.0 - np.abs(current_obs['linear_vels_x'][0] - safe_velocity[0])) * coefficient.safe_velocity

        reward_info['reward_collision'] = current_obs['collisions'][0] * coefficient.collision

        # Longitudinal velocity
        reward_info['reward_long_vel'] = current_obs['linear_vels_x'][0] * coefficient.long_vel

        # Lateral velocity
        reward_info['reward_lat_vel'] = np.abs(current_obs['linear_vels_y'][0]) * coefficient.lat_vel

        # Action change
        action_change = np.sum(np.abs(current_obs['prev_action'][0] - prev_obs['prev_action'][0]) / (
                self.action_space.high - self.action_space.low))
        reward_info['reward_action_change'] = action_change * coefficient.action_change

        # Yaw rate change
        yaw_rate_change = np.abs(current_obs['yaw_rate'][0] - self.prev_obs['yaw_rate'][0])
        reward_info['reward_yaw_change'] = yaw_rate_change * coefficient.yaw_change

        # Overtaking
        reward_info['reward_overtaking'] = coefficient.overtaking if current_obs['overtaking_success'][0] else 0.0

        # Total reward
        reward = np.sum(list(reward_info.values())) * coefficient.scaling
        reward_info['reward'] = reward

        return reward, reward_info

    def _shuffle_map(self, map_name=None):
        """Shuffle the map."""
        if map_name is None:
            idx = np.random.randint(0, len(self.map_names)) if len(self.map_names) > 1 else 0
            self.current_map_name = self.map_names[idx]
        else:
            self.current_map_name = map_name

        map_path = self.map_paths[self.current_map_name]
        wpt_path = self.wpt_paths[self.current_map_name]

        self.update_map(map_path, self.map_ext)

        self.waypoints = np.loadtxt(wpt_path, delimiter=self.config.maps.wpt_delim,
                                    skiprows=self.config.maps.wpt_rowskip)
        self.start_positions = self.waypoints[:, (self.config.maps.wpt_xind, self.config.maps.wpt_yind,
                                                  self.config.maps.wpt_thind, self.config.maps.wpt_kappa)]
        self.pose_start = self.start_positions[0, ...]

    def _check_done(self, obs):
        """Check if the episode is done."""
        n_laps = 1

        # This is assuming 2 agents.py
        left_t = 2
        right_t = 2

        poses_x = np.array(obs['poses_x']) - self.start_xs
        poses_y = np.array(obs['poses_y']) - self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1, :]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :] ** 2 + temp_y ** 2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] <= 4:
                self.lap_times[i] = self.current_time

        # Only look at the ego vehicle
        terminated = obs['collisions'][self.ego_idx]
        truncated = (self.toggle_list >= 2 * n_laps)[self.ego_idx]
        return bool(terminated), bool(truncated)

    def update_map(self, map_path, map_ext):
        self.sim.set_map(map_path + '.yaml', map_ext)
        if self.renderer is not None:
            self.renderer.update_map(map_path, map_ext)

    def update_params(self, params, index=-1):
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        self.render_callbacks.append(callback_func)

    def render(self):
        """Render the environment."""
        if self.render_mode is not None:
            self.renderer.update_obs(self.current_obs)
            for render_callback in self.render_callbacks:
                render_callback(self.renderer)
            self.renderer.dispatch_events()
            self.renderer.on_draw()
            self.renderer.flip()
            if self._render_mode == 'human':
                time.sleep(0.005)
            elif self._render_mode == 'human_fast':
                pass

    def _check_init_render(self):
        """Initialize the renderer if needed."""
        if self._render_mode in ['human', 'human_fast']:
            if self.renderer is None:
                # first call, initialize everything
                from f110_gym.envs.rendering import EnvRenderer
                from pyglet import options as pygl_options
                pygl_options['debug_gl'] = False
                self.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
        else:
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None

    def close(self):
        """Close the environment."""
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = None
        super().close()
