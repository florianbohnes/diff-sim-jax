import copy
import gymnasium as gym
from planning.planner_utils import *
from planning.pure_puresuit import PurePursuitPlanner
from copy import deepcopy
import numpy as np

class GymActionObservationWrapper(gym.Wrapper):
    """A wrapper that modifies the action and observation space of the environment."""

    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        """Returns a modified observation."""
        raise NotImplementedError

    def action(self, action):
        """Returns a modified action before :meth:`env.step` is called."""

        raise NotImplementedError

    def reverse_action(self, action):
        """Returns a reversed ``action``."""
        raise NotImplementedError

    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        observation, reward, terminated, truncated, info = self.env.step(self.action(action))
        observation_ = self.observation(observation)
        info['obs_'] = deepcopy(observation_)
        return observation_, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        observation, info = self.env.reset(**kwargs)
        observation_ = self.observation(observation)
        info['obs_'] = deepcopy(observation_)
        return observation_, info


class OpponentAutopilotEnvWrapper(GymActionObservationWrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        self.n_opponents = self.unwrapped.num_agents - 1
        self.autopilot_type = config.planner.opponents.type
        self.last_observation = None

        self.autopilots = [make_planner(
            self.config.planner.opponents,
            self.config.sim.vehicle_params
        ) for _ in range(self.n_opponents)]

        # Purge action and observation space of other agents.py
        observation_space_ = gym.spaces.Dict()
        for k, v in self.get_wrapper_attr('observation_space').items():
            space_class = type(v)
            observation_space_[k] = space_class(v.low[0][jnp.newaxis], v.high[0][jnp.newaxis], (1, *v.shape[1:]), v.dtype)
            # observation_space_[k] = space_class(v.low[1:], v.high[1:], (1, *v.shape[1:]), v.dtype)
        self.observation_space = observation_space_
        # self.action_space = deepcopy(self.action_space)

    def observation(self, observation):
        """Returns a modified observation."""
        self.last_observation = copy.deepcopy(observation)
        observation = {k: jnp.expand_dims(v[0], axis=0) for k, v in observation.items()}
        return observation

    def action(self, action):
        """Returns a modified action before :meth:`env.step` is called."""
        actions = jnp.vstack((action, jnp.zeros((self.n_opponents, 2))))
        for i in range(self.n_opponents):
            observation = {
                k: jnp.expand_dims(v[i + 1], axis=0) for k, v in self.last_observation.items() if i + 1 < len(v)
            }
            steering, velocity, waypoints = self.autopilots[i].plan(observation, self.env.unwrapped.waypoints)
            #actions[i + 1] = jnp.array([steering, velocity])
            actions = actions.at[i+1].set(jnp.array([steering, velocity]))
        return actions

    def reset(self, *args, **kwargs):
        """Calls :meth:`env.reset` and renders the environment."""
        out = super().reset(*args, **kwargs)
        self.autopilots = [make_planner(
            self.config.planner.opponents,
            self.config.sim.vehicle_params
        ) for _ in range(self.n_opponents)]
        return out


class BasePlannerEnvWrapper(GymActionObservationWrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config

        self.n_next_points = config.planner.ego.n_next_points
        self.skip_next_points = config.planner.ego.skip_next_points

        # Note: Important to deepcopy, otherwise the super().observation_space will be altered
        self.observation_space = deepcopy(self.get_wrapper_attr('observation_space'))
        self.planner = make_planner(self.config.planner.ego, self.config.sim.vehicle_params)

        v_params = self.get_wrapper_attr('v_params')
        self.observation_space['action_planner'] = gym.spaces.Box(
            low=np.array([v_params['s_min'], v_params['v_min']]),
            high=np.array([v_params['s_max'], v_params['v_max']]),
            dtype=np.float64)

        self.action_space_high = self.env.action_space.high
        self.action_space_low = self.env.action_space.low

        self.action_applied = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float64
        )

    @property
    def waypoints(self):
        """Returns the waypoints."""
        return self.env.unwrapped.waypoints

    def observation(self, obs):
        """Returns a modified observation."""
        # try:
        steering, velocity, wps = self.planner.plan(obs, self.waypoints)

        # Make sure that planner cannot generate actions that are not in the env's action space
        action_planner = jnp.clip(
            jnp.array([steering, velocity]),
            a_min=self.observation_space['action_planner'].low,
            a_max=self.observation_space['action_planner'].high
        )

        obs['action_planner'] = (action_planner - self.action_space_low) / (
                self.action_space_high - self.action_space_low) * 2 - 1

        return obs

    def action(self, action):
        out = (self.action_space_high - self.action_space_low) * (action + 1) / 2 + self.action_space_low
        return out

    def step(self, *args, **kwargs):
        """Calls :meth:`env.step` and renders the environment."""
        out = super().step(*args, **kwargs)
        self.render()
        return out

    def reset(self, *args, **kwargs):
        """Calls :meth:`env.reset` and renders the environment."""
        out = super().reset(*args, **kwargs)
        self.render()
        self.planner = make_planner(self.config.planner.ego, self.config.sim.vehicle_params)
        return out

    def render(self):
        """Renders the environment."""
        if self.env.render_mode is not None:
            if self._render not in self.env.unwrapped.render_callbacks:
                self.env.unwrapped.add_render_callback(self._render)
            super().render()

    def _render(self, x):
        """Renders the environment."""
        return render_callback(x, self.planner, self.waypoints)


def make_planner(config_planner, vehicle_params):
    if config_planner.type == 'pure_pursuit':
        config_planner = copy.deepcopy(config_planner)
        # TODO
        #config_planner['vgain'] = float(jnp.random.choice(config_planner['vgain']))
        config_planner['vgain'] = 1.0 # for now
        return PurePursuitPlanner(config_planner, vehicle_params)
    else:
        raise ValueError(f'Unknown planner type {config_planner.type}.')
