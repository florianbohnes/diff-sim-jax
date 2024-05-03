import gymnasium as gym

from planning.planner_wrapper import BasePlannerEnvWrapper, OpponentAutopilotEnvWrapper


def make_env(config, map_names):
    map_names = [map_names] if type(map_names) is not list else map_names

    def thunk():
        env = gym.make(
            'f110_gym:f110-v1',
            config=config,
            map_names=map_names,
            render_mode=config.sim.render_mode if config.render else None
        )
        if config.sim.n_agents > 1:
            # Opponent autopilot wrapper
            env = OpponentAutopilotEnvWrapper(env, config)

        # Planner wrapper that includes the planner's action in the observation
        env = BasePlannerEnvWrapper(env, config)
        return env

    return thunk


def make_multi_env(config, make_functions: list, daemon=True):
    if config.async_env:
        return gym.vector.AsyncVectorEnv(make_functions, daemon=daemon, shared_memory=True)
    else:
        return gym.vector.SyncVectorEnv(make_functions)
