from datetime import datetime
import hydra
from hydra.core.hydra_config import DictConfig

from source import evaluation
from source import maker
from source.agents import BaseControllerAgent
import time



@hydra.main(version_base=None, config_path='../configs/', config_name='config')
def main(config: DictConfig) -> None:
    """Main function for the experiment."""
    print(f'Starting a new run @ {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}.')
    maps = (config.maps.maps_train + config.maps.maps_test)
    envs = maker.make_multi_env(config, ([maker.make_env(config, maps[i]) for i in range(len(maps))]))
    agent = BaseControllerAgent()

    start = time.time()
    evaluation.vec_eval(envs, agent, n_eval=1, map_name=maps)
    end = time.time()
    number = config.sim.n_agents
    duration = end - start
    print(f"\nSimulation finished\n"
          f" agents: {number} \n"
          f" duration: {duration} seconds\n")
    envs.close()
