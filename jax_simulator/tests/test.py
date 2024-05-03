import jax.numpy as jnp
import jax.tree_util
from jax import jit
from jax import lax
import numpy as np
from jax import device_get
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from functools import partial
from jax import vmap
from jax.tree_util import tree_structure
from f110_gym.envs.base_classes import RaceCars
from f110_gym.envs.base_classes import Integrator
from f110_gym.envs.dynamic_models import *
from f110_gym.envs.laser_models import *

params = {
'v_switch' : 7.319,
'a_max' : 7.51,
'v_min' : 0.0,
'v_max' : 8.0,
'mu' : 0.8,
'C_Sf' : 4.718,
'C_Sr' : 5.4562,
'lf' : 0.15,
'lr' : 0.17,
'h' : 0.07,
'm' : 3.47,
'I' : 0.04,
's_min' : -0.4189,
's_max' : 0.4189,
'sv_min' : -3.2,
'sv_max' : 3.2,
'length' : 0.51,
'width' : 0.27
}


agent = RaceCars(params, seed=1, num_agents=10, time_step=0.01, num_beams=1080, fov=4.7)

scans_1 = []

print(f"{agent.num_agents} agents initiated, running simulation now for 100 time_steps")
start_time = time.time()
for _ in tqdm(range(100)):
    # Desired values:
    raw_steer = jnp.array([-0.2, 0.5, 0., 0.4, 0.5, -0.2, 0.5, 0., 0.4, 0.5])
    velocity = jnp.array([3., 6., 6., 3., 4., 3., 6., 6., 3., 4.])

    # Update the pose of the car
    scan = agent.update_pose(raw_steer, velocity)
    scans_1.append(scan)

end_time = time.time()
elapsed = end_time - start_time
print(f"simulated {agent.num_agents} agents for 100 steps in {elapsed} seconds")
print(scans_1)



def plot_paths():
    velocity_1 = [row[3] for row in state_1]
    steering_1 = [row[2] for row in state_1]
    x_pos_1 = [row[0] for row in state_1]
    y_pos_1 = [row[1] for row in state_1]

    velocity_2 = [row[3] for row in state_2]
    steering_2 = [row[2] for row in state_2]
    x_pos_2 = [row[0] for row in state_2]
    y_pos_2 = [row[1] for row in state_2]

    velocity_3 = [row[3] for row in state_3]
    steering_3 = [row[2] for row in state_3]
    x_pos_3 = [row[0] for row in state_3]
    y_pos_3 = [row[1] for row in state_3]

    #Plot
    print(f"{x_pos_3}")
    # 2D-Position
    plt.plot(x_pos_1, y_pos_1, color='orange', linewidth = 1., label = 'Vehicle 1')
    plt.plot(x_pos_2, y_pos_2, color='blue', linewidth = 1., label = 'Vehicle 2')
    plt.plot(x_pos_3, y_pos_3, color = 'green', linewidth =1., label = 'Vehicle 3' )
    plt.legend()
    plt.title('JAX - Position over time')
    plt.show()





