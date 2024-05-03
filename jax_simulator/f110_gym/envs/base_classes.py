from enum import Enum
import jax.debug
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax import vmap
from jax import jit
from functools import partial

from f110_gym.envs.dynamic_models import vehicle_dynamics_st, pid, accl_constraints, steering_constraint
from f110_gym.envs.laser_models import ScanSimulator2D, check_ttc_jit, ray_cast
from f110_gym.envs.collision_models import get_vertices, collision_multiple
from planning.planner_utils import pi_2_pi


class Integrator(Enum):
    RK4 = 1
    # Euler integrator has not been implemented for the JAX simulator.

@jit
@partial(vmap, in_axes=(0, 0))
def steer_buffer_fun(steer_buffer, raw_steer):
    # steer buffer helper
    steer = steer_buffer[-1]
    steer_buffer = steer_buffer[:-1]
    steer_buffer = jnp.append(raw_steer, steer_buffer)
    return steer, steer_buffer


@jit
@partial(vmap, in_axes=0)
def bound_yaw_angle(state):
    yaw = state[4]
    yaw = lax.cond(yaw > 2 * jnp.pi,
                   lambda _: yaw - 2 * jnp.pi,
                   lambda _: lax.cond(yaw < 0,
                                      lambda _: yaw + 2 * jnp.pi,
                                      lambda _: yaw,
                                      yaw),
                   yaw)
    state = state.at[4].set(yaw)
    return state


@jit
@partial(vmap, in_axes=(0, 0, 0, 0))
def dynamics_integration(k1, k2, k3, k4):
    # dynamics integration
    f = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return f


@jit
@partial(vmap, in_axes=(0, 0, 0, None, None, None, None, None, None, None, None,
                                        None, None, None, None, None, None, None, None))
def f_getter(state, sv, accl, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min,
             sv_max, v_switch, a_max, v_min, v_max):
    f = vehicle_dynamics_st(
        state,
        jnp.array([sv, accl]),
        mu,
        C_Sf,
        C_Sr,
        lf,
        lr,
        h,
        m,
        I,
        s_min,
        s_max,
        sv_min,
        sv_max,
        v_switch,
        a_max,
        v_min,
        v_max
    )
    return f


@jit
@partial(vmap, in_axes=(0, 0, None, None, None, None))
def check_ttc(current_scan, vel, scan_angles, cosines, side_distances, ttc_thresh):
    """ Check iTTC against the environment.

    Sets vehicle states accordingly if collision occurs. Note that this does NOT check collision with other
    agents.py.
    State is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
    Args:
        current_scan(np.ndarray, (n, )): current scan range array
    """

    # check_ttc_jit returns a boolean array
    in_collision = check_ttc_jit(current_scan,
                                 vel,
                                 scan_angles,
                                 cosines,
                                 side_distances,
                                 ttc_thresh)

    # if in collision stop vehicle
    vel = jnp.where(in_collision, 0., vel)

    return in_collision, vel


@jit
def join_scans(original_scan, new_scans):
    """
    Helper functions to join scans together
    The legacy code did a loop in ray_cast_agents, updating the scan iteratively
    for every opp_pose. We now vectorize the ray_cast function over all opp_poses
    at once, resulting in #opp_poses scans. This function joins those scans together
    """

    # Boolean helper array to decide where to change entries
    mask = original_scan != new_scans

    output = original_scan

    for row_mask, row_input in zip(mask, new_scans):
        output = jnp.where(row_mask, row_input, output)

    new_scan = output
    return new_scan


@jit
@partial(vmap, in_axes=(0, 0, None, 0, None, None))
def _ray_cast(pose, scan, scan_angles, opp_poses, length, width):
    '''
    Wrapper function to call ray_cast
    This now enables calling the ray_cast function for only one agent and its scan,
    but it enables vmap over all of that agents opponents poses
    Args:
        pose(jnp.array(3, )) of one vehicles
        scans(jnp.array(3, )) of one vehicles
        opp_poses(jnp.array(num_agents-1, 3)) for one vehicles
    Returns:

    '''
    # jnp.array(num_agents-1, 4)
    opp_vertices = vmap_vertices_poses(opp_poses, length, width)

    # This function returns num_agents-1 scan arrays,
    # in each array the original scan is adjusted only for the ray cast of one opponent
    new_scans = ray_cast(pose, scan, scan_angles, opp_vertices)
    new_scan = join_scans(scan, new_scans)

    return new_scan

vmap_vertices_poses = jit(vmap(get_vertices, in_axes=(0, None, None)))
vmap_vertices = vmap(get_vertices, in_axes=(0, None, None), out_axes=0)
# TODO: warum zweimal ?


class RaceCars(object):
    # Instantiation of RaceCars object will now be one object containing ALL cars that are simulated. For now, we
    # treat all cars as having the same parameters -> self of initialization is carries parameters for all.
    # New parameter num_agents is added to RaceCar class init function.

    def __init__(self, params, seed, num_agents, time_step=0.01, num_beams=1080, fov=4.7, integrator=Integrator.RK4):
        """ Init function.


        Args:
            params (dict): vehicle parameter dictionary, includes
                {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max',
                'sv_min', 'sv_max', 'v_switch', 'a_max': 9.51, 'v_min', 'v_max', 'length', 'width'}
            is_ego (bool, default=False): ego identifier
            time_step (float, default=0.01): physics sim time step
            num_beams (int, default=1080): number of beams in the laser scan
            fov (float, default=4.7): field of view of the laser
            num_agents: number of agents initialized
        """

        self.scan_simulator = None
        self.cosines = None
        self.scan_angles = None
        self.side_distances = None

        # initialization
        self.params = params
        # Accessing params from dict is expensive, so we cache them here
        self.mu = params['mu']
        self.C_Sf = params['C_Sf']
        self.C_Sr = params['C_Sr']
        self.lf = params['lf']
        self.lr = params['lr']
        self.h = params['h']
        self.m = params['m']
        self.I = params['I']
        self.s_min = params['s_min']
        self.s_max = params['s_max']
        self.sv_min = params['sv_min']
        self.sv_max = params['sv_max']
        self.v_switch = params['v_switch']
        self.a_max = params['a_max']
        self.v_min = params['v_min']
        self.v_max = params['v_max']
        self.length = params['length']
        self.width = params['width']

        self.seed = seed
        self.time_step = time_step
        self.num_beams = num_beams
        self.fov = fov
        self.integrator = integrator
        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        self.state = jnp.zeros((7,))
        # Add another state: Lateral acceleration (no independent state)
        self.a_x = jnp.zeros((num_agents,))
        self.a_y = jnp.zeros((num_agents,))

        self.max_slip = None
        self.max_yaw = None

        # steering delay buffer
        # changed from empty initialization to 0 initialization,
        # simplifies control flow when filling buffer
        self.steer_buffer_size = 2
        self.steer_buffer = jnp.zeros((self.steer_buffer_size,))

        # collision threshold for iTTC to environment
        self.ttc_thresh = 0.005

        # initialize scan sim
        if self.scan_simulator is None:
            self.scan_simulator = ScanSimulator2D(num_beams, fov)
            global vmap_scan
            vmap_scan = vmap(self.scan_simulator.scan, in_axes=0)

            scan_ang_incr = self.scan_simulator.get_increment()

            # angles of each scan beam, distance from lidar to edge of car at each beam,
            # and precomputed cosines of each angle
            self.cosines = jnp.zeros((num_beams,))
            self.scan_angles = jnp.zeros((num_beams,))
            self.side_distances = jnp.zeros((num_beams,))

            dist_sides = self.width / 2.
            dist_fr = (self.lf + self.lr) / 2.

            for i in range(num_beams):
                angle = -fov / 2. + i * scan_ang_incr
                self.scan_angles = self.scan_angles.at[i].set(angle)
                self.cosines = self.cosines.at[i].set(jnp.cos(angle))

                if angle > 0:
                    if angle < jnp.pi / 2:
                        # between 0 and pi/2
                        to_side = dist_sides / jnp.sin(angle)
                        to_fr = dist_fr / jnp.cos(angle)
                        self.side_distances = self.side_distances.at[i].set(min(to_side, to_fr))
                    else:
                        # between pi/2 and pi
                        to_side = dist_sides / jnp.cos(angle - jnp.pi / 2.)
                        to_fr = dist_fr / jnp.sin(angle - jnp.pi / 2.)
                        self.side_distances = self.side_distances.at[i].set(min(to_side, to_fr))
                else:
                    if angle > -jnp.pi / 2:
                        # between 0 and -pi/2
                        to_side = dist_sides / jnp.sin(-angle)
                        to_fr = dist_fr / jnp.cos(-angle)
                        self.side_distances = self.side_distances.at[i].set(min(to_side, to_fr))
                    else:
                        # between -pi/2 and -pi
                        to_side = dist_sides / jnp.cos(-angle - jnp.pi / 2)
                        to_fr = dist_fr / jnp.sin(-angle - jnp.pi / 2)
                        self.side_distances = self.side_distances.at[i].set(min(to_side, to_fr))

        # create multiple vehicles on instantiation:
        # Initiated as an array of shape (num_agents, 12)
        self.num_agents = num_agents
        self.agents = jnp.zeros((self.num_agents, 12))
        for i in range(num_agents):
            self.agents = self.agents.at[i].set(self.initiate_agents())

    def initiate_agents(self):
        """ New function (in JAX simulator)to initiate all agents within
            one instance of RaceCars class.
            Every agent is initialized with the same parameters.
            Indexes:
            0-6: state
            7-8: steering delay buffer
            9: lateral acceleration x
            10: lateral acceleration y
            11: Collision Indicator
             """
        agent = jnp.concatenate((
            self.state,
            # steering delay buffer
            self.steer_buffer,
            # Lateral acceleration (no independent state) a_x, a_y
            jnp.array([0]),
            jnp.array([0]),
            # collision indicator (bool)
            jnp.array([False])
        ))
        return agent

    def update_params(self, params):
        """
        Currently, params are the same for all vehicles simulated
        """
        self.params = params
        self.mu = params['mu']
        self.C_Sf = params['C_Sf']
        self.C_Sr = params['C_Sr']
        self.lf = params['lf']
        self.lr = params['lr']
        self.h = params['h']
        self.m = params['m']
        self.I = params['I']
        self.s_min = params['s_min']
        self.s_max = params['s_max']
        self.sv_min = params['sv_min']
        self.sv_max = params['sv_max']
        self.v_switch = params['v_switch']
        self.a_max = params['a_max']
        self.v_min = params['v_min']
        self.v_max = params['v_max']
        self.length = params['length']
        self.width = params['width']

    def set_map(self, map_path, map_ext):
        """
        For scan simulator
        """
        self.scan_simulator.set_map(map_path, map_ext)

    def reset(self, pose):
        """ Resets the vehicles to a pose.

        Args:
            pose (np.ndarray (3, )): pose to reset the vehicle to
        """

        # self.in_collision = False

        # reset state for all vehicles
        for i in range(self.num_agents):
            self.agents = self.agents.at[i, 0].set(pose[i, 0])
            self.agents = self.agents.at[i, 1].set(pose[i, 1])
            self.agents = self.agents.at[i, 4].set(pose[i, 2])
            self.agents = self.agents.at[i, 7].set(self.steer_buffer[0])
            self.agents = self.agents.at[i, 8].set(self.steer_buffer[1])
            # set lateral acceleration = 0
            self.agents = self.agents.at[i, 9].set(0)
            self.agents = self.agents.at[i, 10].set(0)

    def _get_vertices(self, poses):
        '''
        Wrapper function to enable calling get_vertices as vmap
        for all poses passed to this function at once.
        Returns:
            jnp.ndarray (num_agents, 4, 2): vertices of all agents.py
        '''

        return vmap_vertices_poses(poses, self.length, self.width)

    def split_poses(self, poses):
        '''
        This splits an array n poses into n pairs of (pose, remaining_poses),
        creating paris of own pose and opp_poses
        Args:
            poses:
        '''

        # Since any argument to vmap is traced by design, an input axes to vmap
        # cannot be used as an index inside the vmapped function. Therefore, this
        # has to be done in a loop sadly.

        n = self.num_agents
        opp_poses = jnp.zeros((n, n - 1, 3))

        # Creates an array  of shape (n, n-1, 3)
        # Each row is missing the pose of one car
        for i in range(n):
            _ = jnp.delete(poses, i, 0)
            opp_poses = opp_poses.at[i].set(_)

        return poses, opp_poses

    def ray_cast_agents(self, scan):
        """ Ray cast onto other agents.py in the env, modify original scan.

           Args:
               scan (jnp.ndarray, (num_agents, num_beams )): original scan range array
           Returns:
               new_scan (jnp.ndarray, (num_agents, num_beams ))): modified scan
        """

        # starting from original scan
        # jnp.array of shape(num_agents, num_beams)
        poses = self.agents[:, [0, 1, 4]]

        poses, opp_poses = self.split_poses(poses)

        new_scan = _ray_cast(poses, scan, self.scan_angles, opp_poses, self.length, self.width)

        return new_scan

    # Below are a couple of function definitions that are moved outside of update_pose
    # for vmapping

    def update_pose(self, raw_steer, vel):
        """ Steps the vehicles' physical simulation.

        Args:
            steer (float): array of desired steering angle for all cars
            vel (float): array of desired longitudinal velocity for all cars

        Returns:
            current_scan
        """
        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        # steering delay
        # steer = jnp.zeros((self.num_agents,))
        # Helper array for steer buffer vmap
        steer_buffer = self.agents[:, [7, 8]]
        # Reshape raw_steer for axis mapping in vmap
        raw_steer = jnp.reshape(raw_steer, (self.num_agents,))
        steer, steer_buffer = steer_buffer_fun(steer_buffer, raw_steer)

        # update self.agents with new steer_buffer values
        self.agents = self.agents.at[:, [7, 8]].set(steer_buffer)

        # Retrieve arrays of current speed and steer for all vehicles for use in pid
        current_speed = self.agents[:, 3]
        current_steer = self.agents[:, 2]
        # steering angle velocity input to steering velocity acceleration input
        accl, sv = pid(vel, steer, current_speed, current_steer, self.sv_max, self.a_max,
                            self.v_max, self.v_min)

        # Note: accl and sv get saturated in the integrator

        # Removed the Euler Integrator from Simulation, only using RK4 now

        # RK4 integration
        states = self.agents[:, [0, 1, 2, 3, 4, 5, 6]]
        timestep = self.time_step

        k1 = self._get_vehicle_dynamics_st(states, sv, accl)
        k2_state = states + self.time_step * (k1 / 2)
        k2 = self._get_vehicle_dynamics_st(k2_state, sv, accl)

        k3_state = states + self.time_step * (k2 / 2)
        k3 = self._get_vehicle_dynamics_st(k3_state, sv, accl)

        k4_state = states + self.time_step * k3
        k4 = self._get_vehicle_dynamics_st(k4_state, sv, accl)

        f = dynamics_integration(k1, k2, k3, k4)

        slip = self.agents[:, 6]

        self.agents = self.agents.at[:, 9].set(f[:, 3] * jnp.cos(slip))
        self.agents = self.agents.at[:, 10].set(f[:, 3] * jnp.sin(slip))

        self.agents = self.agents.at[:, [0, 1, 2, 3, 4, 5, 6]].set(
            self.agents[:, [0, 1, 2, 3, 4, 5, 6]] + self.time_step * f[:])

        # bound yaw angle
        states = self.agents[:, [0, 1, 2, 3, 4, 5, 6]]
        state = bound_yaw_angle(states)
        self.agents = self.agents.at[:, [0, 1, 2, 3, 4, 5, 6]].set(state)


        current_scan = self._get_scan()
        # Returns an array carrying current scans of all vehicles
        return current_scan

    def _update_pose(self, control_inputs):
        '''
        Wrapper for update_pose function
        Args:
            control_inputs(np.ndarray(num_agents, 2)): control
            inputs of all agents.py, first column is desired
            steering angle, second column is desired velocity
        Returns:
            jnp.array of scans of all vehicles, vmap is done within update_pose
        '''
        steer = jnp.array(control_inputs[:, 0])
        vel = jnp.array(control_inputs[:, 1])

        scan = self.update_pose(steer, vel)
        return scan

    def _get_vehicle_dynamics_st(self, states, sv, accl):
        # This function now receives an array "states" which carries
        # multiple arrays, each representing the state of one of the
        # agents of the simulation
        mu = self.mu
        C_Sf = self.C_Sf
        C_Sr = self.C_Sr
        lf = self.lf
        lr = self.lr
        h = self.h
        m = self.m
        I = self.I
        s_min = self.s_min
        s_max = self.s_max
        sv_min = self.sv_min
        sv_max = self.sv_max
        v_switch = self.v_switch
        a_max = self.a_max
        v_min = self.v_min
        v_max = self.v_max

        f = f_getter(states, sv, accl, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min,
                          sv_max, v_switch, a_max, v_min, v_max)

        return f

    # This is not ideal, because type of scan_simulator is an attribute of racecar
    # and should not be hardcoded
    vmap_scan = vmap(ScanSimulator2D.scan, in_axes=0)

    def _get_scan(self):
        '''
        Wrapper for scan simulator scan
        Returns:
            current_scan as array for all vehicles
        '''
        # Map scan function over axis 0 (agents)
        # Retrieve all states
        states = self.agents[:, [0, 1, 2, 3, 4, 5, 6]]
        # create an array that carries the pose of each agent
        pose = states[:, [0, 1, 4]]

        current_scan = vmap_scan(pose)
        return current_scan

    def update_scan(self, agent_scan):
        """ Steps the vehicle's laser scan simulation.

        Separated from update_pose because needs to update scan based on NEW poses of agents.py in the environment

        Args:
            agents.py scans list (modified in-place),
            agents.py index (int)
        """
        # Do not call as vmap, but rather vmap within this function. Return new_scans for all vehicles from this function

        # jnp.array of shape(num_agents, num_beams)
        current_scan = agent_scan

        # check ttc
        # This function sets certain state values to zero if a collision is detected
        in_collision, vel = check_ttc(current_scan, self.agents[:, 3], self.scan_angles, self.cosines,
                                           self.side_distances, self.ttc_thresh)
        self.agents = self.agents.at[:, 3].set(vel)  # velocity unchanged or set to zero (in event of collision)

        # ray cast other agents.py to modify scan
        new_scans = self.ray_cast_agents(current_scan)
        agent_scans = new_scans
        return agent_scans


class Simulator(object):
    """Simulator class, handles the interaction and update of all vehicles in the environment.
    Attributes:
        num_agents (int): number of agents in the environment
        time_step (float): physics time step
        agent_poses (np.ndarray(num_agents, 3)): all poses of all agents.py
        cars (list[RaceCars]): container for RaceCar objects
        collisions (np.ndarray(num_agents, )): array of collision indicator for each agents.py
        collision_idx (np.ndarray(num_agents, )): which agents.py is each agents.py in collision with

    """

    def __init__(self, params, num_agents, seed, time_step=0.01, ego_idx=0, integrator=Integrator.RK4):
        """Init function.

        Args: params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I',
        's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 'a_max', 'v_min', 'v_max', 'length', 'width'} num_agents (
        int): number of agents.py in the environment seed (int): seed of the rng in scan simulation time_step (float,
        default=0.01): physics time step ego_idx (int, default=0): ego vehicle's index in list of agents.py
        """
        self.num_agents = num_agents
        self.seed = seed
        self.time_step = time_step
        self.ego_idx = ego_idx
        self.params = params
        self.length = params['length']
        self.width = params['width']
        self.agent_poses = jnp.empty((self.num_agents, 3))

        self.collisions = jnp.zeros((self.num_agents,))
        # self.collision_idx = -1 * jnp.ones((self.num_agents,))
        # self.overtaking_idx = jnp.zeros((self.num_agents,))

        self.integrator = integrator

        # initializing agents.py
        # The new simulator handles every agent within one instance of the RaceCars class
        # The ego vehicle is defined as the one having idx 0
        # All other vehicles are defined as non-ego vehicles and follow in the list

        self.cars = RaceCars(params, self.seed, time_step=self.time_step, num_agents=self.num_agents, num_beams=1080,
                             fov=4.7)

    def set_map(self, map_path, map_ext):
        """Sets the map of the environment and sets the map for scan simulator of all agents.py.

        Args:
            map_path (str): path to the map yaml file
            map_ext (str): extension for the map image file
        """
        self.cars.set_map(map_path, map_ext)

    def update_params(self, params, agent_idx=-1):
        """Updates the params of agents.py, if an index of an agents.py is given, update only that agents.py's params

        Args:
            params (dict): dictionary of params, see details in docstring of __init__
            agent_idx (int, default=-1): index for agents.py that needs param update, if negative, update all agents.py
        """

        # The RaceCar.update_params function does only support general updates right now
        # We are currently assuming that all agents have the same params at all times
        # Individual params are not stored within the agents right now, only within RaceCars class

        # if agent_idx < 0:
        # update params for all
        #   for agent in self.agents:
        #       agent.update_params(params)
        # elif 0 <= agent_idx < self.num_agents:
        # only update one agents.py's params
        #    self.agents[agent_idx].update_params(params)
        # else:
        # index out of bounds, throw error
        #    raise IndexError('Index given is out of bounds for list of agents.py.')

        self.cars.update_params(self, params)

    def check_collision(self):
        """Checks for collision between agents.py using GJK and agents.py' body vertices."""
        all_vertices = vmap_vertices(self.agent_poses, self.length, self.width)
        # collisions = np.zeros_like(len(self.agent_poses))
        collisions = collision_multiple(all_vertices)
        return collisions

    def check_overtaking(self):

        # Not currently in use

        # Get relevant states for overtaking success
        # ego_vehicle (x, y, steering_angle)
        ego_vehicle_xy_steer = self.cars.agents[0, [0, 1, 4]]
        # other_vehicle (x, y)
        other_vehicle_xy = self.cars.agents[1:, [0, 1]]

        success = self._compare_positions_with_direction(
            ego_vehicle_xy_steer,
            other_vehicle_xy
        )
        return success

    def _compare_positions_with_direction(self, ego_vehicle_state, other_vehicle_states):
        # Does not work with JAX right now
        x_a = ego_vehicle_state[0]
        y_a = ego_vehicle_state[1]
        psi_a = ego_vehicle_state[2]
        x_b = other_vehicle_states[:, 0]
        y_b = other_vehicle_states[:, 1]
        diff_x = jnp.array([x_b - x_a])
        diff_y = jnp.array([y_b - y_a])

        distances = jnp.sqrt(diff_x ** 2 + diff_y ** 2)
        indices = jnp.where(distances < 3.0)[0]
        jax.debug.breakpoint()
        if indices.size == 0:
            return False

        for idx in indices:
            diff_y_ = [diff_y[idx]]
            diff_x_ = [diff_x[idx]]
            # Get angle in radians from x-axis to point2 with the positive direction being counter-clockwise
            angle_to_point_b = jnp.arctan2(diff_y_, diff_x_)

            # bound theta_a effectively between -pi and pi
            theta_a = pi_2_pi(psi_a)
            diff_angle = pi_2_pi(angle_to_point_b - psi_a)

            # Get the relative angle from vehicle A's orientation to vehicle B
            relative_angle = jnp.rad2deg(diff_angle)

            # print(f"Relative angle: {relative_angle}")

            success = False
            # JAX simulator does not suppor overtaking_idx
            if -90.0 <= relative_angle <= 90.0:
                # print("Vehicle B is in front")
                # Overtaking has started
                self.overtaking_idx[idx + 1] = 1
            else:
                # print("Vehicle A is in front")
                if self.overtaking_idx[idx + 1] and distances[idx] > 1.2:
                    self.overtaking_idx[idx + 1] = 0
                    success = True
            return success

    def step(self, control_inputs):
        """Steps the simulation environment.

        Args:
            control_inputs (np.ndarray (num_agents, 2)): control inputs of all agents.py,
            first column is desired steering angle, second column is desired velocity

        Returns:
            observations (dict): dictionary for observations: poses of agents.py,
            current laser scan of each agents.py, collision indicators, etc.
        """

        # Instead of looping over agents to call .update_pose
        # as in the legacy simulator, this is now
        # done within one vectorized step

        # update pose for every agent in one step
        # output now vectorized
        # scan is pose: [x, y, yaw]
        agent_scans = self.cars._update_pose(control_inputs)

        # update sim's information of agents poses
        poses = self.cars.agents[:, [0, 1, 4]]
        self.agent_poses = self.agent_poses.at[:].set(poses)

        # check collisions between all agents.py
        collisions_agents = self.check_collision()
        self.cars.agents = self.cars.agents.at[:, 11].set(collisions_agents)
        # self.collision_idx = collision_idx_agents

        # collision with environment
        collisions_env = jnp.zeros((self.num_agents,))

        new_scans = self.cars.update_scan(agent_scans)
        agent_scans = agent_scans.at[:].set(new_scans)

        # collisions with environment
        collisions_env = jnp.where(self.cars.agents[:, 11], 1., 0.)

        overtaking_success = False  # self.check_overtaking()

        observations = dict()
        observations['aaa_scans'] = jnp.array([scan for scan in agent_scans])

        observations['poses_x'] = self.cars.agents[:, 0]
        observations['poses_y'] = self.cars.agents[:, 1]

        observations['steering_angle'] = self.cars.agents[:, 2]
        observations['linear_vels_x'] = jnp.cos(self.cars.agents[:, 6]) * self.cars.agents[:, 3]
        observations['linear_vels_y'] = jnp.sin(self.cars.agents[:, 6]) * self.cars.agents[:, 3]  # there was some if else logic in legacy here

        observations['poses_theta'] = self.cars.agents[:, 4]
        observations['yaw_rate'] = self.cars.agents[:, 5]
        observations['ang_vels_z'] = self.cars.agents[:, 5]
        observations['slip_angle'] = self.cars.agents[:, 6]

        observations['acc_x'] = self.cars.agents[:, 9]
        observations['acc_y'] = self.cars.agents[:, 10]

        observations['collisions'] = jax.lax.convert_element_type(self.collisions, jnp.bool_)
        observations['collisions_env'] = jax.lax.convert_element_type(collisions_env, jnp.bool_)
        observations['collisions_agents'] = jax.lax.convert_element_type(self.cars.agents[:, 11], jnp.bool_)

        # that obs is only for the ego vehicle
        observations['overtaking_success'] = jnp.array([bool(overtaking_success)])
        return observations

    def reset(self, poses):
        """Resets the simulation environment by given poses."""

        if poses.shape[0] != self.num_agents:
            raise ValueError('Number of poses for reset does not match number of agents.py.')

        # loop over poses to reset
        self.cars.reset(poses)
        # Dummy action to get the first observation
        num = self.num_agents
        obs = self.step(jnp.zeros((num, 2)))
        return obs
