import jax.numpy as jnp

from planning.planner_utils import *


class PlannerBase:
    """Base class for planners."""

    def __init__(self, conf):
        self.conf = conf
        self.drawn_waypoints = []

    def plan(self, state, waypoints):
        """Plan a trajectory. Returns a list of (x, y, v) tuples."""
        raise NotImplementedError()

    def render_waypoints(self, e, waypoints):
        """Render the waypoints e."""
        points = jnp.vstack((waypoints[:, self.conf.maps.wpt_xind], waypoints[:, self.conf.maps.wpt_yind])).T

        scaled_points = 50. * points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                from pyglet.gl import GL_POINTS
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]


class PurePursuitPlanner(PlannerBase):
    """Planner that uses pure pursuit to follow a trajectory."""

    def __init__(self, config, vehicle_params):
        super().__init__(config)
        # Config for the planner
        self.config = config
        self.tlad = config.tlad
        self.vgain = config.vgain
        self.vmax = config.vmax
        self.n_next_points = config.n_next_points
        self.skip_next_points = config.skip_next_points

        # Vehicle parameters from the car model
        self.vehicle_params = vehicle_params
        self.wheelbase = vehicle_params.lf + vehicle_params.lr

        self.max_reacquire = 20.
        self.drawn_waypoints = []

    def plan(self, obs, waypoints):
        """Plan a trajectory to follow the waypoints.

        Args:
            obs (dict): observation dict
            waypoints (np.ndarray): waypoints to follow
        Returns:
            steering (float): steering angle
            velocity (float): velocity
            lookahead_points_relative (np.ndarray): relative lookahead points
        """
        poses_global = jnp.array(
            [
                obs['poses_x'][0],
                obs['poses_y'][0],
                obs['poses_theta'][0]
            ],
            dtype=jnp.float64
        )

        waypoints = get_current_waypoint(
            waypoints,
            20,
            self.config.tlad,
            jnp.array([obs['poses_x'][0], obs['poses_y'][0]]),
            n_next_points=self.config.n_next_points
        )

        if waypoints is None:
            return 0.0, 0.0, jnp.array([[0.0, 0.0, 0.0]])

        # Skip some waypoints since they are very close together
        lookahead_points_relative = (waypoints[::self.skip_next_points, :2] - poses_global[:2]).flatten().astype(
            jnp.float64)

        speed, steering = get_actuation(
            poses_global[2], waypoints[0, [0, 1, 3]], poses_global[:2], self.tlad, self.wheelbase
        )
        velocity = self.vgain * speed

        if self.vmax != 'None':
            velocity = jnp.clip(velocity, 0, self.vmax)

        return steering, velocity, lookahead_points_relative
