import jax.numpy as jnp
import jax
from jax import jit


def render_callback(env_renderer, planner, waypoints=None):
    """Callback function for rendering the environment by updating the camera to follow the car.
    Args:
        env_renderer (f110_gym.envs.env_renderer.EnvRenderer): Environment renderer.
        planner (planner.Planner): Planner.
    """
    x = env_renderer.cars[0].vertices[::2]
    y = env_renderer.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    env_renderer.score_label.x = left
    env_renderer.score_label.y = top - 700
    env_renderer.left = left - 800
    env_renderer.right = right + 800
    env_renderer.top = top + 800
    env_renderer.bottom = bottom - 800
    # planner.render_waypoints(env_renderer)


@jit
def nearest_point_on_trajectory(point, trajectory):
    """Return the nearest point along the given piecewise linear trajectory.

    Note: Trajectories must be unique. If they are not unique, a divide by 0 error will occur

    Args:
        point(np.ndarray): size 2 numpy array
        trajectory: Nx2 matrix of (x,y) trajectory waypoints

    Returns:
        projection(np.ndarray): size 2 numpy array of the nearest point on the trajectory
        dist(float): distance from the point to the projection
        t(float): the t value of the projection along the trajectory
        min_dist_segment(int): the index of the segment of the trajectory that the projection is on
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    # Equivalent to dot product
    t = jnp.sum((point - trajectory[:-1, :]) * diffs, axis=1) / (diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
    t = jnp.clip(t, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    temp = point - projections
    dists = jnp.sqrt(jnp.sum(temp * temp, axis=1))
    min_dist_segment = jnp.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


# TODO: remove for_loop to jit
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """Return the first point along the given piecewise linear trajectory that intersects the given circle.

    Assumes that the first segment passes within a single radius of the point
    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm

    Args:
        point(np.ndarray): size 2 numpy array
        radius(float): radius of the circle
        trajectory: Nx2 matrix of (x,y) trajectory waypoints
        t(float): the t value of the trajectory to start searching from
        wrap(bool): if True, wrap the trajectory around to the beginning if the end is reached

    Returns:
        projection(np.ndarray): size 2 numpy array of the nearest point on the trajectory
        dist(float): distance from the point to the projection
        t(float): the t value of the projection along the trajectory
        min_dist_segment(int): the index of the segment of the trajectory that the projection is on
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    #trajectory = jax.device_put(jnp.array([trajectory]))
    trajectory = jnp.array(trajectory)

    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        # V has been stored ascontigouuous array in legacy
        V = jnp.array(end - start)
        a = jnp.dot(V, V)
        b = 2.0 * jnp.dot(V, start - point)
        c = jnp.dot(start, start) + jnp.dot(point, point) - 2.0 * jnp.dot(start, point) - radius * radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = jnp.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start
            a = jnp.dot(V, V)
            b = 2.0 * jnp.dot(V, start - point)
            c = jnp.dot(start, start) + jnp.dot(point, point) - 2.0 * jnp.dot(start, point) - radius * radius
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = jnp.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t



def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = jnp.dot(jnp.array([jnp.sin(-pose_theta), jnp.cos(-pose_theta)]), lookahead_point[0:2] - position)
    speed = lookahead_point[2]
    if jnp.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1 / (2.0 * waypoint_y / lookahead_distance ** 2)
    steering_angle = jnp.arctan(wheelbase / radius)
    return speed, steering_angle


@jit
def pi_2_pi(angle):
    if angle > jnp.pi:
        return angle - 2.0 * jnp.pi
    if angle < -jnp.pi:
        return angle + 2.0 * jnp.pi
    return angle


@jit
def pi_2_pi2(angle):
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    return jnp.arctan2(sin_angle, cos_angle)


@jit
def quat_2_rpy(x, y, z, w):
    """
    Converts a quaternion into euler angles (roll, pitch, yaw)

    Args:
        x, y, z, w (float): input quaternion

    Returns:
        r, p, y (float): roll, pitch yaw
    """
    t0 = 2. * (w * x + y * z)
    t1 = 1. - 2. * (x * x + y * y)
    roll = jnp.atan2(t0, t1)

    t2 = 2. * (w * y - z * x)
    t2 = 1. if t2 > 1. else t2
    t2 = -1. if t2 < -1. else t2
    pitch = jnp.asin(t2)

    t3 = 2. * (w * z + x * y)
    t4 = 1. - 2. * (y * y + z * z)
    yaw = jnp.atan2(t3, t4)
    return roll, pitch, yaw



def get_current_waypoint(waypoints, max_reacquire, lookahead_distance, position, n_next_points=5):
    """Get the current waypoint.

    Args:
        waypoints(np.ndarray): the waypoints
        lookahead_distance(float): the lookahead distance
        position(np.ndarray): the current position
        theta(float): the current pose angle
        n_next_points(int): the number of next points to return

    Returns:
        current_waypoint(np.ndarray): the current waypoint
    """

    wpts = jnp.stack((waypoints[:, 1], waypoints[:, 2], waypoints[:, 3], waypoints[:, 5])).T

    nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts[:, 0:2])
    if nearest_dist < lookahead_distance:
        lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
            position, lookahead_distance, wpts[:, 0:2],
            i + t, wrap=True
        )
        if i2 is None:
            return None

        current_waypoint = jnp.empty((n_next_points, 4))

        if i2 + n_next_points > wpts.shape[0]:
            o = i2 + n_next_points - wpts.shape[0]
            # current_waypoint[:n_next_points - o, :] = wpts[i2:, :]
            current_waypoint = current_waypoint.at[:n_next_points - o, :].set(wpts[i2:, :])
            if o != 0:
                # current_waypoint[n_next_points - o:, :] = wpts[:o, :]
                current_waypoint = current_waypoint.at[n_next_points - o:, :].set(wpts[:o, :])
        else:
            if i2 < 0:
                #current_waypoint[0:(-i2), :] = wpts[i2 - 1:i2, :]
                # current_waypoint[(-i2):, :] = wpts[0:i2 + n_next_points, :]
                current_waypoint = current_waypoint.at[0:(-i2), :].set(wpts[i2 - 1:i2, :])
                current_waypoint = current_waypoint.at[(-i2):, :].set(wpts[0:i2 + n_next_points, :])

            else:
                #current_waypoint[:, :] = wpts[i2:i2 + n_next_points, :]
                current_waypoint = current_waypoint.at[:, :].set(wpts[i2:i2 + n_next_points, :])

        return current_waypoint

    elif nearest_dist < max_reacquire:
        current_waypoint = jnp.empty((n_next_points, 4))

        if i + n_next_points > wpts.shape[0]:
            o = i + n_next_points - wpts.shape[0]
            # x, y
            # current_waypoint[:n_next_points - o, :] = wpts[i:, :]
            current_waypoint = current_waypoint.at[:n_next_points - o, :].set(wpts[i:, :])
            if o != 0:
                # current_waypoint[n_next_points - o:, :] = wpts[:o, :]
                current_waypoint = current_waypoint.at[n_next_points - o:, :].set(wpts[:o, :])

        else:
            # current_waypoint[:, :] = wpts[i:i + n_next_points, :]
            current_waypoint = current_waypoint.at[:, :].set(wpts[i:i + n_next_points, :])
        return current_waypoint
    else:
        return None


@jit
def curvature(x, y, i):
    # Get points before and after the current point
    x_prev, y_prev = x[i - 1], y[i - 1]
    x_curr, y_curr = x[i], y[i]
    x_next, y_next = x[i + 1], y[i + 1]

    # Calculate the derivatives
    dxdt = (x_next - x_prev) / 2.0
    dydt = (y_next - y_prev) / 2.0
    dx2dt2 = x_next - 2 * x_curr + x_prev
    dy2dt2 = y_next - 2 * y_curr + y_prev

    # Calculate curvature using the formula
    k = (dxdt * dy2dt2 - dydt * dx2dt2) / ((dxdt ** 2 + dydt ** 2) ** (3 / 2))

    return k


def moving_average(data, window_size):
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


@jit
def distance_between_points(point1, point2):
    return jnp.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


@jit
def find_close_obstacles(current_pos, obstacles):
    # Calculate the distance to the closest obstacle for each of the corners
    t_vec_all_length_center = jnp.clip(jnp.sqrt(jnp.sum((current_pos - obstacles) ** 2, axis=1)), 0.01, 100)
    close_ob = obstacles[jnp.argsort(t_vec_all_length_center)[:50]]

    # t_vec_all_length = np.zeros((obstacles.shape[0], 6))
    # t_vec_all_length[:, 0] = np.sqrt(np.sum((current_pos - np.array([0.15, 0.25]) - obstacles) ** 2, axis=1))
    # t_vec_all_length[:, 1] = np.sqrt(np.sum((current_pos - np.array([-0.15, 0.25]) - obstacles) ** 2, axis=1))
    # t_vec_all_length[:, 2] = np.sqrt(np.sum((current_pos - np.array([0.15, -0.25]) - obstacles) ** 2, axis=1))
    # t_vec_all_length[:, 3] = np.sqrt(np.sum((current_pos - np.array([-0.15, -0.25]) - obstacles) ** 2, axis=1))
    # t_vec_all_length[:, 4] = np.sqrt(np.sum((current_pos - np.array([0.15, 0.0]) - obstacles) ** 2, axis=1))
    # t_vec_all_length[:, 5] = np.sqrt(np.sum((current_pos - np.array([-0.15, 0.0]) - obstacles) ** 2, axis=1))
    # t_vec_all_length = np.clip(t_vec_all_length, 0.01, 100)
    # Select the 50 closest obstacles
    # t_vec_all_length_idx = np.argmin(t_vec_all_length, axis=0)
    # close_ob = obstacles[t_vec_all_length_idx]
    return close_ob
