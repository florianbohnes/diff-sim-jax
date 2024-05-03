import jax.debug
import jax.numpy as jnp
from jax import jit
from jax import lax
from jax import vmap
from scipy.ndimage import distance_transform_edt as edt
from PIL import Image
import os
import yaml
from jax import random
from functools import partial


def get_dt(bitmap, resolution):
    """ Distance transformation of the input bitmap.

        Uses scipy.ndimage, cannot be JITted.

        Args:
            - bitmap (numpy.ndarray, (n, m)): input binary bitmap of the environment, where 0 is obstacles, and 255 (or
                anything > 0) is freespace
            - resolution (float): resolution of the input bitmap (m/cell)

        Returns: dt (numpy.ndarray, (n, m)): output distance matrix, where each cell has the corresponding distance (in
        meters) to the closest obstacle
        """
    dt = resolution * edt(bitmap)
    dt = jnp.array(dt)
    return dt


@jit
def xy_2_rc(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution):
    """
    Translate (x, y) coordinate into (r, c) in the matrix.

    Args:
        x (float): coordinate in x (m)
        y (float): coordinate in y (m)
        orig_x (float): x coordinate of the map origin (m)
        orig_y (float): y coordinate of the map origin (m)

    Returns:
        r (int): row number in the transform matrix of the given point
        c (int): column number in the transform matrix of the given point
    """

    # translation
    x_trans = x - orig_x
    y_trans = y - orig_y

    # rotation
    x_rot = x_trans * orig_c + y_trans * orig_s
    y_rot = -x_trans * orig_s + y_trans * orig_c

    # clip the state to be a cell
    def true_fun(x_rot, y_rot, resolution):
        c = -1
        r = -1
        return c, r

    def false_fun(x_rot, y_rot, resolution):
        c = (x_rot / resolution).astype(int)
        r = (y_rot / resolution).astype(int)
        return c, r

    c, r = lax.cond(
        x_rot < 0,
        lambda _: true_fun(x_rot, y_rot, resolution),
        lambda _: lax.cond(
            x_rot >= width * resolution,
            lambda _: true_fun(x_rot, y_rot, resolution),
            lambda _: lax.cond(
                y_rot < 0,
                lambda _: true_fun(x_rot, y_rot, resolution),
                lambda _: lax.cond(
                    y_rot >= height * resolution,
                    lambda _: true_fun(x_rot, y_rot, resolution),
                    lambda _: false_fun(x_rot, y_rot, resolution),
                    None
                ),
                None
            ),
            None
        ),
        None
    )

    return r, c


@jit
def distance_transform(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt):
    """
    Look up corresponding distance in the distance matrix.

    Args:
        x (float): x coordinate of the lookup point
        y (float): y coordinate of the lookup point
        orig_x (float): x coordinate of the map origin (m)
        orig_y (float): y coordinate of the map origin (m)

    Returns:
        distance (float): corresponding shortest distance to obstacle in meters
    """
    r, c = xy_2_rc(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution)
    distance = dt[r, c]

    return distance

@jit
def trace_ray_loop_body(carry):
    (dist_to_nearest, eps, total_dist, max_range, xy_tuple, orig_x, orig_y, orig_c, orig_s,
     height, width, resolution, dt, s, c) = carry

    x, y = xy_tuple
    x += dist_to_nearest * c
    y += dist_to_nearest * s

    dist_to_nearest = distance_transform(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt)
    total_dist += dist_to_nearest
    return (dist_to_nearest, eps, total_dist, max_range, (x, y), orig_x, orig_y, orig_c, orig_s,
     height, width, resolution, dt, s, c)


def trace_ray_loop_cond(carry):
    (dist_to_nearest, eps, total_dist, max_range, xy_tuple, orig_x, orig_y, orig_c, orig_s,
     height, width, resolution, dt, s, c) = carry
    return lax.bitwise_and(dist_to_nearest > eps, total_dist <= max_range)


@jit
def trace_ray(x, y, theta_index, sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt,
              max_range):
    """
    Find the length of a specific ray at a specific scan angle theta.

    Purely math calculation and loops, should be JITted.

    Args:
        x (float): current x coordinate of the ego (scan) frame
        y (float): current y coordinate of the ego (scan) frame
        theta_index(int): current index of the scan beam in the scan range
        sines (numpy.ndarray (n, )): pre-calculated sines of the angle array
        cosines (numpy.ndarray (n, )): pre-calculated cosines ...

    Returns:
        total_distance (float): the distance to first obstacle on the current scan beam
    """

    # int casting, and index precal trigs
    theta_index_ = theta_index.astype(int)

    s = sines[theta_index_]
    c = cosines[theta_index_]

    # distance to nearest initialization
    dist_to_nearest = distance_transform(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt)
    total_dist = dist_to_nearest

    init_carry = (dist_to_nearest, eps, total_dist, max_range, (x, y), orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt, s, c)

    final_carry = lax.while_loop(trace_ray_loop_cond, trace_ray_loop_body, init_carry)

    total_dist = lax.cond(total_dist > max_range,
                          lambda _: max_range,
                          lambda _: total_dist,
                          total_dist)

    return total_dist


vmap_tracer = jit(vmap(trace_ray,
                   in_axes=(None, None, 0, None, None, None, None, None, None, None, None, None, None, None, None)))


@jit
def get_scan_cond(init):
    theta_index, _ = init # unpack
    return theta_index < 0


@jit
def get_scan_loop_body(init):
    theta_index, theta_dis = init # unpack
    return theta_index + theta_dis, theta_dis


@partial(jit, static_argnames='num_beams')
def get_scan(pose, theta_dis, fov, num_beams, theta_index_increment, sines, cosines, eps, orig_x, orig_y, orig_c,
             orig_s, height, width, resolution, dt, max_range):
    '''
    Perform the scan for each discrete angle of each beam of the laser.

    Loop heavy, should be JITted

    Args:
        pose (numpy.ndarray(3, )): current pose of the scan frame in the map
        theta_dis (int): number of steps to discretize the angles between 0 and 2pi for look up
        fov (float): field of view of the laser scan
        num_beams (int): number of beams in the scan
        theta_index_increment (float): increment between angle indices after discretization
        sines (numpy.ndarray (n, )): pre-calculated sines of the angle array
        cosines (numpy.ndarray (n, )): pre-calculated cosines ...
        eps (float): epsilon for distance transform
        orig_x (float): x coordinate of the map origin (m)
        orig_y (float): y coordinate of the map origin (m)
        orig_c (float): cosine of the map origin rotation
        orig_s (float): sine of the map origin rotation
        height (int): height of the map in cells
        width (int): width of the map in cells
        resolution (float): resolution of the map in meters/cell
        dt (numpy.ndarray (height, width)): distance transform matrix
        max_range (float): maximum range of the laser

    Returns:
        scan (jnp.array(n, )): resulting laser scan at the pose, n=num_beams
    '''
    # make theta discrete by mapping the range [-pi, pi] onto [0, theta_dis]
    theta_index_init = theta_dis * (pose[2] - fov / 2.) / (2. * jnp.pi)
    # make sure it's wrapped properly
    theta_index_init = jnp.fmod(theta_index_init, theta_dis)

    # The loop below is equivalent to this python:
    #     while theta_index < 0:
    #         theta_index += theta_dis
    init = theta_index_init, theta_dis
    theta_index_, _ = lax.while_loop(get_scan_cond, get_scan_loop_body, init)

    # theta_index_arr = jnp.empty((num_beams, )).at[0].set(theta_index_)
    theta_index_arr = jnp.linspace(theta_index_, theta_index_ + (num_beams * theta_index_increment), num_beams,
                                   endpoint=False)

    # clip array to ensure it's within the range [0, theta_dis)
    theta_index_arr_ = jnp.minimum(theta_index_arr, theta_dis)

    scan = vmap_tracer(pose[0], pose[1], theta_index_arr_, sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height,
                       width, resolution, dt, max_range)

    return scan


@jit
@partial(vmap, in_axes=(None, 0, 0, 0, None))
def check(vel, cosine, side_distance, scan, ttc_thresh):
    proj_vel = vel * cosine
    ttc = (scan - side_distance) / proj_vel
    in_collision = lax.cond(ttc < ttc_thresh,
                            lambda _: lax.cond(ttc >= 0.0,
                                               lambda _: True,
                                               lambda _: False,
                                               ttc),
                            lambda _: False,
                            ttc)
    return in_collision


@jit
def call_check(vel, cosines, side_distances, scan, ttc_thresh):
    num_beams = scan.shape[0]
    # check is done for every angle at the same time
    # therefore, result is array with booleans
    # return True if any is true
    bool_array = check(vel, cosines, side_distances, scan, ttc_thresh)
    return jnp.any(bool_array)


@jit
def check_ttc_jit(scan, vel, scan_angles, cosines, side_distances, ttc_thresh):
    """Checks the iTTC of each beam in a scan for collision with environment.

        Args:
            scan (np.ndarray(num_beams, )): current scan to check
            vel (float): current velocity
            scan_angles (np.ndarray(num_beams, )): precomped angles of each beam
            cosines (np.ndarray(num_beams, )): precomped cosines of the scan angles
            side_distances (np.ndarray(num_beams, )): precomped distances at each beam from the laser to the sides of the car
            ttc_thresh (float): threshold for iTTC for collision

        Returns:
            in_collision (bool): whether vehicle is in collision with environment
        """

    in_collision = call_check(vel, cosines, side_distances, scan, ttc_thresh)
    return in_collision


@jit
def cross(v1, v2):
    """Cross product of two 2-vectors.

        Args:
            v1, v2 (jnp.array(2, )): input vectors

        Returns:
            crossproduct (float): cross product
        """
    return v1[0] * v2[1] - v1[1] * v2[0]


@jit
def are_colinear(pt_a, pt_b, pt_c):
    """Checks if three points are colinear.

    Args:
        pt_a, pt_b, pt_c (jnp.array(2, )): input points

    Returns:
        colinear (bool): whether points are colinear
    """
    v1 = pt_b - pt_a
    v2 = pt_c - pt_a
    return cross(v1, v2) == 0


@jit
def get_range_true_fun(vars):
    v1, v2, v3, denom, o, va, vb, distance = vars
    d1 = cross(v2, v1) / denom
    d2 = v1.dot(v3) / denom
    distance = lax.cond(d1 >= 0.0,
                        lambda _: lax.cond(d2 >= 0.0,
                                           lambda _: lax.cond(d2 <= 1.0,
                                                              lambda _: d1,
                                                              lambda _: distance,
                                                              d1),
                                           lambda _: distance,
                                           d1),
                        lambda _: distance,
                        d1)
    return distance


@jit
def get_range_false_fun(vars):
    v1, v2, v3, denom, o, va, vb, distance = vars
    distance = lax.cond(
        are_colinear(o, va, vb),
        lambda _: jnp.min(jnp.array(jnp.linalg.norm(va - o), jnp.linalg.norm(vb - o))),
        lambda _: distance,
        distance
    )
    return distance


def get_range(pose, beam_theta, va, vb):
    """Get the distance at a beam angle to the vector formed by two of the four vertices of a vehicle.

    Args:
        pose (np.ndarray(3, )): pose of the scanning vehicle
        beam_theta (float): angle of the current beam (world frame)
        va, vb (np.ndarray(2, )): the two vertices forming an edge

    Returns:
        distance (float): smallest distance at beam theta from scanning pose to edge
    """
    o = pose[0:2]
    v1 = o - va
    v2 = vb - va
    v3 = jnp.array([jnp.cos(beam_theta + jnp.pi / 2), jnp.sin(beam_theta + jnp.pi / 2)])

    denom = v2.dot(v3)
    distance = jnp.inf

    vars = v1, v2, v3, denom, o, va, vb, distance

    distance = lax.cond(jnp.fabs(denom) > 0.0,
                        get_range_true_fun,
                        get_range_false_fun,
                        vars)

    return distance


@jit
def angle_logic(angle):
    angles_with_x = lax.cond(angle > 2 * jnp.pi,
                             lambda _: angle - 2 * jnp.pi,
                             lambda _: lax.cond(angle < - jnp.pi,
                                                lambda _: angle + 2 * jnp.pi,
                                                lambda _: angle,
                                                angle),
                             angle)
    return angles_with_x


vmap_angles = vmap(angle_logic, in_axes=(0,))


@jit
def get_blocked_view_indices(pose, vertices, scan_angles):
    """Get the indices of the beams that are blocked by the vehicle.

    Args:
        pose (np.ndarray(3, )): pose of the scanning vehicle
        vertices (np.ndarray(4, 2)): vertices of the vehicle
        scan_angles (np.ndarray(num_beams, )): angles of the beams

    """
    # find four vectors formed by pose and 4 vertices
    vecs = vertices - pose[0:2]
    vec_sq = jnp.square(vecs)
    norms = jnp.sqrt(jnp.sum(vec_sq, axis=1))
    unit_vecs = vecs / norms.reshape(norms.shape[0], 1)

    # find angles between all four and pose vector
    ego_x_vec = jnp.array([[jnp.cos(pose[2])], [jnp.sin(pose[2])]])

    angles_with_x = jnp.empty((4,))

    angles = jnp.arctan2(ego_x_vec[1], ego_x_vec[0]) - jnp.arctan2(unit_vecs[:, 1], unit_vecs[:, 0])
    angles_with_x = vmap_angles(angles)

    ind1 = jnp.argmin(jnp.abs(scan_angles - angles_with_x[0]))
    ind2 = jnp.argmin(jnp.abs(scan_angles - angles_with_x[1]))
    ind3 = jnp.argmin(jnp.abs(scan_angles - angles_with_x[2]))
    ind4 = jnp.argmin(jnp.abs(scan_angles - angles_with_x[3]))
    inds = jnp.array([ind1, ind2, ind3, ind4])

    min_ind = jnp.min(inds)
    max_ind = jnp.max(inds)

    return min_ind, max_ind


@jit
@partial(vmap, in_axes=(None, 0, None, 0))
def beams(pose, scan, vertices, scan_angles):
    # create two arrays with vertices shifted by one position
    # this way there are two different vertices available in the vmap one level below
    vertices1 = vertices
    vertices2 = jnp.empty((4, 2))
    vertices2 = vertices2.at[0:3, :].set(vertices[1:4, :])
    vertices2 = vertices2.at[3, :].set(vertices[0, :])
    scan = vmap_vertices(pose, scan_angles, vertices1, vertices2, scan)
    scan = jnp.min(scan)
    return scan


# for vmap over vertices
def vertices_fun(pose, scan_angle, vertex1, vertex2, scan):
    # checks if original scan is longer than ray-casted distance
    scan_range = get_range(pose, pose[2] + scan_angle, vertex1, vertex2)
    scan = lax.cond(scan_range < scan,
                    lambda _: scan_range,
                    lambda _: scan,
                    scan_range)
    return scan


vmap_vertices = vmap(vertices_fun, in_axes=(None, None, 0, 0, None))


@jit
@partial(vmap, in_axes=(None, None, None, 0))
def ray_cast(pose, scan, scan_angles, vertices):
    """Modify a scan by ray casting onto another agents.py's four vertices.

    Args:
        pose (np.ndarray(3, )): pose of the vehicle performing scan
        scan (np.ndarray(num_beams, )): original scan to modify
        scan_angles (np.ndarray(num_beams, )): corresponding beam angles
        vertices (np.ndarray(4, 2)): four vertices of a vehicle pose

    Returns:
        new_scan (jnp.array(num_beams, )): modified scan
    """
    # Result of get_blocked_view_indices could not be used here because ray_cast function is called within
    # a vmap, so all values inside this function are only available as tracer values. Tracer values cannot be used for
    # indexing and array sizes can also not be dynamic, but must be statically known at compile time.
    # For this reason, ray cast is currently not performed only on the view indices where it would be necessary,
    # but on the entire scan, only updating where necessary. This is not efficient, but it works for now.

    # min_ind, max_ind = get_blocked_view_indices(pose, vertices, scan_angles)

    # for vmap over beams

    scan = beams(pose, scan, vertices, scan_angles)

    return scan


class ScanSimulator2D(object):
    """2D LIDAR scan simulator class.

   Attributes:
       num_beams (int): number of beams in the scan
       fov (float): field of view of the laser scan
       eps (float, default=0.0001): ray tracing iteration termination condition
       theta_dis (int, default=2000): number of steps to discretize the angles between 0 and 2pi for look up
       max_range (float, default=30.0): maximum range of the laser
   """

    def __init__(self, num_beams, fov, eps=0.0001, theta_dis=2000, max_range=30.0):
        # initialization
        self.num_beams = num_beams
        self.fov = fov
        self.eps = eps
        self.theta_dis = theta_dis
        self.max_range = max_range
        self.angle_increment = self.fov / (self.num_beams - 1)
        self.theta_index_increment = theta_dis * self.angle_increment / (2. * jnp.pi)
        self.orig_c = None
        self.orig_s = None
        self.orig_x = None
        self.orig_y = None
        self.map_height = None
        self.map_width = None
        self.map_resolution = None
        self.dt = None

        theta_arr = jnp.linspace(0.0, 2 * jnp.pi, num=theta_dis)
        self.sines = jnp.sin(theta_arr)
        self.cosines = jnp.cos(theta_arr)

    def set_map(self, map_path, map_ext):
        """ Set the bitmap of the scan simulator by path.
        This is legacy code.

        Args:
            map_path (str): path to the map yaml file
            map_ext (str): extension (image type) of the map image

        Returns:
            flag (bool): if image reading and loading is successful
        """

        # load map image
        map_img_path = os.path.splitext(map_path)[0] + map_ext
        self.map_img = jnp.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(jnp.float64)

        # grayscale -> binary
        self.map_img = self.map_img.at[self.map_img < 128.].set(0)
        self.map_img = self.map_img.at[self.map_img > 128.].set(255.)

        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]

        # load map yaml
        with open(map_path, 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                self.map_resolution = map_metadata['resolution']
                self.origin = map_metadata['origin']
            except yaml.YAMLError as ex:
                print(ex)

        # calculate map parameters
        self.orig_x = self.origin[0]
        self.orig_y = self.origin[1]
        self.orig_s = jnp.sin(self.origin[2])
        self.orig_c = jnp.cos(self.origin[2])

        # get the distance transform
        self.dt = get_dt(self.map_img, self.map_resolution)
        return True

    def scan(self, pose, std_dev=0.01):
        """Perform simulated 2D scan by pose on the given map.

        Args:
            pose (numpy.ndarray (3, )): pose of the scan frame (x, y, theta)
            std_dev (float, default=0.01): standard deviation of the generated whitenoise in the scan

        Returns:
            scan (numpy.ndarray (n, )): data array of the laserscan, n=num_beams

        Raises:
            ValueError: when scan is called before a map is set
        """
        if self.map_height is None:
            raise ValueError('Map is not set for scan simulator.')

        scan = get_scan(pose, self.theta_dis, self.fov, self.num_beams, self.theta_index_increment, self.sines,
                        self.cosines, self.eps, self.orig_x, self.orig_y, self.orig_c, self.orig_s, self.map_height,
                        self.map_width, self.map_resolution, self.dt, self.max_range)
        #
        # # check if scan and scan2 are the same
        # check = np.allclose(scan, scan2)
        #
        # if not check:
        #     print("get_scan failed")

        # add noise to the scan
        # specify which jax random distribution to choose here
        # (legacy would let you choose rng as attribute of the class, here fixed to normal)

        jax_rng = random.PRNGKey(seed=1)
        noise = random.normal(jax_rng, (self.num_beams,), dtype=jnp.float64) * std_dev
        scan += noise

        return scan

    def get_increment(self):
        return self.angle_increment
