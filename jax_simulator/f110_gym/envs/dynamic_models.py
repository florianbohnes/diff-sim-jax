import unittest
import time
import jax.numpy as jnp
from jax import jit
from jax import lax
from jax import config
from jax import vmap
from functools import partial
config.update("jax_enable_x64", True)

@jit
def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    # This works with parameters being passed as arrays
    """Acceleration constraints, adjusts the acceleration based on constraints.

        Args:
            vel (float): current velocity of the vehicle
            accl (float): unconstraint desired acceleration
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to
                create wheel spin)
            a_max (float): maximum allowed acceleration
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            accl (float): adjusted acceleration
        """

    def non_zero_vel(vel):
        pos_limit = lax.cond(vel > v_switch,
                             lambda _: (a_max * v_switch / vel),
                             lambda _: a_max,
                             vel)
        return pos_limit

    def zero_vel(vel):
        pos_limit = a_max
        return pos_limit

    pos_limit = lax.cond(vel != 0., non_zero_vel, zero_vel, vel)

    def accl_out_of_bounds(accl):
        accl = lax.cond(accl<=-a_max,
                        lambda _: -a_max,
                        lambda _: lax.cond(accl>=pos_limit,
                                           lambda _: pos_limit,
                                           lambda _: accl,
                                           accl),
                        accl)
        return accl

    accl = lax.cond(vel <= v_min,
                  lambda _: lax.cond(accl<=0,
                                     lambda _: 0.,
                                     lambda _: lax.cond(vel >= v_max,
                                                        lambda _: lax.cond(accl >= 0,
                                                                           lambda _: 0.,
                                                                           accl_out_of_bounds,
                                                                           accl),
                                                        accl_out_of_bounds,
                                                        accl
                                                        ),
                                     accl),
                  lambda _: lax.cond(vel >= v_max,
                                       lambda _: lax.cond(accl >= 0,
                                                          lambda _: 0.,
                                                          accl_out_of_bounds,
                                                          accl),
                                       accl_out_of_bounds,
                                       accl
                    ), vel)

    return accl


@jit
def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    """Steering constraints, adjusts the steering velocity based on constraints.

    Args:
        steering_angle (float): current steering_angle of the vehicle
        steering_velocity (float): unconstraint desired steering_velocity
        s_min (float): minimum steering angle
        s_max (float): maximum steering angle
        sv_min (float): minimum steering velocity
        sv_max (float): maximum steering velocity

    Returns:
        steering_velocity (float): adjusted steering velocity
    """

    # constraint steering velocity
    def non_zero_steering_velocity(steering_velocity):
        steering_velocity = lax.cond(
            steering_velocity <= sv_min,
            lambda _: sv_min,
            lambda _: lax.cond(steering_velocity >= sv_max,
                               lambda _: sv_max,
                               lambda _: steering_velocity,
                               steering_velocity),
            steering_velocity
        )
        return steering_velocity


    steering_velocity = lax.cond(
        steering_angle <= s_min,
        lambda _: lax.cond(
            steering_velocity <= 0,
            lambda _: 0.,
            non_zero_steering_velocity,
            steering_velocity
        ),
        lambda _: lax.cond(
            steering_angle >= s_max,
            lambda _: lax.cond(
                steering_velocity >= 0,
                lambda _: 0.,
                non_zero_steering_velocity,
                steering_velocity
            ),
            non_zero_steering_velocity,
            steering_velocity
        ),
        steering_velocity
    )
    return steering_velocity

@jit
def vehicle_dynamics_ks(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max,
                        v_min, v_max):
    """Single Track Kinematic Vehicle Dynamics.

    Args:
        x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5)
            x1: x position in global coordinates
            x2: y position in global coordinates
            x3: steering angle of front wheels
            x4: velocity in x direction
            x5: yaw angle
        u (numpy.ndarray (2, )): control input vector (u1, u2)
            u1: steering angle velocity of front wheels
            u2: longitudinal acceleration

    Returns:
        f (numpy.ndarray): right hand side of differential equations
    """
    # wheelbase
    lwb = lf + lr

    # constraints
    u = jnp.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max),
                  accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # system dynamics
    f = jnp.array([x[3] * jnp.cos(x[4]),
                  x[3] * jnp.sin(x[4]),
                  u[0],
                  u[1],
                  x[3] / lwb * jnp.tan(x[2])])

    return f


def vehicle_dynamics_st_true_fun(vars):
    """
    Conditional function called from within vehicle_dynamics_st
    """
    g, x, u, mu, m, I, lr, lf, C_Sf, C_Sr, h, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max = vars
    # wheelbase
    lwb = lf + lr

    x_ks = x[0:5]

    # vehicle dynamics
    f_ks = vehicle_dynamics_ks(x_ks, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch,
                               a_max, v_min, v_max)

    f = jnp.hstack((f_ks, jnp.array([u[1] / lwb * jnp.tan(x[2]) + x[3] / (lwb * jnp.cos(x[2]) ** 2) * u[0],
                                   0])))

    return f


def vehicle_dynamics_st_false_fun(vars):
    """
       Conditional function called from within vehicle_dynamics_st
   """
    g, x, u, mu, m, I, lr, lf, C_Sf, C_Sr, h, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max = vars
    # system dynamics
    f = jnp.array([
        x[3] * jnp.cos(x[6] + x[4]),
        x[3] * jnp.sin(x[6] + x[4]),
        u[0],
        u[1],
        x[5],
        -mu * m / (x[3] * I * (lr + lf)) * (
                lf ** 2 * C_Sf * (g * lr - u[1] * h) + lr ** 2 * C_Sr * (g * lf + u[1] * h)) * x[5] \
        + mu * m / (I * (lr + lf)) * (lr * C_Sr * (g * lf + u[1] * h) - lf * C_Sf * (g * lr - u[1] * h)) * x[6] \
        + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - u[1] * h) * x[2],
        (mu / (x[3] ** 2 * (lr + lf) + 1e-5) * (
                C_Sr * (g * lf + u[1] * h) * lr - C_Sf * (g * lr - u[1] * h) * lf) - 1) *
        x[5] \
        - mu / (x[3] * (lr + lf)) * (C_Sr * (g * lf + u[1] * h) + C_Sf * (g * lr - u[1] * h)) * x[6] \
        + mu / (x[3] * (lr + lf)) * (C_Sf * (g * lr - u[1] * h)) * x[2]])

    return f


@jit
def vehicle_dynamics_st(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max,
                        v_min, v_max):
    """Single Track Dynamic Vehicle Dynamics.

    Args:
        x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
            x1: x position in global coordinates
            x2: y position in global coordinates
            x3: steering angle of front wheels
            x4: velocity in x direction
            x5: yaw angle
            x6: yaw rate
            x7: slip angle at vehicle center
        u (numpy.ndarray (2, )): control input vector (u1, u2)
            u1: steering angle velocity of front wheels
            u2: longitudinal acceleration

    Returns:
        f (jax.numpy.ndarray): right hand side of differential equations
    """
    # gravity
    g = 9.81

    # constraints
    u = jnp.array(
        [
            steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max),
            accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)
        ]
    )

    vars = g, x, u, mu, m, I, lr, lf, C_Sf, C_Sr, h, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max

    # switch to kinematic model for small velocities
    f = lax.cond(x[3] < 1, vehicle_dynamics_st_true_fun, vehicle_dynamics_st_false_fun, vars)

    return f

@jit
@partial(vmap, in_axes=(0, 0, 0, 0, None, None, None, None))
def pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v):
    """Basic controller for speed/steer -> accl./steer vel.

    Args:
        speed (float): desired input speed
        steer (float): desired input steering angle

    Returns:
        accl (float): desired input acceleration
        sv (float): desired input steering velocity
    """

    # steering
    min_v = min_v - 1e-5

    steer_diff = steer - current_steer
    sv = lax.cond(jnp.fabs(steer_diff) > 1e-4,
                  lambda _: (steer_diff / jnp.fabs(steer_diff)) * max_sv,
                  lambda _: 0.0,
                  steer_diff)


    # accl
    vel_diff = speed - current_speed
    # currently forward
    accl = lax.cond(current_speed > 0.0,
                    lambda _: lax.cond(vel_diff > 0.0,
                                       lambda _: (10.0 * max_a/max_v) * vel_diff,
                                       lambda _: (10.0 * max_a/(-min_v)) * vel_diff,
                                       vel_diff),
                    lambda _: lax.cond(vel_diff > 0.0,
                                       lambda _: (2.0 * max_a / max_v) * vel_diff,
                                       lambda _: (2.0 * max_a / (-min_v)) * vel_diff,
                                       vel_diff),
                    vel_diff)

    return accl, sv

@jit
def func_KS(x, t, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    f = vehicle_dynamics_ks(x, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min,
                            v_max)
    return f

@jit
def func_ST(x, t, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    f = vehicle_dynamics_st(x, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min,
                            v_max)
    return f

class DynamicsTest(unittest.TestCase):
    def setUp(self):
        # test params
        self.mu = 1.0489
        self.C_Sf = 21.92 / 1.0489
        self.C_Sr = 21.92 / 1.0489
        self.lf = 0.3048 * 3.793293
        self.lr = 0.3048 * 4.667707
        self.h = 0.3048 * 2.01355
        self.m = 4.4482216152605 / 0.3048 * 74.91452
        self.I = 4.4482216152605 * 0.3048 * 1321.416

        # steering constraints
        self.s_min = -1.066  # minimum steering angle [rad]
        self.s_max = 1.066  # maximum steering angle [rad]
        self.sv_min = -0.4  # minimum steering velocity [rad/s]
        self.sv_max = 0.4  # maximum steering velocity [rad/s]

        # longitudinal constraints
        self.v_min = -13.6  # minimum velocity [m/s]
        self.v_max = 50.8  # minimum velocity [m/s]
        self.v_switch = 7.319  # switching velocity [m/s]
        self.a_max = 11.5  # maximum absolute acceleration [m/s^2]

    def test_derivatives(self):
        # ground truth derivatives
        f_ks_gt = jnp.array(
            [16.3475935934250209, 0.4819314886013121, 0.1500000000000000, 5.1464424102339752, 0.2401426578627629]
        )
        f_st_gt = jnp.array(
            [15.7213512030862397, 0.0925527979719355, 0.1500000000000000, 5.3536773276413925, 0.0529001056654038,
                   0.6435589397748606, 0.0313297971641291]
        )

        # system dynamics
        g = 9.81
        x_ks = jnp.array(
            [3.9579422297936526, 0.0391650102771405, 0.0378491427211811, 16.3546957860883566, 0.0294717351052816])
        x_st = jnp.array(
            [2.0233348142065677, 0.0041907137716636, 0.0197545248559617, 15.7216236334290116, 0.0025857914776859,
             0.0529001056654038, 0.0033012170610298])
        v_delta = 0.15
        acc = 0.63 * g
        u = jnp.array([v_delta, acc])

        f_ks = vehicle_dynamics_ks(x_ks, u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I,
                                   self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max,
                                   self.v_min, self.v_max)
        f_st = vehicle_dynamics_st(x_st, u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I,
                                   self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max,
                                   self.v_min, self.v_max)

        start = time.time()
        for i in range(10000):
            f_st = vehicle_dynamics_st(x_st, u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I,
                                       self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max,
                                       self.v_min, self.v_max)
        duration = time.time() - start
        avg_fps = 10000 / duration

        self.assertAlmostEqual(jnp.max(jnp.abs(f_ks_gt - f_ks)), 0.)
        self.assertAlmostEqual(jnp.max(jnp.abs(f_st_gt - f_st)), 0.)

        self.assertGreater(avg_fps, 5000)

    def test_zeroinit_roll(self):
        #from scipy.integrate import odeint
        from jax.experimental.ode import odeint

        # testing for zero initial state, zero input singularities
        g = 9.81
        t_start = 0.
        t_final = 1.
        delta0 = 0.
        vel0 = 0.
        Psi0 = 0.
        dotPsi0 = 0.
        beta0 = 0.
        sy0 = 0.
        initial_state = [0, sy0, delta0, vel0, Psi0, dotPsi0, beta0]

        x0_KS = jnp.array(initial_state[0:5])
        x0_ST = jnp.array(initial_state)

        # time vector
        t = jnp.arange(t_start, t_final, 1e-4)

        # set input: rolling car (velocity should stay constant)
        u = jnp.array([0., 0.])

        # simulate single-track model
        x_roll_st = odeint(func_ST, x0_ST, t,
            u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max,
            self.sv_min,
            self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max)
        # simulate kinematic single-track model
        x_roll_ks = odeint(func_KS, x0_KS, t,
            u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max,
            self.sv_min,
            self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max)

        self.assertTrue(all(x_roll_st[-1] == x0_ST))
        self.assertTrue(all(x_roll_ks[-1] == x0_KS))

    def test_zeroinit_dec(self):
        # from scipy.integrate import odeint
        from jax.experimental.ode import odeint

        # testing for zero initial state, decelerating input singularities
        g = 9.81
        t_start = 0.
        t_final = 1.
        delta0 = 0.
        vel0 = 0.
        Psi0 = 0.
        dotPsi0 = 0.
        beta0 = 0.
        sy0 = 0.
        initial_state = [0, sy0, delta0, vel0, Psi0, dotPsi0, beta0]

        x0_KS = jnp.array(initial_state[0:5])
        x0_ST = jnp.array(initial_state)

        # time vector
        t = jnp.arange(t_start, t_final, 1e-4)

        # set decel input
        u = jnp.array([0., -0.7 * g])

        # simulate single-track model
        x_dec_st = odeint(func_ST, x0_ST, t,
            u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max,
            self.sv_min,
            self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max)
        # simulate kinematic single-track model
        x_dec_ks = odeint(func_KS, x0_KS, t,
            u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max,
            self.sv_min,
            self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max)

        # ground truth for single-track model
        x_dec_st_gt = jnp.array(
            [-3.4335000000000013, 0.0000000000000000, 0.0000000000000000, -6.8670000000000018,
                       0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        )
        # ground truth for kinematic single-track model
        x_dec_ks_gt = jnp.array(
            [-3.4335000000000013, 0.0000000000000000, 0.0000000000000000, -6.8670000000000018,
                       0.0000000000000000]
        )

        self.assertTrue(all(abs(x_dec_st[-1] - x_dec_st_gt) < 1e-2))
        self.assertTrue(all(abs(x_dec_ks[-1] - x_dec_ks_gt) < 1e-2))

    def test_zeroinit_acc(self):
        from scipy.integrate import odeint
        #from jax.experimental.ode import odeint

        # testing for zero initial state, accelerating with left steer input singularities
        # wheel spin and velocity should increase more wheel spin at rear
        g = 9.81
        t_start = 0.
        t_final = 1.
        delta0 = 0.
        vel0 = 0.
        Psi0 = 0.
        dotPsi0 = 0.
        beta0 = 0.
        sy0 = 0.
        initial_state = [0, sy0, delta0, vel0, Psi0, dotPsi0, beta0]

        x0_KS = jnp.array(initial_state[0:5])
        x0_ST = jnp.array(initial_state)

        # time vector
        t = jnp.arange(t_start, t_final, 1e-4)

        # set decel input
        u = jnp.array([0.15, 0.63 * g])
        # simulate single-track model
        x_acc_st = odeint(func_ST, x0_ST, t,args=(
            u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max,
            self.sv_min,
            self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))
        # simulate kinematic single-track model
        x_acc_ks = odeint(func_KS, x0_KS, t, args=(
            u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max,
            self.sv_min,
            self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))

        # ground truth for single-track model
        x_acc_st_gt = jnp.asarray(
            [3.0731976046859715, 0.2869835398304389, 0.1500000000000000, 6.1802999999999999,
                       0.1097747074946325, 0.3248268063223301, 0.0697547542798040]
        )
        # ground truth for kinematic single-track model
        x_acc_ks_gt = jnp.array(
            [3.0845676868494927, 0.1484249221523042, 0.1500000000000000, 6.1803000000000017,
                       0.1203664469224163]
        )

        self.assertTrue(all(abs(x_acc_st[-1] - x_acc_st_gt) < 1e-2))
        self.assertTrue(all(abs(x_acc_ks[-1] - x_acc_ks_gt) < 1e-2))

    def test_zeroinit_rollleft(self):
        #from scipy.integrate import odeint
        from jax.experimental.ode import odeint

        # testing for zero initial state, rolling and steering left input singularities
        g = 9.81
        t_start = 0.
        t_final = 1.
        delta0 = 0.
        vel0 = 0.
        Psi0 = 0.
        dotPsi0 = 0.
        beta0 = 0.
        sy0 = 0.
        initial_state = [0, sy0, delta0, vel0, Psi0, dotPsi0, beta0]

        x0_KS = jnp.array(initial_state[0:5])
        x0_ST = jnp.array(initial_state)

        # time vector
        t = jnp.arange(t_start, t_final, 1e-4)

        # set decel input
        u = jnp.array([0.15, 0.])

        # simulate single-track model
        x_left_st = odeint(func_ST, x0_ST, t,
            u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max,
            self.sv_min,
            self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max)
        # simulate kinematic single-track model
        x_left_ks = odeint(func_KS, x0_KS, t,
            u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max,
            self.sv_min,
            self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max)

        # ground truth for single-track model
        x_left_st_gt = jnp.array(
            [0.0000000000000000, 0.0000000000000000, 0.1500000000000000, 0.0000000000000000,
                        0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        )
        # ground truth for kinematic single-track model
        x_left_ks_gt = jnp.array(
            [0.0000000000000000, 0.0000000000000000, 0.1500000000000000, 0.0000000000000000,
                        0.0000000000000000]
        )

        self.assertTrue(all(abs(x_left_st[-1] - x_left_st_gt) < 1e-2))
        self.assertTrue(all(abs(x_left_ks[-1] - x_left_ks_gt) < 1e-2))


if __name__ == '__main__':
    unittest.main()