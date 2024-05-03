import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from jax import config
config.update("jax_enable_x64", True)


def legacy_accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
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

    # positive accl limit
    if vel > v_switch:
        pos_limit = a_max * v_switch / vel
    else:
        pos_limit = a_max

    # accl limit reached?
    if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
        accl = 0.
    elif accl <= -a_max:
        accl = -a_max
    elif accl >= pos_limit:
        accl = pos_limit

    return accl

def jax_accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
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

def legacy_steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
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
    if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
        steering_velocity = 0.
    elif steering_velocity <= sv_min:
        steering_velocity = sv_min
    elif steering_velocity >= sv_max:
        steering_velocity = sv_max

    return steering_velocity

def jax_steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
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

def test_legacy_accl_constraints():
    vel = 3.0
    accl = 6.0
    v_switch = 7.319
    a_max = 11.5
    v_min = -13.6
    v_max = 50.8

    #assert legacy_accl_constraints(vel, accl, v_switch, a_max, v_min, v_max) == jax_accl_constraints(vel, accl, v_switch, a_max, v_min, v_max)
    print(f"Legacy Accl: {legacy_accl_constraints(vel, accl, v_switch, a_max, v_min, v_max)}")
    print(f"Jax Accl: {jax_accl_constraints(vel, accl, v_switch, a_max, v_min, v_max)}")

def test_legacy_steering_constraint():
    steering_angle = -1.066
    steering_velocity = 0.15
    s_min = -1.066
    s_max = 1.066
    sv_min = -0.4
    sv_max = 0.4

    assert legacy_steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max) == jax_steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max)
    print(f"Legacy Steering: {legacy_steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max)}")
    print(f"Jax Steering: {jax_steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max)}")

test_legacy_accl_constraints()
test_legacy_steering_constraint()