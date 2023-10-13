""" Dynamical system integrators for state vector/covariance propogation.

"""
from typing import Callable

import jax.numpy as jnp


def autonomous_euler(
    f: Callable,
    x: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """ Euler's method for integrating autonomous dynamical system f(x).

    Args:
        f (Callable): Callable system of equations that takes matrix x_old as
            as its argument.
        x (jnp.ndarray): Nx1 state column vector.
        dt (float): time interval of integration.

    Returns:
        jnp.ndarray: Integrated Nx1 state vector prediction.
    """
    return x + f(x) * dt


def euler(
    f: Callable,
    x: jnp.ndarray,
    t: float,
    dt: float,
) -> jnp.ndarray:
    """ Euler's method for integrating dynamical system f(x, t).

    Args:
        f (Callable): Callable system of equations that takes matrix x_old as
            as its argument.
        x (jnp.ndarray): Nx1 state column vector.
        t (float): Time step t value.
        dt (float): time interval of integration.

    Returns:
        jnp.ndarray: Integrated Nx1 state vector prediction.
    """
    return x + f(x, t) * dt


def rk4(
    f: Callable,
    x: jnp.ndarray,
    t: float,
    dt: float,
) -> jnp.ndarray:
    """ RK4 method for integrating dynamical system f(x, t).

    Args:
        f (Callable): Callable system of equations that takes matrix x_old as
            as its argument.
        x (jnp.ndarray): Nx1 state column vector.
        t (float): Time step t value.
        dt (float): time interval of integration.

    Returns:
        jnp.ndarray: Integrated Nx1 state vector prediction.
    """
    k1 = f(x, t)
    k2 = f(x + dt * k1 * 0.5, t + dt * 0.5)
    k3 = f(x + dt * k2 * 0.5, t + dt * 0.5)
    k4 = f(x + dt * k3, t + dt)

    return x + dt * (k1 + 2. * k2 + 2. * k3 + k4) / 6.


def autonomous_rk4(
    f: Callable,
    x: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """ RK4 method for integrating autonomous dynamical system f(x, t).

    Args:
        f (Callable): Callable system of equations that takes matrix x_old as
            as its argument.
        x (jnp.ndarray): Nx1 state column vector.
        t (float): Time step t value.
        dt (float): time interval of integration.

    Returns:
        jnp.ndarray: Integrated Nx1 state vector prediction.
    """
    k1 = f(x)
    k2 = f(x + dt * k1 * 0.5)
    k3 = f(x + dt * k2 * 0.5)
    k4 = f(x + dt * k3)

    return x + dt * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
