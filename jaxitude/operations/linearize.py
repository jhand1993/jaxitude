""" Linearization functionality for dynamics and control equaltions.
"""
from typing import Callable

import jax.numpy as jnp
from jax import jacfwd


def linearize_dynamics_control(
    f: Callable,
    x_r: jnp.ndarray,
    u_r: jnp.ndarray
) -> jnp.ndarray:
    """ Linearizes system of equations f at reference state vector x_r and
        reference control vector u_r.  f must be callable, take state vector x
        and control vector u as arguments, returning a vector the same dimension
        of x when called.

    Args:
        f (Callable): Callable function for a dynamical system's set of
            equations
        x_r (jnp.ndarray): Nx1 array of reference state column vector, where
            N is the number of state variables.
        u_r (jnp.ndarray): Mx1 array of reference control column vector, where
            M is the number of control variables.

    Returns:
        Callable: Linearized version of callable f at references x_r and u_r
            which takes the same callable f arguments jnp.ndarrays x and u.
    """
    # Get state vector and control vector dimensions.
    n = x_r.shape[1]
    m = u_r.shape[1]

    # Calculate jacobian evaluated at references.
    jac_fx = jacfwd(f, argnums=0)(x_r, u_r).reshape((n, n))
    jac_fu = jacfwd(f, argnums=1)(x_r, u_r).reshape((n, m))

    # Return the lienarized dynamical system evaluated at references.
    return lambda x, u: jac_fx @ (x - x_r) + jac_fu @ (u - u_r)


def linearize_dynamics(
    f: Callable,
    x_r: jnp.ndarray,
) -> jnp.ndarray:
    """ Linearizes dynamical system of equations f at reference vector x_r.
        f must be callable and take vector x as an argument; f(x) should have
        the same dimension as x.

    Args:
        f (Callable): Callable function for a dynamical system's set of
            equations
        x_r (jnp.ndarray): Nx1 array of reference state column vector, where
            N is the number of state variables.

    Returns:
        Callable: Linearized version of callable f at references x_r and u_r
            which takes the same callable f arguments jnp.ndarrays x and u.
    """
    # Get state dimension.
    n = x_r.shape[1]

    # Calculate jacobian evaluated at reference.
    jac_fx = jacfwd(f, argnums=0)(x_r).reshape((n, n))

    # Return the linearized dynamical system evaluated at reference.
    return lambda x: jac_fx @ (x - x_r)
