""" Dynamical system integrators for state vector/covariance propogation.

"""
from typing import Callable

import jax.numpy as jnp


def autonomous_euler(
    f: Callable,
    x: jnp.ndarray,
    dt: float,
    *args
) -> jnp.ndarray:
    """ Euler's method for integrating autonomous dynamical system f(x, *args).
        *args, though optional, is important: any vector argument for equation
        f(x, a1, a2, ..., ak) other than the state vector x will be fed into
        f(x, a1, a2, ..., ak) via *args.

        For example, if the dynamics relate observed rates w to the evolution of
        MRP parameters s, then f(x->s, a1>w) are is the resulting function f.
        That means if you call `autonomous_euler(f, x, *args)`, then args=(w,)

    Args:
        f (Callable): Callable system of equations with matrix x and *args as
            arguments, in that order.
        x (jnp.ndarray): Nx1 state column vector.
        dt (float): Integration time interval.

    Returns:
        jnp.ndarray: Integrated Nx1 state vector prediction.
    """
    return x + f(x, *args) * dt


def euler(
    f: Callable,
    t: float,
    x: jnp.ndarray,
    dt: float,
    *args
) -> jnp.ndarray:
    """ Euler's method for integrating dynamical system f(t, x).
        *args, though optional, is important: any vector argument for equation
        f(t, x, a1, a2, ..., ak) other than the state vector x will be fed into
        f(t, x, a1, a2, ..., ak) via *args.

        For example, if the dynamics relate observed rates w to the evolution of
        MRP parameters s, then f(x->s, a1>w) are is the resulting function f.
        That means if you call `autonomous_euler(f, x, *args)`, then args=(w,).

    Args:
        f (Callable): Callable system of equations with t, matrix x, and *args
            as arguments, in that order.
        t (float): Time step t value.
        x (jnp.ndarray): Nx1 state column vector.
        dt (float): Integration time interval.

    Returns:
        jnp.ndarray: Integrated Nx1 state vector prediction.
    """
    return x + f(t, x, *args) * dt


def autonomous_rk4(
    f: Callable,
    x: jnp.ndarray,
    dt: float,
    *args
) -> jnp.ndarray:
    """ RK4 method for integrating autonomous dynamical system f(x, t).
        *args, though optional, is important: any vector argument for equation
        f(x, a1, a2, ..., ak) other than the state vector x will be fed into
        f(x, a1, a2, ..., ak) via *args.

        For example, if the dynamics relate observed rates w to the evolution of
        MRP parameters s, then f(x->s, a1>w) are is the resulting function f.
        That means if you call `autonomous_euler(f, x, *args)`, then args=(w,)

        A caveat of using *args is that these arguments are not properly
        propogated through the four intermediate rk4 calculations.

    Args:
        f (Callable): Callable system of equations matrix x and *args as
            arguments, in that order.
        x (jnp.ndarray): Nx1 state column vector.
        t (float): Time step t value.
        dt (float): Integration time interval.

    Returns:
        jnp.ndarray: Integrated Nx1 state vector prediction.
    """
    k1 = f(x, *args)
    k2 = f(x + dt * k1 * 0.5, *args)
    k3 = f(x + dt * k2 * 0.5, *args)
    k4 = f(x + dt * k3, *args)

    return x + dt * (k1 + 2. * k2 + 2. * k3 + k4) / 6.


def rk4(
    f: Callable,
    t: float,
    x: jnp.ndarray,
    dt: float,
    *args
) -> jnp.ndarray:
    """ RK4 method for integrating dynamical system f(t, x).
        *args, though optional, is important: any vector argument for equation
        f(t, x, a1, a2, ..., ak) other than the state vector x will be fed into
        f(t, x, a1, a2, ..., ak) via *args.

        For example, if the dynamics relate observed rates w to the evolution of
        MRP parameters s, then f(x->s, a1>w) are is the resulting function f.
        That means if you call `autonomous_euler(f, x, *args)`, then args=(w,).

        A caveat of using *args is that these arguments are not properly
        propogated through the four intermediate rk4 calculations.

    Args:
        f (Callable): Callable system of equations with t, matrix x, and *args
            as arguments, in that order.
        t (float): Time step t value.
        x (jnp.ndarray): Nx1 state column vector.
        dt (float): Integration time interval.

    Returns:
        jnp.ndarray: Integrated Nx1 state vector prediction.
    """
    k1 = f(t, x, *args)
    k2 = f(t + dt * 0.5, x + dt * k1 * 0.5, *args)
    k3 = f(t + dt * 0.5, x + dt * k2 * 0.5, *args)
    k4 = f(t + dt, x + dt * k3)

    return x + dt * (k1 + 2. * k2 + 2. * k3 + k4) / 6.


def lie_euler(
    f: Callable,
    t: float,
    x: jnp.ndarray,
    dt: float,
    expm: Callable,
    compfunc: Callable,
    *args
) -> jnp.ndarray:
    """ Lie-Euler method for integrating dynamical system f(t, x) given the
        exponential map 'expm' and composition function 'compfunc'.

        *args, though optional, is important: any vector argument for equation
        f(t, x, a1, a2, ..., ak) other than the state vector x will be fed into
        f(t, x, a1, a2, ..., ak) via *args.

        For example, if the dynamics relate observed rates w to the evolution of
        MRP parameters s, then f(x->s, a1>w) are is the resulting function f.
        That means if you call `lie_euler(f, x, *args)`, then args=(w,).

    Args:
        f (Callable): Callable system of equations with t, matrix x, and *args
            as arguments, in that order.
        t (float): Time step t value.
        x (jnp.ndarray): Nx1 state column vector.
        dt (float): Integration time interval.
        exmp (Callable): Exponential map for x.
        compfunc (Callable): Composition operator for x-like objects. Should
            be of the form x_f = x_i * exp(dx), where * is the composition
            operation.

    Returns:
        jnp.ndarray: Integrated Nx1 state vector prediction.
    """
    return compfunc(
        x,
        expm(f(t, x, *args) * dt)
    )
