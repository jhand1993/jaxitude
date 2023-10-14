""" Linearization functionality for dynamics and control equaltions.
"""
from typing import Callable, Sequence

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


def linearize(
    f: Callable,
    n_out: int,
    argnums: int or jnp.ndarray[int],
    ref_vectors: jnp.ndarray or Sequence[jnp.ndarray],
) -> Callable:
    """ For system of differential equations f(x1, x2, ..., xk), this function
        linearizes f about reference vectors corresponding to argnums indices.
        For example, if f = f(x, u), where x is a state vector and u is a
        control vector, then linearize(f, [0, 1], [ref_x, ref_u]) will return
        f(x, u) approx Jacobian_x(f)(ref_x) @ x + Jacobian_u(f)(ref_u) @ u.

        Note that the return function will not accept arguments otherwise held
        constant during linearization.

    Args:
        f (Callable): General system of differential equations with multiple
            argument vectors.
        n_out (int): Dimensionality of output vector from f(x1, x2, ..., xk).
        argnums (int or jnp.ndarray[int]): Argument indices to linearize at.
            If an argument is skipped, then no Jacobian w.r.t. this argument is
            calculated.  Must be the same length as ref_vectors.
        ref_vectors (jnp.ndarrayorSequence[jnp.ndarray]): argument values to
            linearize at. Must be the same length as ref_vectors.

    Returns:
        Callable: Linearized approximation of function f.
    """
    # First, convert input argnums and ref_vectors to Lists if not sequences.
    argnums = jnp.where(
        isinstance(argnums, int),
        jnp.asarray([argnums]),
        argnums
    )
    ref_vectors = jnp.where(
        isinstance(ref_vectors, jnp.ndarray),
        jnp.asarray([ref_vectors]),
        ref_vectors
    )
    k = len(argnums)

    # Second, get Jacobians.  This is always calcualted with jax.jacfwd since
    # in dynamical systems linearization, the jacobian is usually square or
    # 'tall' (more rows than columns).
    jacs = [
        jacfwd(
            f,
            argnums=argnums[i]
        )(ref_vectors[i]).reshape((n_out, ref_vectors[i].shape[0]))
        for i in range(k)
    ]
    print(jacs)

    # Here, it will be expected that *args will be of length equal to
    # ref_vectors.
    return lambda *args: sum(
        [jacs[i] @ args[i] for i in range(k)]
    )
