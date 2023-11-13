""" Linearization functionality for dynamics and control equaltions.
"""
from typing import Callable, Sequence, Iterable

import jax.numpy as jnp
from jax import jacfwd


def tangent(
    f: Callable,
    n_out: int,
    argnum: int,
    ref_vector: Iterable[jnp.ndarray]
) -> jnp.ndarray:
    """ Calculates the system's dynamics tangent to the linearizing point(s)
        at the ref_vector(s).

    Args:
        f (Callable): Callable system dynamics f(x1, x2, ..., xk).
        n_out (int): Dimensionality of output vector from f(x1, x2, ..., xk).
        argnum (int): Argument indices to linearize at.
        ref_vector (Iterable[jnp.ndarray]): System of reference values to
            linearize at. The first element must be the vector in which the
            Jacobian is being calculated -- all other input ref vectors
            are treated as parameters, not variables to differentiate against.

    Returns:
        jnp.ndarray: NxN matrix, Jacobian of linearized dynamics.
    """
    # This is always calculated with jax.jacfwd since in dynamical systems
    # linearization, the jacobian is usually square or 'tall' (more rows than
    # columns).
    return jacfwd(
        f,
        argnums=argnum
    )(ref_vector).reshape((n_out, ref_vector.shape[0]))


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
        df(x, u) = Jac_x(f)(ref_x) @ (x - x_r) + Jac_u(f)(ref_u) @ (u - u_r).

        Note that the return function will not accept arguments otherwise held
        constant during linearization.  For example, if f = f(x, u) but you
        only want to linearize w.r.t. vector x, then
        df(x) = Jac_x(f)(ref_x) @ (x - x_r) is returned.  If you instead want
        df(x, u=const), you need first to define f_uconst = f(x, u=const),
        then linearize f_uconst w.r.t. reference vector x_r.

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

    # Second, get Jacobians.
    jacs = [tangent(f, n_out, argnums[i], ref_vectors[i])
            for i in range(k)]

    # Here, it will be expected that *args will be equal to
    # ref_vectors.shape[0] after ref_vectors becomes an array of arrays.
    return lambda *args: sum(
        [jacs[i] @ (args[i] - ref_vectors[i]) for i in range(k)]
    )
