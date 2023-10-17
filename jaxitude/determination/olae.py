""" OLAE sensor fusion algorithm.
"""
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax import jit

from jaxitude.base import cpo


def olae_get_CRPq(
    w: jnp.ndarray,
    v_b_set: jnp.ndarray,
    v_n_set: jnp.ndarray
) -> jnp.ndarray:
    """ OLAE sensor fusion method which converts fusion problem into a
        least-squares problem using Cayley transforms.

    Args:
        w (jnp.ndarray): 1xN array, weights for N measurements.
        v_b_set (jnp.ndarray): 3xN array, body frame headings.
        v_n_set (jnp.ndarray): 3xN array, inertial frame headings.

    Returns:
        jnp.ndarray: 3x1 array of CRP parameters q
    """
    n = len(w)
    # First, convert 3xN or 1x3N arrays into 3Nx1 arrays.
    v_b_vec = v_b_set.reshape((3 * n, 1), order='F')
    v_n_vec = v_n_set.reshape((3 * n, 1), order='F')

    # Calculate summation and difference column vectors
    s = v_b_vec + v_n_vec
    d = v_b_vec - v_n_vec

    # Calculate diagonal weight block matrix.
    w_block = weight_blockmatrix(w)

    # Calculate cross product operator stack matrix for vectors.
    s_mat = stacked_s_cpo(s, n)

    # Calculate inverse matrix.
    mat1 = jnp.linalg.inv(s_mat.T @ w_block @ s_mat)

    return mat1 @ s_mat.T @ w_block @ d


@jit
def weight_blockmatrix(w: jnp.ndarray) -> jnp.ndarray:
    """ Creates a block matrix of w_i * I_(3x3) blocks along diagonal.

    Args:
        w (jnp.ndarray): 1xN matrix, sensor weights.

    Returns:
        jnp.ndarray: 3Nx3N matrix, weights block matrix.
    """
    blocks = tuple(w_i * jnp.eye(3) for w_i in w)
    return block_diag(*blocks)


def stacked_s_cpo(s: jnp.ndarray, n: float) -> jnp.ndarray:
    """ Creates a stacked array of cross product operators for every three
        elements of 3Nx1 array s.

    Args:
        s (jnp.ndarray): 3Nx1 matrix, stacked s column vectors.
        n (float): Number of measurements.

    Returns:
        jnp.ndarray: 3Nx3 matrix, cross product operators stacked.
    """
    # Need to figure out a better way to do this without list comprehension
    return jnp.vstack(
        jnp.array([cpo(s[3 * i:3 * (i + 1)]) for i in range(n)])
    )
