""" Some useful attitude transformations.
"""
import jax.numpy as jnp
from jax import jit


@jit
def cayley_transform(
    X: jnp.ndarray
) -> jnp.ndarray:
    """ Cayley transformation for matrix X.  X must be either skew-symmetric
        or it is orthogonal.

    Args:
        X (jnp.jndarray): NxN matrix, Skew-symmetric or orthogonal matrix.

    Returns:
        jnp.ndarray: NxN matriox, corresponding orthogonal or skew-symmetric
            matrix.
    """
    # must be a square matrix to work.
    n = X.shape[0]
    eye = jnp.identity(n)

    return jnp.matmul(eye - X, jnp.linalg.inv(eye + X))


@jit
def q_from_cayley(
    dcm: jnp.ndarray
) -> jnp.ndarray:
    """ Converts a rotation matrix (dcm attribute) to q via Cayley's
    transformation.

    Args:
        dcm (jnp.ndarray): 3x3 matrix, DCM.

    Returns:
        jnp.ndarray: 3x1 matrix, CRP q parameters.
    """
    Q = cayley_transform(dcm)
    return jnp.ndarray(
        [[-Q[1, 2]],
         [Q[0, 2]],
         [-Q[0, 1]]]
    )


# def extended_cayley_transform(X: jnp.ndarray) -> jnp.ndarray:
#     """ Cayley transformation for matrix X.  X must be either skew-symmetric
#         or it is orthogonal.

#     Args:
#         X (jnp.jndarray): Skew-symmetric or orthogonal matrix.

#     Returns:
#         jnp.ndarray: Corresponding orthogonal or or skew-symmetric matrix.
#     """"
