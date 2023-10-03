"""davenport's q method for quaternion parameters
"""
import jax.numpy as jnp
from jax import jit
from attitude.primitives import MiscUtil


def get_B(
    w: jnp.ndarray,
    v_b_set: jnp.ndarray,
    v_n_set: jnp.ndarray
) -> jnp.ndarray:
    """ Generate the intermediate B matrix for davenport's q method. Heading vectors should
        be unit vectors.

    Args:
        w (jnp.ndarray): 1xN matrix of sensor weights
        v_b_set (jnp.ndarray): Nx3 matrix of N body frame headings from each sensor.
        v_n_set (jnp.ndarray): Nx3 matrix of N inertial frame headings from each sensor.

    Returns:
        jnp.ndarray: B matrix
    """
    return sum([w[i] * jnp.outer(v_b_set[i], v_n_set[i]) for i in range(w.shape[0])])


def get_K(
    w: jnp.ndarray,
    v_b_set: jnp.ndarray,
    v_n_set: jnp.ndarray
) -> jnp.ndarray:
    """ Generate the intermediate K matrix for davenport's q method. Heading vectors should
        be unit vectors.

    Args:
        w (jnp.ndarray): 1xN matrix of sensor weights
        v_b_set (jnp.ndarray): Nx3 matrix of N body frame headings from each sensor.
        v_n_set (jnp.ndarray): Nx3 matrix of N inertial frame headings from each sensor.

    Returns:
        jnp.ndarray: K matrix
    """
    B = get_B(w, v_b_set, v_n_set)
    sigma = jnp.trace(B)
    S = B + B.T
    Z = jnp.expand_dims(MiscUtil.antisym_dcm_vector(B), axis=-1)

    return jnp.block([[sigma, Z.T], [Z, S - jnp.eye(3) * sigma]])


def get_g(
    beta: jnp.ndarray,
    w: jnp.ndarray,
    v_b_set: jnp.ndarray,
    v_n_set: jnp.ndarray
) -> float:
    """ Get g from sensor heading and weights. Heading vectors should be unit vectors.

    Args:
        beta (jnp.ndarray): 1x4 matrix representation of Euler parameters.  Input for optimization.
            Not usually used directly, but provided for completeness.
        w (jnp.ndarray): 1xN matrix of sensor weights
        v_b_set (jnp.ndarray): Nx3 matrix of N body frame headings from each sensor.
        v_n_set (jnp.ndarray): Nx3 matrix of N inertial frame headings from each sensor.

    Returns:
        float: scalar function g.
    """
    K = get_K(w, v_b_set, v_n_set)
    return jnp.matmul(
        jnp.expand_dims(beta, axis=-1), jnp.matmul(K, jnp.expand_dims(beta, axis=-1).T)
    )
    