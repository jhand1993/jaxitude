"""davenport's q method for quaternion parameters
"""
import jax.numpy as jnp

from jaxitude.base import antisym_dcm_vector


def get_B(
    w: jnp.ndarray,
    v_b_set: jnp.ndarray,
    v_n_set: jnp.ndarray
) -> jnp.ndarray:
    """ Generate the intermediate B matrix for davenport's q method. Heading
        vectors should be unit vectors.

    Args:
        w (jnp.ndarray): 1xN matrix, sensor weights.
        v_b_set (jnp.ndarray): 3xN matrix, N body frame headings from each
            sensor.
        v_n_set (jnp.ndarray): 3xN matrix, N inertial frame headings from
            each sensor.

    Returns:
        jnp.ndarray: B matrix
    """
    return sum(
        [w[i] * jnp.outer(v_b_set[:, i], v_n_set[:, i])
         for i in range(w.shape[0])]
    )


def get_K(
    w: jnp.ndarray,
    v_b_set: jnp.ndarray,
    v_n_set: jnp.ndarray
) -> jnp.ndarray:
    """ Generate the intermediate K matrix for davenport's q method. Heading
        vectors should be unit vectors.

    Args:
        w (jnp.ndarray): 1xN matrix, sensor weights
        v_b_set (jnp.ndarray): 3xN matrix, N body frame headings from each
            sensor.
        v_n_set (jnp.ndarray): 3xN matrix, N inertial frame headings from
            each sensor.

    Returns:
        jnp.ndarray: K matrix
    """
    B = get_B(w, v_b_set, v_n_set)
    sigma = jnp.trace(B)
    S = B + B.T
    Z = antisym_dcm_vector(B)

    return jnp.block([[sigma, Z.T], [Z, S - jnp.eye(3) * sigma]])


def get_g(
    beta: jnp.ndarray,
    w: jnp.ndarray,
    v_b_set: jnp.ndarray,
    v_n_set: jnp.ndarray
) -> float:
    """ Get g from sensor heading and weights. Heading vectors should be unit
        vectors.

    Args:
        beta (jnp.ndarray): 4x1 matrix, Euler parameters (quaternions).
            Input for optimization. Not usually used directly, but provided for
            completeness.
        w (jnp.ndarray): 1xN matrix, sensor weights
        v_b_set (jnp.ndarray): 3xN matrix, N body frame headings from each
            sensor.
        v_n_set (jnp.ndarray): 3xN matrix, N inertial frame headings from
            each sensor.

    Returns:
        float: scalar function g.
    """
    K = get_K(w, v_b_set, v_n_set)
    return jnp.matmul(
        beta,
        jnp.matmul(K, beta)
    )
