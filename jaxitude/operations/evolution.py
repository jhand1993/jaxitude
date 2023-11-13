import jax.numpy as jnp

from jaxitude.base import cpo


def euler_eqs(
    w: jnp.ndarray,
    I: jnp.ndarray
) -> jnp.ndarray:
    """ Euler equations of motion for a rigid body w.r.t. its center of mass.
        Torques are ignored.

    Args:
        w (jnp.ndarray): rate vector as 3x1 matrix
        I (jnp.ndarray): 3x3 matrix inertia tensor

    Returns:
        jnp.ndarray: 3x1 matrix, product of inertia matrix I with rate w.
    """
    I_w_dot = -cpo(w) @ I @ w
    return I_w_dot


def evolve_P_ricatti(
    P: jnp.ndarray,
    F: jnp.ndarray,
    G: jnp.ndarray,
    Q_a: jnp.ndarray,
    Q_i: jnp.ndarray
) -> jnp.ndarray:
    """ Ricatti differential equation to propogate process noise P EKF
        workflows.

    Args:
        P (jnp.ndarray): 6x6 matrix, process noise matrix.
        F (jnp.ndarray): 6x6 matrix, linearized dynamics model matrix.
        G (jnp.ndarray): 6x6 matrix, linearized noise model matrix.
        Q_a (jnp.ndarray): 6x6 matrix, additive process noise covariance matrix.
        Q_i (jnp.ndarray): 6x6 matrix, internal process noise covariance matrix.

    Returns:
        jnp.ndarray: 6x6 matrix, rates for P matrix.
    """
    return F @ P + P @ F.T + G @ Q_i @ G.T + Q_a


def evolve_P_ricatti2(
    P: jnp.ndarray,
    F: jnp.ndarray,
    Q_a: jnp.ndarray
) -> jnp.ndarray:
    """ Ricatti differential equation to propogate process noise P EKF
        workflows, but ignores the error model mapping (G matrix is not used).

    Args:
        P (jnp.ndarray): 6x6 matrix, process noise matrix.
        F (jnp.ndarray): 6x6 matrix, linearized dynamics model matrix.
        Q_a (jnp.ndarray): 6x6 matrix, additive process noise covariance matrix.

    Returns:
        jnp.ndarray: 6x6 matrix, rates for P matrix.
    """
    return F @ P + P @ F.T + Q_a
