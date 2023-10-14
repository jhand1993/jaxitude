""" Attitude rates from body rates.  For all functions defined here apart from
    evolve_w_from_MRP, the first argument is the state vector whose rate of
    change is returned by said function's call.  For example, evolve_CRP(q, w)
    returns dqdt, not dwdt.  This API decision is important when calling
    integrators and filters.
"""
import jax.numpy as jnp
# from jax import jit

from jaxitude.base import MiscUtil


def evolve_CRP(q: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """ Returns dq/dt given q(t) and w(t).

    Args:
        q (jnp.ndarray): 3x1 matrix of q parameteters.
        w (jnp.ndarray): Body angular rotation rates.

    Returns:
        jnp.ndarray: dq/dt at time t.
    """
    m = jnp.array(
        [[1. + q[0, 0]**2., q[0, 0] * q[1, 0] - q[2, 0], q[0, 0] * q[2, 0] + q[1, 0]],
         [q[0, 0] * q[1, 0] + q[2, 0], 1. + q[1, 0]**2., q[1, 0] * q[2, 0] - q[0, 0]],
         [q[0, 0] * q[2, 0] - q[1, 0], q[1, 0] * q[2, 0] + q[0, 0], 1. + q[2, 0]**2.]]
    ) * 0.5
    return jnp.dot(m, w)


def evolve_MRP(s: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """ Returns ds/dt given s(t) and w(t).

    Args:
        s (jnp.ndarray): 3x1 matrix of s parameters.
        w (jnp.ndarray): Body angular rotation rates.

    Returns:
        jnp.ndarray: ds/dt at time t.
    """
    s2 = (s.T @ s)[0, 0]
    m = jnp.array(
        [[1. - s2 + 2. * s[0, 0]**2., 2. * (s[0, 0] * s[1, 0] - s[2, 0]), 2. * (s[0, 0] * s[2, 0] + s[1, 0])],
         [2. * (s[0, 0] * s[1, 0] + s[2, 0]), 1. - s2 + 2. * s[1, 0]**2., 2. * (s[1, 0] * s[2, 0] - s[0, 0])],
         [2. * (s[0, 0] * s[2, 0] - s[1, 0]), 2. * (s[1, 0] * s[2, 0] + s[0, 0]), 1. - s2 + 2. * s[2, 0]**2.]]
    ) * 0.25
    return jnp.dot(m, w)


def evolve_MRP_Pmatrix(
    P: jnp.ndarray,
    F: jnp.ndarray,
    G: jnp.ndarray,
    Lambda: jnp.ndarray,
    Q: jnp.ndarray
) -> jnp.ndarray:
    """ Ricatti differential equation to propogate process noise P for MRP EKF
        workflow.

    Args:
        P (jnp.ndarray): Process noise matrix.
        F (jnp.ndarray): Linearized kinematics system matrix.
        G (jnp.ndarray): Linearized noise system matrix.
        Lambda (jnp.ndarray): MRP measurement covariance.
        Q (jnp.ndarray): Process noise covariance matrix.

    Returns:
        jnp.ndarray: Rates for P matrix.
    """
    return F @ P + P @ F.T + G @ Lambda @ G.T + Q


def evolve_w_from_MRP(s: jnp.ndarray, s_dot: jnp.ndarray) -> jnp.ndarray:
    """Returns w given s(t), s_dot(t).

    Args:
        s (jnp.ndarray): 3x1 matrix of s parameters.
        s_dot (jnp.ndarray) jnp.ndarray: 3x1 matrix of ds/dt at time t.

    Returns:
        jnp.ndarray: Body angular rotation rates.
    """
    s2 = (s.T @ s)[0, 0]
    m = jnp.array(
        [[1. - s2 + 2. * s[0, 0]**2., 2. * (s[0, 0] * s[1, 0] + s[2, 0]), 2. * (s[0, 0] * s[2, 0] - s[1, 0])],
         [2. * (s[0, 0] * s[1, 0] - s[2, 0]), 1. - s2 + 2. * s[1, 0]**2., 2. * (s[1, 0] * s[2, 0] + s[0, 0])],
         [2. * (s[0, 0] * s[2, 0] + s[1, 0]), 2. * (s[1, 0] * s[2, 0] - s[0, 0]), 1. - s2 + 2. * s[2, 0]**2.]]
    ) * 4. / (1. + s2)**2.
    return jnp.dot(m, s_dot)


def evolve_MRP_shadow(s: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """ Returns ds/dt from s(t) and w(t) for the corresponding MRP shadow
        set.

    Args:
        s (jnp.ndarray): 3x1 matrix of s parameteters.
        w (jnp.ndarray): Body angular rotation rates.

    Returns:
        jnp.ndarray: ds/dt for MRP shadow set at time t
    """
    s2 = (s.T @ s)[0, 0]
    in_shape = s.shape
    s_dot = evolve_MRP(w, s)
    return -s_dot / s2 + 0.5 * (1. + s2) / s2**2 * jnp.dot(
        s @ s.T, w.reshape((3, 1))
    ).reshape(in_shape)


def evolve_quat(b: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """ Returns db/dt given b(t) and w(t).

    Args:
        b (jnp.ndarray): 4x1 matrix of b parameteters.
        w (jnp.ndarray): Body angular rotation rates.

    Returns:
    jnp.ndarray: db/dt at time t.
    """
    # Append zero to w rate vector
    zero = jnp.zeros((1, 1))
    w4 = jnp.hstack([zero, w])

    row1 = jnp.array([b[0, 0], -b[1, 0], -b[2, 0], -b[3, 0]])
    row2 = jnp.array([b[1, 0], b[0, 0], -b[3, 0], b[2, 0]])
    row3 = jnp.array([b[2, 0], b[3, 0], b[0, 0], -b[1, 0]])
    row4 = jnp.array([b[3, 0], -b[2, 0], b[1, 0], b[0, 0]])

    # |b|^2=1 at all times, so enforcing orthonormality is important.
    m = jnp.vstack(
        [row1 / jnp.linalg.norm(row1),
         row2 / jnp.linalg.norm(row2),
         row3 / jnp.linalg.norm(row3),
         row4 / jnp.linalg.norm(row4)]
    ) * 0.5
    return jnp.dot(m, w4)


def euler_eqs(w: jnp.array, I: jnp.array) -> jnp.array:
    """ Euler equations of motion for a rigid body w.r.t. its center of mass.
        Torques are ignored.

    Args:
        w (jnp.array): rate vector as 3x1 matrix
        I (jnp.array): 3x3 matrix inertia tensor

    Returns:
        jnp.array: I*w_dot vector as 3x1 matrix
    """
    I_w_dot = -MiscUtil.cpo(w) @ I @ jnp.expand_dims(w, axis=-1)
    return I_w_dot.flatten()
