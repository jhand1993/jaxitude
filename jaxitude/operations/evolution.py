""" Attitude rates from body rates.  For all functions defined here apart from
    evolve_w_from_MRP, the first argument is the state vector whose rate of
    change is returned by said function's call.  For example, evolve_CRP(q, w)
    returns dqdt, not dwdt.  This API decision is important when calling
    integrators and filters.
"""
import jax.numpy as jnp
from jax import jit

from jaxitude.base import cpo
from jaxitude.operations.composition import compose_quat


@jit
def B_crp(q: jnp.ndarray) -> jnp.ndarray:
    """ Calculates B matrix relating q and w to dq/dt.

    Args:
        q (jnp.ndarray): 3x1 matrix, CRP q parameters.

    Returns:
        jnp.ndarray: B matrix for q.
    """
    return jnp.array(
        [[1. + q[0, 0]**2., q[0, 0] * q[1, 0] - q[2, 0], q[0, 0] * q[2, 0] + q[1, 0]],
         [q[0, 0] * q[1, 0] + q[2, 0], 1. + q[1, 0]**2., q[1, 0] * q[2, 0] - q[0, 0]],
         [q[0, 0] * q[2, 0] - q[1, 0], q[1, 0] * q[2, 0] + q[0, 0], 1. + q[2, 0]**2.]]
    )


@jit
def B_mrp(s: jnp.ndarray) -> jnp.ndarray:
    """ Calculates B matrix relating s and w to ds/dt.

    Args:
        q (jnp.ndarray): 3x1 matrix, MRP s parameters.

    Returns:
        jnp.ndarray: B matrix for s.
    """
    s2 = jnp.vdot(s, s)
    return jnp.array(
        [[1. - s2 + 2. * s[0, 0]**2., 2. * (s[0, 0] * s[1, 0] - s[2, 0]), 2. * (s[0, 0] * s[2, 0] + s[1, 0])],
         [2. * (s[0, 0] * s[1, 0] + s[2, 0]), 1. - s2 + 2. * s[1, 0]**2., 2. * (s[1, 0] * s[2, 0] - s[0, 0])],
         [2. * (s[0, 0] * s[2, 0] - s[1, 0]), 2. * (s[1, 0] * s[2, 0] + s[0, 0]), 1. - s2 + 2. * s[2, 0]**2.]]
    )


@jit
def inv_B_mrp(s: jnp.ndarray) -> jnp.ndarray:
    """ Calculates inverse B matrix relating s and ds/dt to w.

    Args:
        q (jnp.ndarray): 3x1 matrix, MRP s parameters.

    Returns:
        jnp.ndarray: inverse B matrix for s.
    """
    s2 = jnp.vdot(s, s)
    return jnp.array(
        [[1. - s2 + 2. * s[0, 0]**2., 2. * (s[0, 0] * s[1, 0] + s[2, 0]), 2. * (s[0, 0] * s[2, 0] - s[1, 0])],
         [2. * (s[0, 0] * s[1, 0] - s[2, 0]), 1. - s2 + 2. * s[1, 0]**2., 2. * (s[1, 0] * s[2, 0] + s[0, 0])],
         [2. * (s[0, 0] * s[2, 0] + s[1, 0]), 2. * (s[1, 0] * s[2, 0] - s[0, 0]), 1. - s2 + 2. * s[2, 0]**2.]]
    ) / (1. + s2)**2.


def evolve_CRP(
    q: jnp.ndarray,
    w: jnp.ndarray
) -> jnp.ndarray:
    """ Returns dq/dt given q(t) and w(t).

    Args:
        q (jnp.ndarray): 3x1 matrix, CRP q parameters.
        w (jnp.ndarray): 3x1 matrix, body angular rotation rates.

    Returns:
        jnp.ndarray: 3x1 matrix, dq/dt at time t.
    """
    return jnp.dot(0.5 * B_crp(q), w)


def evolve_MRP(
    s: jnp.ndarray,
    w: jnp.ndarray
) -> jnp.ndarray:
    """ Returns ds/dt given s(t) and w(t).

    Args:
        s (jnp.ndarray): 3x1 matrix, MRP s parameters.
        w (jnp.ndarray): 3x1 matrix, body angular rotation rates.

    Returns:
        jnp.ndarray: 3x1 matrix, ds/dt at time t.
    """
    return jnp.dot(0.25 * B_mrp(s), w)


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
        P (jnp.ndarray): 6x6 matrix, process noise matrix.
        F (jnp.ndarray): 6x6 matrix, linearized kinematics system matrix.
        G (jnp.ndarray): 6x6 matrix, linearized noise system matrix.
        Lambda (jnp.ndarray): 6x6 matrix, MRP measurement covariance.
        Q (jnp.ndarray): 6x6 matrix, process noise covariance matrix.

    Returns:
        jnp.ndarray: 6x6 matrix, rates for P matrix.
    """
    return F @ P + P @ F.T + Q
    # return F @ P + P @ F.T + G @ Lambda @ G.T + Q


def evolve_w_from_MRP(
    s: jnp.ndarray,
    s_dot: jnp.ndarray
) -> jnp.ndarray:
    """Returns w given s(t), s_dot(t).

    Args:
        s (jnp.ndarray): 3x1 matrix, MRP s parameters.
        s_dot (jnp.ndarray) jnp.ndarray: 3x1 matrix, ds/dt at time t.

    Returns:
        jnp.ndarray: 3x1 matrix, body angular rotation rates.
    """
    return jnp.dot(4. * inv_B_mrp(s), s_dot)


def evolve_MRP_shadow(
    s: jnp.ndarray,
    w: jnp.ndarray
) -> jnp.ndarray:
    """ Returns ds/dt from s(t) and w(t) for the corresponding MRP shadow
        set.

    Args:
        s (jnp.ndarray): 3x1 matrix, MRP s parameters.
        w (jnp.ndarray): 3x1 matrix, body angular rotation rates.

    Returns:
        jnp.ndarray: 3x1 matrix, ds/dt for MRP shadow set at time t
    """
    s2 = jnp.vdot(s, s)
    s_dot = evolve_MRP(w, s)
    return -s_dot / s2 + 0.5 * (1. + s2) / s2**2 * jnp.matmul(
        s @ s.T, w
    )


def evolve_quat(
    b: jnp.ndarray,
    w: jnp.ndarray
) -> jnp.ndarray:
    """ Returns db/dt given b(t) and w(t).

    Args:
        b (jnp.ndarray): 4x1 matrix, quaternion b parameters.
        w (jnp.ndarray): 3x1 matrix, body angular rotation rates.

    Returns:
        jnp.ndarray: 4x1 matrix, db/dt at time t.
    """
    # Append zero to w rate vector
    zero = jnp.zeros((1, 1))
    w_4 = jnp.vstack([zero, w])
    return compose_quat(b, 0.5 * w_4)


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
        jnp.ndarray: I*w_dot vector as 3x1 matrix
    """
    I_w_dot = -cpo(w) @ I @ jnp.expand_dims(w, axis=-1)
    return I_w_dot.flatten()
