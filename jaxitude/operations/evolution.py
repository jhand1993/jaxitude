""" Simple attitude rate from body rate ODEs.
"""
import jax.numpy as jnp
from jax import jit

from jaxitude.base import MiscUtil


@jit
def evolve_CRP(omega: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """ Returns dq/dt given q(t) and omega(t).

    Args:
        omega (jnp.ndarray): Body angular rotation rates.
        q (jnp.ndarray): 3x1 matrix of q parameteters.

    Returns:
        jnp.ndarray: dq/dt at time t.
    """
    m = jnp.array(
        [[1. + q[0, 0]**2., q[0, 0] * q[1, 0] - q[2, 0], q[0, 0] * q[2, 0] + q[1, 0]],
         [q[0, 0] * q[1, 0] + q[2, 0], 1. + q[1, 0]**2., q[1, 0] * q[2, 0] - q[0, 0]],
         [q[0, 0] * q[2, 0] - q[1, 0], q[1, 0] * q[2, 0] + q[0, 0], 1. + q[2, 0]**2.]]
    ) * 0.5
    return jnp.dot(m, omega)


@jit
def evolve_MRP(omega: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    """Returns ds/dt given s(t) and omega(t).

    Args:
        omega (jnp.ndarray): Body angular rotation rates.
        s (jnp.ndarray): 3x1 matrix of s parameters.

    Returns:
        jnp.ndarray: ds/dt at time t.
    """
    s2 = (s.T @ s)[0, 0]
    m = jnp.array(
        [[1. - s2 + 2. * s[0, 0]**2., 2. * (s[0, 0] * s[1, 0] - s[2, 0]), 2. * (s[0, 0] * s[2, 0] + s[1, 0])],
         [2. * (s[0, 0] * s[1, 0] + s[2, 0]), 1. - s2 + 2. * s[1, 0]**2., 2. * (s[1, 0] * s[2, 0] - s[0, 0])],
         [2. * (s[0, 0] * s[2, 0] - s[1, 0]), 2. * (s[1, 0] * s[2, 0] + s[0, 0]), 1. - s2 + 2. * s[2, 0]**2.]]
    ) * 0.25
    return jnp.dot(m, omega)


@jit
def evolve_omega_from_MRP(s: jnp.ndarray, s_dot: jnp.ndarray) -> jnp.ndarray:
    """Returns omega given s(t), s_dot(t).

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


@jit
def evolve_MRP_shadow(omega: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    """ Returns ds/dt from s(t) and omega(t) for the corresponding MRP shadow
        set.

    Args:
        omega (jnp.ndarray): Body angular rotation rates.
        s (jnp.ndarray): 3x1 matrix of s parameteters.

    Returns:
        jnp.ndarray: ds/dt for MRP shadow set at time t
    """
    s2 = (s.T @ s)[0, 0]
    in_shape = s.shape
    s_dot = evolve_MRP(omega, s)
    return -s_dot / s2 + 0.5 * (1. + s2) / s2**2 * jnp.dot(
        s @ s.T, omega.reshape((3, 1))
    ).reshape(in_shape)


@jit
def evolve_quat(omega: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """ Returns db/dt given b(t) and omega(t).

    Args:
        omega (jnp.ndarray): Body angular rotation rates.
        b (jnp.ndarray): 4x1 matrix of b parameteters.

    Returns:
    jnp.ndarray: db/dt at time t.
    """
    # Append zero to omega rate vector
    zero = jnp.zeros((1, 1))
    omega4 = jnp.hstack([zero, omega])

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
    return jnp.dot(m, omega4)


@jit
def euler_eqs(omega: jnp.array, I: jnp.array) -> jnp.array:
    """ Euler equations of motion for a rigid body w.r.t. its center of mass.
        Torques are ignored.

    Args:
        omega (jnp.array): rate vector as 3x1 matrix
        I (jnp.array): 3x3 matrix inertia tensor

    Returns:
        jnp.array: I*omega_dot vector as 3x1 matrix
    """
    I_omega_dot = -MiscUtil.cpo(omega) @ I @ jnp.expand_dims(omega, axis=-1)
    return I_omega_dot.flatten()
