""" CRP: Classical Rodrigues Parameters: q_i = b_i / b_0.
    MRP: Modified Rodrigues Parameters: s_i = b_i / (1 + b_0).
"""
import jax.numpy as jnp
from jax import jit

from jaxitude.base import Primitive, colvec_cross


@jit
def compose_crp(q_p: jnp.ndarray, q_pp: jnp.ndarray) -> jnp.ndarray:
    """ Compose CRP parameters directly in the following order:
        R(q) = R(q_pp)R(q_p) for CRP rotation matrix R.

    Args:
        q_p (jnp.ndarray): First CRP parameters as 3x1 matrix.
        q_pp (jnp.ndarray): second CRP parameters as 3x1 matrix.

    Returns:
        jnp.ndarray: Composed CRP parameters as 3x1 matrix.
    """
    num = jnp.subtract(
        jnp.add(q_pp, q_p),
        colvec_cross(q_pp, q_p)
    )
    denom = 1. - jnp.vdot(q_p, q_pp)
    return num / denom


@jit
def compose_mrp(s_p: jnp.ndarray, s_pp: jnp.ndarray) -> jnp.ndarray:
    """ Compose MRP parameters direclty in the following order:
        R(s) = R(s_pp)R(s_p) for MRP rotation matrix R.

    Args:
        s_p (jnp.ndarray): First MRP parameters as 1x3 matrix.
        s_pp (jnp.ndarray): Second MRP parameters as 1x3 matrix.

    Returns:
        jnp.ndarray: composed MRP parameters as 1x3 matrix.
    """
    dot_p = jnp.vdot(s_p, s_p)
    dot_pp = jnp.vdot(s_pp, s_pp)
    cross_spp_sp = colvec_cross(s_pp, s_p)
    return ((1. - dot_pp) * s_p + (1. - dot_p) * s_pp - 2. * cross_spp_sp) /\
        (1. + dot_p * dot_pp - 2. * jnp.vdot(s_p, s_pp))


@jit
def shadow(s: jnp.array) -> jnp.array:
    """ Calculates shadow set from 3x1 matrix of MRP s values.

    Args:
        s (jnp.array): 3x1 matrix, MRP s set.

    Returns:
        jnp.array: 3x1 matrix, shadow set of MRP s set.
    """
    return -s / jnp.vdot(s, s)


@jit
def B_crp(q: jnp.ndarray) -> jnp.ndarray:
    """ Calculates B matrix relating q and w to dq/dt.

    Args:
        q (jnp.ndarray): 3x1 matrix, CRP q parameters.

    Returns:
        jnp.ndarray: 3x3 matrix, B matrix for q.
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
        jnp.ndarray: 3x3 matrix, B matrix for s.
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
        jnp.ndarray: 3x3 matrix, inverse B matrix for s.
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


class CRP(Primitive):
    """ Classical Rodrigues parameter rotation class.
    """
    def __init__(self, q: jnp.ndarray) -> None:
        super().__init__()
        self.q = q
        self.dcm = CRP._build_crp_dcm(q)

    @staticmethod
    @jit
    def _build_crp_dcm(q: jnp.ndarray) -> jnp.ndarray:
        """ Builds dcm from CRP q parameters.

        Args:
            q (jnp.ndarray): 3xx matrix of CRP parameters

        Returns:
            jnp.ndarray: 3x3 rotation matrix
        """
        c = 1. + jnp.vdot(q, q)
        return jnp.array(
            [[1. + q[0, 0]**2. - q[1, 0]**2. - q[2, 0]**2., 2. * (q[0, 0] * q[1, 0] + q[2, 0]), 2. * (q[0, 0] * q[2, 0] - q[1, 0])],
             [2. * (q[0, 0] * q[1, 0] - q[2, 0]), 1. - q[0, 0]**2. + q[1, 0]**2. - q[2, 0]**2., 2. * (q[1, 0] * q[2, 0] + q[0, 0])],
             [2. * (q[0, 0] * q[2, 0] + q[1, 0]), 2. * (q[1, 0] * q[2, 0] - q[0, 0]), 1. - q[0, 0]**2. - q[1, 0]**2. + q[2, 0]**2.]]
        ) / c

    def get_b_from_q(self) -> jnp.ndarray:
        """ Builds Euler parameters from CRP q.

        Returns:
            jnp.ndarray: 4x1 matrix of Euler parameters b.
        """
        q_dot_q = jnp.vdot(self.q, self.q)
        b0 = 1. / jnp.sqrt(1. + q_dot_q)
        return jnp.array(
            [[b0],
             [self.q[0, 0] * b0],
             [self.q[1, 0] * b0],
             [self.q[2, 0] * b0]]
        )

    def get_s_from_q(self) -> jnp.ndarray:
        """ Build MRP s from CRP q.

        Returns:
            jnp.ndarray: 3x1 matrix of MRP s parameters
        """
        denom = 1. + jnp.sqrt(1. + jnp.vdot(self.q, self.q))
        return self.q / denom

    def inv_copy(self):
        """ Returns new instance of CRP with q -> -q.

        Returns:
            CRP: new instance inverse CRP.
        """
        return self.__class__(-self.q)


class MRP(Primitive):
    """ Classical Rodrigues parameter rotation class.
    """
    def __init__(self, s: jnp.ndarray) -> None:
        super().__init__()
        self.s = s
        self.dcm = MRP._build_mrp_dcm(s)

    @staticmethod
    @jit
    def _build_mrp_dcm(s: jnp.ndarray) -> jnp.ndarray:
        """ Builds dcm from MRP s parameters.

        Args:
            s (jnp.ndarray): 3x1 matrix of MRP s parameters

        Returns:
            jnp.ndarray: 3x3 rotation matrix
        """
        c = 1. - jnp.vdot(s, s)
        return jnp.array(
            [[4. * (s[0, 0]**2. - s[1, 0]**2. - s[2, 0]**2.) + c**2., 8. * s[0, 0] * s[1, 0] + 4. * c * s[2, 0], 8. * s[0, 0] * s[2, 0] - 4. * c * s[1, 0]],
             [8. * s[0, 0] * s[1, 0] - 4. * c * s[2, 0], 4. * (- s[0, 0]**2. + s[1, 0]**2. - s[2, 0]**2.) + c**2., 8. * s[1, 0] * s[2, 0] + 4. * c * s[0, 0]],
             [8. * s[0, 0] * s[2, 0] + 4. * c * s[1, 0], 8. * s[1, 0] * s[2, 0] - 4. * c * s[0, 0], 4. * (- s[0, 0]**2. - s[1, 0]**2. + s[2, 0]**2.) + c**2.]]
        ) / (1. + jnp.vdot(s, s))**2.

    def get_b_from_s(self) -> jnp.ndarray:
        """ Builds quaternion b from MRP s.

        Returns:
            jnp.ndarray: 4x1 matrix, quaternion b.
        """
        s_dot_s = jnp.vdot(self.s, self.s)
        c = 1. / (1. + s_dot_s)
        return jnp.array(
            [[(1. - s_dot_s) * c],
             [2. * self.s[0, 0] * c],
             [2. * self.s[1, 0] * c],
             [2. * self.s[2, 0] * c]]
        )

    def get_q_from_s(self) -> jnp.ndarray:
        """ Build CRP q from MRP s.

        Returns:
            jnp.ndarray: 3x1 matrix of CRP q parameters:
        """
        denom = 1. - jnp.vdot(self.s, self.s)
        return 2. * self.s / denom

    def inv_copy(self):
        """ Returns new instance of MRP with s -> -s.

        Returns:
            MRP: new instance inverse MRP.
        """
        return self.__class__(-self.s)

    def get_shadow(self):
        """ Returns shadow set instance of MRP class.

        Returns:
            CRP: new instance shadow MRP.
        """
        s_shadow = shadow(self.s)
        return self.__class__(s_shadow)
