""" CRP: Classical Rodrigues Parameters: q_i = b_i / b_0.
    MRP: Modified Rodrigues Parameters: s_i = b_i / (1 + b_0).
"""
import jax.numpy as jnp
from jax import jit

from jaxitude.base import Primitive


@jit
def shadow(s: jnp.array) -> jnp.array:
    """ Calculates shadow set from 3x1 matrix of MRP s values.

    Args:
        s (jnp.array): 3x1 matrix, MRP s set.

    Returns:
        jnp.array: 3x1 matrix, shadow set of MRP s set.
    """
    return -s / jnp.vdot(s, s)


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
