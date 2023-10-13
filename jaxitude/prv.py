"""
PRV: Principle Rotation Vector.
"""
import jax.numpy as jnp
from jax import jit

from jaxitude.base import Primitive


class PRV(Primitive):
    """ Principle rotation vector object.
    """
    def __init__(self, phi: float, e: jnp.ndarray) -> None:
        """
        Attributes:
            phi (float): principle rotation in radians.
            e (jax.ndarray): 3x1 array reprentation of principle axis.
            dcm (jax.ndarray): 3x3 rotation matrix.
        """
        super().__init__()
        self.phi = phi
        self.e = e
        self.dcm = PRV._build_prv(phi, e)

    @staticmethod
    @jit
    def _build_prv(phi: float, e: jnp.ndarray):
        """ Takes input phi scalar and e vector to build PRV rotation matrix.

        Args:
            phi (float): principle rotation in radians.
            e (jax.ndarray): 3x1 array reprentation of principle axis.

        Returns:
            jnp.ndarray: PRV 3x3 dcm.
        """
        cosphi = jnp.cos(phi)
        sinphi = jnp.sin(phi)
        sigma = 1. - cosphi
        e1, e2, e3 = e.copy().flatten()
        return jnp.array(
            [[e1**2. * sigma + cosphi, e1 * e2 * sigma + e3 * sinphi, e1 * e3 * sigma - e2 * sinphi],
             [e1 * e2 * sigma - e3 * sinphi, e2**2. * sigma + cosphi, e2 * e3 * sigma + e1 * sinphi],
             [e1 * e3 * sigma + e2 * sinphi, e2 * e3 * sigma - e1 * sinphi, e3**2. * sigma + cosphi]]
        )

    def get_b_from_PVR(self) -> jnp.ndarray:
        """ Calculates and returns Euler parameters directly from phi and e.

        Returns:
            jnp.ndarray: 4x1 matrix of Euler parameters b.
        """
        return jnp.ndarray(
            [[jnp.cos(self.phi / 2.)],
             [self.e[0] * jnp.sin(self.phi / 2.)],
             [self.e[1] * jnp.sin(self.phi / 2.)],
             [self.e[2] * jnp.sin(self.phi / 2.)]]
        )

    def get_q_from_PVR(self) -> jnp.ndarray:
        """ Calculates and returns CRP q directly from phi and e.

        Returns:
            jnp.ndarray: 3x1 matrix of CRP q values.
        """
        return jnp.ndarray(
            [[self.e[0] * jnp.tan(self.phi / 2.)],
             [self.e[1] * jnp.tan(self.phi / 2.)],
             [self.e[2] * jnp.tan(self.phi / 2.)]]
        )

    def get_s_from_PVR(self) -> jnp.ndarray:
        """ Calculates and returns MRP s directly from phi and e.

        Returns:
            jnp.ndarray: 1x3 matrix of MRP s values.
        """
        return jnp.ndarray(
            [[self.e[0] * jnp.tan(self.phi / 4.)],
             [self.e[1] * jnp.tan(self.phi / 4.)],
             [self.e[2] * jnp.tan(self.phi / 4.)]]
        )
