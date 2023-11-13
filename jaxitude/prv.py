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
            e (jax.ndarray): 3x1 matrx, principle axis vector.
            dcm (jax.ndarray): 3x3 matrix, rotation matrix.
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
        cphi = jnp.cos(phi)
        sphi = jnp.sin(phi)
        sigma = 1. - cphi
        return jnp.array(
            [[e[0, 0]**2. * sigma + cphi, e[0, 0] * e[1, 0] * sigma + e[2, 0] * sphi, e[0, 0] * e[2, 0] * sigma - e[1, 0] * sphi],
             [e[0, 0] * e[1, 0] * sigma - e[2, 0] * sphi, e[1, 0]**2. * sigma + cphi, e[1, 0] * e[2, 0] * sigma + e[0, 0] * sphi],
             [e[0, 0] * e[2, 0] * sigma + e[1, 0] * sphi, e[1, 0] * e[2, 0] * sigma - e[0, 0] * sphi, e[2, 0]**2. * sigma + cphi]]
        )

    def get_b_from_PVR(self) -> jnp.ndarray:
        """ Calculates and returns Euler parameters directly from phi and e.

        Returns:
            jnp.ndarray: 4x1 matrix of Euler parameters b.
        """
        return jnp.array(
            [[jnp.cos(self.phi * 0.5)],
             [self.e[0, 0] * jnp.sin(self.phi * 0.5)],
             [self.e[1, 0] * jnp.sin(self.phi * 0.5)],
             [self.e[2, 0] * jnp.sin(self.phi * 0.5)]]
        )

    def get_q_from_PVR(self) -> jnp.ndarray:
        """ Calculates and returns CRP q directly from phi and e.

        Returns:
            jnp.ndarray: 3x1 matrix of CRP q values.
        """
        return jnp.array(
            [[self.e[0, 0] * jnp.tan(self.phi * 0.5)],
             [self.e[1, 0] * jnp.tan(self.phi * 0.5)],
             [self.e[2, 0] * jnp.tan(self.phi * 0.5)]]
        )

    def get_s_from_PVR(self) -> jnp.ndarray:
        """ Calculates and returns MRP s directly from phi and e.

        Returns:
            jnp.ndarray: 1x3 matrix of MRP s values.
        """
        return jnp.array(
            [[self.e[0, 0] * jnp.tan(self.phi * 0.25)],
             [self.e[1, 0] * jnp.tan(self.phi * 0.25)],
             [self.e[2, 0] * jnp.tan(self.phi * 0.25)]]
        )
