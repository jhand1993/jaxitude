""" Extended Kalman filter algorithms for quaternion sets b and MRP sets s.
    See https://link.springer.com/article/10.1007/BF03321529 for EKF MRP,

"""
import jax.numpy as jnp

from jaxitude.operations import evolution as ev
# from jaxitude.operations.integrator import autonomous_rk4


class MRPEKF(object):
    """ Container class for MRP EKF algorithm functionality.  The dynamics
        model is dxdt = f(x|w) + g(x,eta|w), with f(x|w) describing the system's
        kinematics and g capturing the g(x,eta|w) describing the noise
        evolution.
    """
    @staticmethod
    def f(
        x: jnp.ndarray,
        w: jnp.ndarray,
    ) -> jnp.ndarray:
        """ Kinematics equation f for evolve MRPs.

        Args:
            s (jnp.ndarray): State vector matrix with shape 6x1.  First three
                components are MRP values, second three are bias measurements.
            w (jnp.ndarray): 3x1 matrix of measured rate vector.

        Returns:
            jnp.ndarray: State rate vector matrix with shape 6x1.
        """
        return jnp.vstack(
            [ev.evolve_MRP(w, x[:3, :]), jnp.zeros((3, 1))]
        )

    @staticmethod
    def g(
        x: jnp.ndarray,
        w: jnp.ndarray,
        eta: jnp.ndarray,
    ) -> jnp.ndarray:
        """ Noise evolution equation g for evolve MRPs.

        Args:
            s (jnp.ndarray): State vector matrix with shape 6x1.  First three
                components are MRP values, second three are gyro bias
                measurements.
            w (jnp.ndarray): 3x1 matrix of measured rate vector.
            w (jnp.ndarray): 6x1 matrix of noise vectors. First three
                components are MRP noise, second three are gyro bias noise.

        Returns:
            jnp.ndarray: State rate vector matrix with shape 6x1.
        """
        return jnp.vstack(
            [ev.evolve_MRP(w, x[:3, :]), jnp.zeros((3, 1))]
        )