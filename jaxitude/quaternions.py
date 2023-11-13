from typing import Tuple

import jax.numpy as jnp
from jax import jit

from jaxitude.base import Primitive, cpo


@jit
def compose_quat(b_p: jnp.ndarray, b_pp: jnp.ndarray) -> jnp.ndarray:
    """ Quaternion composition b = b_p x b_pp

    Args:
        b_p (jnp.ndarray): 4x1 matrix, first quaternion parameters.
        b_pp (jnp.ndarray): 4x1 matrix, second quaterion parameters.

    Returns:
        jnp.ndarray: 4x1 matrix, composition of quaternion parameters.
    """
    matrix = jnp.array(
        [[b_pp[0, 0], -b_pp[1, 0], -b_pp[2, 0], -b_pp[3, 0]],
         [b_pp[1, 0], b_pp[0, 0], b_pp[3, 0], -b_pp[2, 0]],
         [b_pp[2, 0], -b_pp[3, 0], b_pp[0, 0], b_pp[1, 0]],
         [b_pp[3, 0], b_pp[2, 0], -b_pp[1, 0], b_pp[0, 0]]]
    )

    # Make sure to satisfy unity condition for quaternion parameters.
    b = jnp.matmul(matrix, b_p)
    return b


@jit
def xi_op(b: jnp.ndarray) -> jnp.ndarray:
    """ 4x3 Xi matrix operator representation of quaternion composition b.

    Args:
        b (jnp.ndarray): 4x1 matrix, quaternion parameters.

    Returns:
        jnp.ndarray: 4x3 matrix, Xi matrix operator.
    """
    return jnp.vstack(
        [-b[1:, :].T[0],
         b[0] + cpo(b[1:, :])]
    )


@jit
def psi_op(b: jnp.ndarray) -> jnp.ndarray:
    """ 4x3 Psi matrix operator representation of quaternion composition.

    Args:
        b (jnp.ndarray): 4x1 matrix, quaternion parameters.

    Returns:
        jnp.ndarray: 4x3 matrix, Psi matrix operator.
    """
    return jnp.vstack(
        [-b[1:, :].T[0],
         b[0] - cpo(b[1:, :])]
    )


@jit
def quat_expm(v: jnp.ndarray) -> jnp.ndarray:
    """ Quaternion exponential map.

    Args:
        v (jnp.ndarray): 3x1 matrix, vector to map to quaternion set b. If
        |v| < 0.01, then a linear approximation is used.

    Returns:
        jnp.ndarray: 4x1 matrix, quaternion set b from exponential map of v.
    """
    angle = jnp.linalg.norm(v)
    return jnp.where(
        jnp.abs(angle) > 0.01,
        quat_expm_angleaxis(angle, v / angle),
        jnp.array(
            [[1.],
             [v[0, 0] * 0.5],
             [v[1, 0] * 0.5],
             [v[2, 0] * 0.5]]
        )
    )


@jit
def quat_expm_angleaxis(
    angle: float,
    e: jnp.ndarray
) -> jnp.ndarray:
    """ Quaternion exponential map given angle and unit vector e. If
        |v| < 0.01, then a linear approximation is used.

    Args:
        angle (float): Angle in radians.
        e (jnp.ndarray): 3x1 matrix, unit vector to map to quaternion set b.

    Returns:
        jnp.ndarray: 4x1 matrix, quaternion set b from exponential map of v.
    """
    return jnp.where(
        jnp.abs(angle) > 0.,
        jnp.array(
            [[jnp.cos(angle * 0.5)],
             [e[0, 0] * jnp.sin(angle * 0.5)],
             [e[1, 0] * jnp.sin(angle * 0.5)],
             [e[2, 0] * jnp.sin(angle * 0.5)]]
        ),
        jnp.array(
            [[1.],
             [e[0, 0] * 0.5],
             [e[1, 0] * 0.5],
             [e[2, 0] * 0.5]]
        )
    )


@jit
def quat_inv(b: jnp.ndarray) -> jnp.ndarray:
    """ Returns the inverse quaternion set.

    Args:
        b (jnp.ndarray): 4x1 matrix, quaternion b set to get inverse of.

    Returns:
        jnp.ndarray: 4x1 matrix, quaternion b set inverse.
    """
    return jnp.array(
        [[b[0, 0]],
         [-b[1, 0]],
         [-b[2, 0]],
         [-b[3, 0]]]
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
    w_4 = jnp.vstack([jnp.zeros((1, 1)), w])
    return compose_quat(b, 0.5 * w_4)


class Quaternion(Primitive):
    """ quaternion rotation object. b0 is the scalar part and must
        be the first element of the input tuple b.
    """
    def __init__(self, b: jnp.ndarray) -> None:
        """

        Args:
            b (jnp.ndarray): 4x1 matrix, quaternion parameters b. First
                component is the scalar component.

        Attributes:
            b (jnp.ndarray): 4x1 matrix, quaternion parameters b. First
                component is the scalar component.
            dcm (jax.ndarray): 3x3 matrix, rotation matrix.
        """
        super().__init__()
        self.b = b
        self.dcm = Quaternion._build_quaternion_dcm(b)

    @staticmethod
    @jit
    def _build_quaternion_dcm(
        b: jnp.ndarray
    ) -> jnp.ndarray:
        """ Builds dcm from quaternion parameters.

        Args:
            b (jnp.ndarray): 4x1 matrix, quaternion parameters. First component
                is the scalar component.

        Returns:
            jnp.ndarray: 3x3 matrix, DCM derived from quanternion parameters b.
        """
        # Need to explicitly unpack for this to be jittable.
        b0 = b[0, 0]
        b1 = b[1, 0]
        b2 = b[2, 0]
        b3 = b[3, 0]

        return jnp.array(
            [[b0**2. + b1**2. - b2**2. - b3**2., 2. * (b1 * b2 + b0 * b3), 2. * (b1 * b3 - b0 * b2)],
             [2. * (b1 * b2 - b0 * b3), b0**2. - b1**2. + b2**2. - b3**2., 2. * (b2 * b3 + b0 * b1)],
             [2. * (b1 * b3 + b0 * b2), 2. * (b2 * b3 - b0 * b1), b0**2. - b1**2. - b2**2. + b3**2.]]
        )

    def get_PRV_from_b(self) -> Tuple:
        """ Generates PVR directly from b.  If the rotation angle is
            sufficiently close to a 4n * pi rotation, then the principal axis is
            returned as a vector with values sqrt(3) / 3, since the principal
            axis is degenerate for such rotations.

        Returns:
            tuple: scalar phi and 3x1 matrix e.
        """
        theta = 2. * jnp.arccos(self.b[0, 0])

        # This conditional compares delta theta values above and below 4n * pi.
        # That way, if theta is close 4n * pi, stability is ensured.
        cond = jnp.logical_or(
            jnp.isclose(
                0., theta - jnp.ceil(0.25 * theta / jnp.pi) * 4. * jnp.pi,
                atol=5e-4
            ),
            jnp.isclose(
                0., theta - jnp.floor(0.25 * theta / jnp.pi) * 4. * jnp.pi,
                atol=5e-4
            )
        )
        e = jnp.where(
            cond,
            jnp.full((3, 1), jnp.sqrt(3.) / 3.),
            jnp.array(
                [[self.b[1, 0] / jnp.sin(theta * 0.5)],
                 [self.b[2, 0] / jnp.sin(theta * 0.5)],
                 [self.b[3, 0] / jnp.sin(theta * 0.5)]]
            )
        )
        return theta, e / jnp.linalg.norm(e)

    def get_q_from_b(self) -> jnp.ndarray:
        """ Generates CRP q vector from b.

        Returns:
            jnp.ndarray: 3x1 matrix, CRP q parameters.
        """
        return jnp.array(
            [[self.b[1, 0] / self.b[0, 0]],
             [self.b[2, 0] / self.b[0, 0]],
             [self.b[3, 0] / self.b[0, 0]]]
        )

    def get_s_from_b(self) -> jnp.ndarray:
        """ Generate MRP s vector from b.

        Returns:
            jnp.ndarray: 3x1 matrix, MRP s parameters.
        """
        return jnp.array(
            [[self.b[1, 0] / (1. + self.b[0, 0])],
             [self.b[2, 0] / (1. + self.b[0, 0])],
             [self.b[3, 0] / (1. + self.b[0, 0])]]
        )

    def get_inv(self):
        """ Returns a new Quaternion instance with inverse b set

        Returns:
            Quaternion: New inverse quaternion instance.
        """
        return self.__class__(quat_inv(self.b))
