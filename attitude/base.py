"""
All attitudes will be represented by products of primitive rotations R,
where R is a rotation along some coordinate axis.  Here R1, will be a
rotation along the first coordinate component, R2 the second component,
and R3 the third. For example, a 3-2-1 (Z-X-Y) Euler angle rotation
sequence by angles (a, b, c) will be M(a,b,c) = R1(c)R2(b)R3(c).
"""
from typing import Tuple
from functools import partial

import jax.numpy as jnp
from jax import jit


# This maps Euler type to angles, given a rotation matrix R.
eulerangle_map = {
    '131': lambda R: (
        jnp.arctan2(R[0, 2], R[0, 1]),
        jnp.arccos(R[0, 0]),
        jnp.arctan2(R[2, 0], -R[1, 0])
    ),
    '121': lambda R: (
        jnp.arctan2(R[0, 1], -R[0, 2]),
        jnp.arccos(R[0, 0]),
        jnp.arctan2(R[1, 0], R[2, 0])
    ),
    '212': lambda R: (
        jnp.arctan2(R[1, 0], R[1, 2]),
        jnp.arccos(R[1, 1]),
        jnp.arctan2(R[0, 1], -R[2, 1])
    ),
    '232': lambda R: (
        jnp.arctan2(R[1, 2], -R[1, 0]),
        jnp.arccos(R[1, 1]),
        jnp.arctan2(R[2, 1], R[0, 1])
    ),
    '323': lambda R: (
        jnp.arctan2(R[2, 1], R[2, 0]),
        jnp.arctan2(jnp.sqrt(1. - R[2, 2]**2), R[2, 2]),
        jnp.arctan2(R[1, 2], -R[0, 2])
    ),
    '313': lambda R: (
        jnp.arctan2(R[2, 0], -R[2, 1]),
        jnp.arccos(R[2, 2]),
        jnp.arctan2(R[0, 2], R[1, 2])
    ),
    '132': lambda R: (
        jnp.arctan2(R[2, 1], R[1, 1]),
        jnp.arcsin(-R[0, 1]),
        jnp.arctan2(R[0, 2], R[0, 0])
    ),
    '123': lambda R: (
        jnp.arctan2(-R[1, 2], R[2, 2]),
        jnp.arcsin(R[0, 2]),
        jnp.arctan2(R[0, 1], R[0, 0])
    ),
    '213': lambda R: (
        jnp.arctan2(R[0, 2], R[2, 2]),
        jnp.arcsin(-R[1, 2]),
        jnp.arctan2(R[1, 0], R[1, 1])
    ),
    '231': lambda R: (
        jnp.arctan2(-R[0, 2], R[0, 0]),
        jnp.arcsin(R[1, 0]),
        jnp.arctan2(-R[1, 2], R[1, 1])
    ),
    '321': lambda R: (
        jnp.arctan2(R[0, 1], R[0, 0]),
        jnp.arcsin(-R[0, 2]),
        jnp.arctan2(R[1, 2], R[2, 2])
    ),
    '312': lambda R: (
        jnp.arctan2(-R[0, 1], R[1, 1]),
        jnp.arcsin(R[2, 1]),
        jnp.arctan2(-R[2, 0], R[2, 2])
    )
}


class MiscUtil(object):
    """ Container class for miscellaneous caluclations.
    """
    @staticmethod
    @jit
    def antisym_dcm_vector(dcm: jnp.ndarray) -> jnp.ndarray:
        """ Returns [
                dcm[1, 2] - dcm[2, 1],
                dcm[2, 0] - dcm[0, 2],
                dcm[0, 1] - dcm[1, 0]
            ], a vector that is used in extracting PRV, CRP q, and
            MRP s.

        Args:
            dcm (jnp.ndarray): 3x3 dcm matrix.

        Returns:
            jnp.ndarray: 1x3 matrix
        """
        return jnp.array(
            [dcm[1, 2] - dcm[2, 1],
             dcm[2, 0] - dcm[0, 2],
             dcm[0, 1] - dcm[1, 0]]
        )

    @staticmethod
    @jit
    def cpo(v: jnp.ndarray) -> jnp.ndarray:
        """ Matrix representation of cross product operator of vector v.

        Args:
            v (jnp.ndarray): 1x3 matrix (or broadcastable) representation
            of 3D vector v.

        Returns:
            jnp.ndarray: 3x3 matrix of cross product operation.
        """
        v_f = v.flatten()
        return jnp.array(
            [[0., -v_f[2], v_f[1]],
             [v_f[2], 0., -v_f[0]],
             [-v_f[1], v_f[0], 0.]]
        )

    @staticmethod
    @jit
    def swapEuler_proper(angles: jnp.array) -> jnp.array:
        """ Swaps proper Euler angles (form i-j-i) as follows:
            angle1 -> angle1 % pi, angle2 -> -angle2,
            angle3 -> angle3 % pi.  Angles should be given in radians.

        Args:
            angles (jnp.array): 1x3 matrix of proper Euler angles.

        Returns:
            jnp.array: 1x3 matrix of swapped proper Euler angles.
        """
        return jnp.array(
            [
                angles[0] - jnp.sign(angles[0]) * jnp.pi,
                -angles[1],
                angles[2] - jnp.sign(angles[2]) * jnp.pi
            ]
        )


class PRVUtil(object):
    """ Container class for PRV calculations from dcm.
    """
    @staticmethod
    @jit
    def get_e(dcm: jnp.ndarray) -> jnp.ndarray:
        """ Calculates e from dcm.

        Args:
            dcm (jnp.ndarray): dcm 3x3 matrix

        Returns:
            jnp.ndarray: 1x3 array representation of e
        """
        phi = PRVUtil.get_phi(dcm)
        e_raw = MiscUtil.antisym_dcm_vector(dcm) * 0.5 / jnp.sin(phi)
        return e_raw / jnp.linalg.norm(e_raw)

    @staticmethod
    @jit
    def get_phi(dcm: jnp.ndarray) -> float:
        """ Calculates phi from dcm.

        Args:
            dcm (jnp.ndarray): dcm 3x3 matrix

        Returns:
            float: phi
        """
        return jnp.arccos(0.5 * (dcm[0, 0] + dcm[1, 1] + dcm[2, 2] - 1.))


class Primitive(object):
    """ Object with __call__ returning self.dcm. Base class for all
        rotations and includes transformation equations from dcm.
        Base dcm attribute is a zero rotation (identity matrix).
    """
    def __init__(self) -> None:
        # Set to identity for primitives.  Does change state in subclass
        # overides, but shouldn't cause issues later.
        self.dcm = jnp.identity(3)

    def __call__(self) -> jnp.ndarray:
        """ Returns DCM matrix.

        Returns:
            jnp.ndarray: self.dcm
        """
        return self.dcm

    def get_eig(self) -> Tuple:
        """ Wrapper to call JAX.numpy.linalg.eig().

        Returns:
            Tuple: array of eigenvalues and array of eigenvectors.
        """
        return jnp.linalg.eig(self.dcm)

    def get_prv(self) -> Tuple:
        """ Returns principle angle phi and principle vector e from
            rotation matrix.

        Returns:
            Tuple: phi, vec(e)
        """
        return PRVUtil.get_phi(self.dcm), PRVUtil.get_e(self.dcm)

    def get_prv2(self) -> Tuple:
        """ Returns principle angle phi and principle vector e from
            rotation matrix for to long rotation phi' = phi - 2pi.

        Returns:
            Tuple: phi, vec(e)
        """
        return PRVUtil.get_phi(self.dcm) - 2. * jnp.pi, PRVUtil.get_e(self.dcm)

    def get_eulerangles(self, ea_type: str) -> jnp.ndarray:
        """ Returns a 1x3 matrix of Euler angles from DCM.

        Args:
            ea_type (str): Euler angle type.  Needs to be of form
                '121', '321', etc for now.
        Returns:
            jnp.ndarray: 1x3 matrix of Euler angles
        """
        return jnp.asarray(eulerangle_map[ea_type](self.dcm))

    # @partial(jit, static_argnums=0)
    def _get_b_base(self) -> jnp.ndarray:
        """ Returns a matrix of quaternion parameters from dcm. Uses Shepard's
            method to avoid singularity at b0=0.  Doesn't decide shortest path.

        Returns:
            jnp.ndarray: 1x4 matrix of quaternion parameters.
        """
        tr = jnp.trace(self.dcm)
        step1 = jnp.array(
            [
                0.25 * (1. + tr),
                0.25 * (1. + 2. * self.dcm[0, 0] - tr),
                0.25 * (1. + 2. * self.dcm[1, 1] - tr),
                0.25 * (1. + 2. * self.dcm[2, 2] - tr)
            ]
        )

        # Important: to jit, you need to cast using jnp.ndarray.astype.
        max_i = jnp.argmax(step1).item()
        step2 = jnp.array(
            [
                0.25 * (self.dcm[1, 2] - self.dcm[2, 1]),
                0.25 * (self.dcm[2, 0] - self.dcm[0, 2]),
                0.25 * (self.dcm[0, 1] - self.dcm[1, 0]),
                0.25 * (self.dcm[1, 2] + self.dcm[2, 1]),
                0.25 * (self.dcm[2, 0] + self.dcm[0, 2]),
                0.25 * (self.dcm[1, 2] + self.dcm[2, 1]),
            ]
        )
        max_sq = jnp.sqrt(step1[max_i])
        choices = {
            0: jnp.array(
                [
                    max_sq,
                    step2[0] / max_sq,
                    step2[1] / max_sq,
                    step2[2] / max_sq
                ]
            ),
            1: jnp.array(
                [
                    step2[0] / max_sq,
                    max_sq,
                    step2[3] / max_sq,
                    step2[4] / max_sq
                ]
            ),
            2: jnp.array(
                [
                    step2[1] / max_sq,
                    step2[3] / max_sq,
                    max_sq,
                    step2[5] / max_sq
                ]
            ),
            3: jnp.array(
                [
                    step2[2] / max_sq,
                    step2[4] / max_sq,
                    step2[5] / max_sq,
                    max_sq
                ]
            )
        }
        return choices[max_i]

    def get_b_short(self) -> jnp.ndarray:
        """ Shepard's method to get b from DCM. Makes sure b0 is positive.

        Returns:
            jnp.ndarray: 1x4 matrix of quaternion parameters.
        """
        b = self._get_b_base()
        return b.at[0].set(jnp.abs(b[0]))

    def get_b_long(self) -> jnp.ndarray:
        """ Shepard's method to get b from DCM. Makes sure b0 is negative.

        Returns:
            jnp.ndarray: 1x4 matrix of quaternion parameters.
        """
        b = self._get_b_base()
        return b.at[0].set(-jnp.abs(b[0]))

    def get_q(self) -> jnp.ndarray:
        """ Gets CRP q parameters from DCM.

        Returns:
            jnp.ndarray: 1x3 matrix of CRP q parameters.
        """
        zeta_squared = jnp.trace(self.dcm) + 1.
        return MiscUtil.antisym_dcm_vector(self.dcm) / zeta_squared

    def get_s(self) -> jnp.ndarray:
        """ Gets MRP s parameters from DCM.

        Returns:
            jnp.ndarray: 1x3 matrix of MRP s parameters.
        """
        zeta = jnp.sqrt(jnp.trace(self.dcm) + 1.)
        return MiscUtil.antisym_dcm_vector(self.dcm) / zeta / (zeta + 2.)


class BaseR(Primitive):
    """ Fundamental coordinate axis rotation primitive subclass.

    """
    def __init__(self, angle: float) -> None:
        """
        Attributes:
            angle (float): Rotation angle in radians.
        """
        super().__init__()
        self.angle = angle

    def inv_copy(self):
        """ Return a new instance of the same rotation class with a negative
        angle.

        Returns:
            self.__class__: New instance of the same class but with negative
            angle.
        """
        return self.__class__(-self.angle)


class R1(BaseR):
    """ Fundamental passive rotation w.r.t. coordinate axis 1.

    Args:
        BaseR: Base class
    """
    def __init__(self, angle: float) -> None:
        """
        Attibutes:
            rotor (jnp.ndarray): Overwrites primitive definition
                appropriate for R1 rotation.
        """
        super().__init__(angle)
        self.dcm = jnp.array(
            [[1., 0., 0.],
             [0., jnp.cos(angle), jnp.sin(angle)],
             [0., -jnp.sin(angle), jnp.cos(angle)]]
        )


class R2(BaseR):
    """ Fundamental passive rotation w.r.t. coordinate axis 2.

    Args:
        BaseR: Base class
    """
    def __init__(self, angle: float) -> None:
        """
        Attibutes:
            rotor (jnp.ndarray): Overwrites primitive definition
                appropriate for R2 rotation.
        """
        super().__init__(angle)
        self.dcm = jnp.array(
            [[jnp.cos(angle), 0., -jnp.sin(angle)],
             [0., 1., 0.],
             [jnp.sin(angle), 0., jnp.cos(angle)]]
        )


class R3(BaseR):
    """ Fundamental passive rotation w.r.t. coordinate axis 3.

    Args:
        BaseR: Base class
    """
    def __init__(self, angle: float) -> None:
        """
        Attibutes:
            rotor (jnp.ndarray): Overwrites primitive definition
                appropriate for R3 rotation.
        """
        super().__init__(angle)
        self.dcm = jnp.array(
            [[jnp.cos(angle), jnp.sin(angle), 0.],
             [-jnp.sin(angle), jnp.cos(angle), 0.],
             [0., 0., 1.]]
        )


class DCM(Primitive):
    """ Custom DCM

    Args:
        BaseR: Base class
    """
    def __init__(self, matrix: jnp.ndarray) -> None:
        """ Builds custom DCM instance

        Args:
            matrix (jnp.ndarray): _description_
        """
        super().__init__()
        self.dcm = jnp.asarray(matrix)
        assert self.dcm.shape == (3, 3), 'Invalid matrix shape.'
