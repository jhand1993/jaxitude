""" 
All attitudes will be represented by products of primitive rotations R, 
where R is a rotation along some coordinate axis.  Here R1 will be a rotation 
along the first coordinate component, R2 the second component, and R3 the third. 
For example, a 3-2-1 (Z-X-Y) Euler angle rotation sequence by angles (a, b, c)
will be:
M(a,b,c) = R1(c)R2(b)R3(c).
"""
import jax.numpy as jnp

# This maps Euler type to angles, given a rotation matrix R.
eulerangle_map = {
    '131': lambda R: (
        jnp.arctan2(R[2, 0], R[2, 1]),
        jnp.arccos(R[0, 0]),
        jnp.arctan2(R[0, 2], -R[1, 2])
    ),
    '121': lambda R: (
        jnp.arctan2(R[1, 0], -R[2, 0]),
        jnp.arccos(R[0, 0]),
        jnp.arctan2(R[0, 1], R[0, 2])       
    ),
    '212': lambda R: (
        jnp.arctan2(R[0, 1], R[2, 1]),
        jnp.arccos(R[1, 1]),
        jnp.arctan2(R[1, 0], -R[1, 2])
    ),
    '232': lambda R: (
        jnp.arctan2(R[2, 1], -R[0, 1]),
        jnp.arccos(R[1, 1]),
        jnp.arctan2(R[1, 2], R[1, 0])
    ),
    '323': lambda R: (
        jnp.arctan2(R[1, 2], R[0, 2]),
        jnp.arctan2(jnp.sqrt(1. - R[2, 2]**2), R[2, 2]),
        jnp.arctan2(R[2, 1], -R[2, 0])
    ),
    '313': lambda R: (
        jnp.arctan2(R[0, 2], -R[1, 2]),
        jnp.arccos(R[2, 2]),
        jnp.arctan2(R[2, 0], R[2, 1])
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
        jnp.arctan2(R[1, 0], R[0, 0]),
        jnp.arcsin(-R[2, 0]),
        jnp.arctan2(R[2, 1], R[2, 2])
    ),
    '312': lambda R: (
        jnp.arctan2(-R[0, 1], R[1, 1]),
        jnp.arcsin(R[2, 1]),
        jnp.arctan2(-R[2, 0], R[2, 2])
    )
}

class PRVUtil(object):
    """ Container class for PRV calculations from dcm. 
    """
    @staticmethod
    def get_e(dcm) -> jnp.ndarray:
        """ Calculates e from dcm.

        Args:
            dcm (jnp.ndarray): dcm matrix

        Returns:
            jnp.ndarray: 3x1 array representation of e
        """
        phi = PRVUtil.get_phi(dcm)
        sinphi = jnp.sin(phi)
        return jnp.array(
            [[dcm[1, 2] - dcm[2, 1]],
            [dcm[2, 0] - dcm[0, 2]],
            [dcm[0, 1] - dcm[1, 0]]]
        ) * 0.5 / sinphi

    @staticmethod
    def get_phi(dcm) -> float:
        """ Calculates phi from dcm.

        Args:
            dcm (jnp.ndarray): dcm matrix

        Returns:
            float: phi
        """
        return jnp.arccos(0.5 * (dcm[0, 0] + dcm[1, 1] + dcm[2, 2] - 1.))


class Primitive(object):
    """ Object with __call__ returning self.dcm. Base class for all 
        rotations and includes transformation equations from dcm. 
        Base dcm attribute is a null rotation (identity matrix). 
    """
    def __init__(self) -> None:
        # Set to identity for primitives.  Does change state in subclass overrides,
        # but shouldn't cause issues down the road...
        self.dcm = jnp.identity(3)

    def __call__(self) -> jnp.ndarray:
        """ Call returns the rotor attribute. 

        Returns:
            jnp.ndarray: self.dcm
        """
        return self.dcm
    
    def get_eig(self) -> tuple:
        """ Wrapper to call JAX.numpy.linalg.eig.

        Returns:
            tuple: array of eigenvalues and array of eigenvectors.
        """
        return jnp.linalg.eig(self.dcm)

    def get_prv(self) -> tuple:
        """ Returns principle angle phi and principle vector e from rotation matrix.

        Returns:
            tuple: phi, vec(e)
        """
        return float(PRVUtil.get_phi(self.dcm)), PRVUtil.get_e(self.dcm)

    def get_prv2(self) -> tuple:
        """ Returns principle angle phi and principle vector e from rotation matrix for 
            to long rotation phi' = phi - 2pi. 

        Returns:
            tuple: phi, vec(e)
        """
        return float(PRVUtil.get_phi(self.dcm)) - 2. * jnp.pi, PRVUtil.get_e(self.dcm)

    def get_eulerangles(self, ea_type) -> tuple:
        """ Returns a tuple of euler angles for given order from dcm.  

        Args: 
            ea_type (str): Euler angle type.  Needs to be of form
                '121', '321', etc for now.
        Returns:
            tuple: set of Euler angles
        """
        return tuple(float(x) for x in eulerangle_map[ea_type](self.dcm))
    
    def _get_q_base(self) -> tuple:
        """ Returns a tuple of quaternion parameters from dcm. Uses Shepard's method 
            to avoid singularity at q0=0.  Doesn't decide shortest path. 

        Returns:
            tuple: Tuple of quaternion parameters. 
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
        max_i = int(jnp.argmax(step1))
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
            0: (max_sq, step2[0] / max_sq, step2[1] / max_sq, step2[2] / max_sq),
            1: (step2[0] / max_sq, max_sq, step2[3] / max_sq, step2[4] / max_sq),
            2: (step2[1] / max_sq, step2[3] / max_sq, max_sq, step2[5] / max_sq),
            3: (step2[2] / max_sq, step2[4] / max_sq, step2[5] / max_sq, max_sq)
        }
        return tuple(float(x) for x in choices[max_i])

    def get_q_short(self) -> tuple:
        """ Makes sure q0 is positive.

        Returns:
            tuple: Tuple of quaternion parameters. 
        """
        q_tup = self._get_q_base()
        return tuple([float(jnp.abs(q_tup[0])), *q_tup[1:]])

    def get_q_long(self) -> tuple:
        """ Makes sure q0 is negative.

        Returns:
            tuple: Tuple of quaternion parameters. 
        """
        q_tup = self._get_q_base()
        return tuple([-float(jnp.abs(q_tup[0])), *q_tup[1:]])

class PrimitiveR(Primitive):
    """ Fundamental coordinate axis rotation primitive subclass.  

    """
    def __init__(self, angle) -> None:
        """
        Attributes:
            angle (str): Rotation angle in radians. 
        """
        super().__init__()
        self.angle = angle

    def inv(self):
        """ Return a new instance of the same rotation class with a negative angle.

        Returns:
            self.__class__: New instance of the same class but with negative angle.
        """
        return self.__class__(-self.angle)

class R1(PrimitiveR):
    """ Fundamental rotation w.r.t. coordinate axis 1.

    Args:
        PrimitiveR: Base class
    """
    def __init__(self, angle) -> None:
        """_summary_

        Attibutes:
            rotor (jnp.ndarray): Overwrites primitive definition
                appropriate for R1 rotation. 
        """
        super().__init__(angle)
        self.dcm = jnp.array(
            [[1., 0., 0.],
             [0., jnp.cos(angle), -jnp.sin(angle)],
             [0., jnp.sin(angle), jnp.cos(angle)]]
        )

class R2(PrimitiveR):
    """ Fundamental rotation w.r.t. coordinate axis 2.

    Args:
        PrimitiveR: Base class
    """
    def __init__(self, angle) -> None:
        """_summary_

        Attibutes:
            rotor (jnp.ndarray): Overwrites primitive definition
                appropriate for R2 rotation. 
        """
        super().__init__(angle)
        self.dcm = jnp.array(
            [[jnp.cos(angle), 0., jnp.sin(angle)],
             [0., 1., 0.],
             [-jnp.sin(angle), 0., jnp.cos(angle)]]
        )


class R3(PrimitiveR):
    """ Fundamental rotation w.r.t. coordinate axis 3.

    Args:
        PrimitiveR: Base class
    """
    def __init__(self, angle) -> None:
        """_summary_

        Attibutes:
            rotor (jnp.ndarray): Overwrites primitive definition
                appropriate for R3 rotation. 
        """
        super().__init__(angle)
        self.dcm = jnp.array(
            [[jnp.cos(angle), -jnp.sin(angle), 0.],
             [jnp.sin(angle), jnp.cos(angle), 0.],
             [0., 0., 1.]]
        )

class DCM(Primitive):
    """ Custom DCM
    
    Args:
        PrimitiveR: Base class  
    """
    def __init__(self, matrix) -> None:
        super().__init__()
        self.dcm = jnp.asarray(matrix)
        assert self.dcm.shape == (3, 3), 'Invalid matrix shape.'