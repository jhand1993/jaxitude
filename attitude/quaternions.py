from attitude.primitives import Primitive
import jax.numpy as jnp

class Quaternion(Primitive):
    """ Quaternion rotation object. q0 is the scalar part and must 
        be the first element of the input tuple q. 
    """
    def __init__(self, q) -> None:
        """

        Args:
            q (tuple[float]): Tuples of quaternion parameters. First component
                is the scalar component
        
        Attributes:
            q (tuple[float]): Tuples of quaternion parameters. First component
                is the scalar component
        """
        super().__init__()
        assert jnp.abs(jnp.linalg.norm(jnp.asarray(q)) - 1.) < 1e-7, 'Elements of q must have unit length.'
        self.q = q
        self.dcm = self._build_quanterion(q)

    
    def _build_quanterion(self, q):
        """ Builds dcm from quaternion parameters. 

        Args:
            q (tuple): Tuples of quaternion parameters. First component
                is the scalar component

        Returns:
            jnp.ndarray: dcm derived from quanternion parameters q. 
        """
        q0, q1, q2, q3 = q

        return jnp.array(
            [[q0**2. + q1**2. - q2**2. - q3**2., 2. * (q1 * q2 + q0 * q3), 2. * (q1 * q3 - q0 * q2)],
            [2. * (q1 * q2 - q0 * q3), q0**2. - q1**2. + q2**2. - q3**2., 2. * (q2 * q3 + q0 * q1)],
            [2. * (q1 * q3 + q0 * q2), 2. * (q2 * q3 - q0 * q1), q0**2. - q1**2. - q2**2. + q3**2.]]
        )
    
    def get_PVR_from_q(self) -> tuple: 
        """ Generates PVR directly from q.

        Returns:
            tuple: sclar phi and 3x1 matrix e.
        """
        theta = 2. * jnp.arccos(self.q[0])
        e = jnp.array(
            [[self.q[1] / jnp.sin(theta)],
             [self.q[2] / jnp.sin(theta)],
             [self.q[3] / jnp.sin(theta)]]
        )
        return theta, e

