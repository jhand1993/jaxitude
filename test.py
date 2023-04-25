import unittest
from attitude.primitives import R1, R2, R3, Primitive, DCM
from attitude.eulerangles import EulerAngle
from attitude.quaternions import Quaternion
import jax.numpy as jnp

class TestPrimitives(unittest.TestCase):
    """ Test for basic rotation matrices.
    """
    def test_R1(self):
        """ Simple pi/2 rotation test.
        """
        test = R1(jnp.pi / 2.)
        target = jnp.array(
            [[1., 0., 0.],
             [0., 0., -1.],
             [0., 1., 0.]]
        )
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                self.assertAlmostEqual(
                    test.dcm[i, j], target[i, j], places=7, msg='Incorrect R1 rotation matrix output.'
                )

    def test_R2(self):
        """ Simple pi/2 rotation test.
        """
        test = R2(jnp.pi / 2.)
        target = jnp.array(
            [[0., 0., 1.],
             [0., 1., 0.],
             [-1., 0., 0.]]
        )
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                self.assertAlmostEqual(
                    test.dcm[i, j], target[i, j], places=7, msg='Incorrect R2 rotation matrix output.'
                )

    def test_R3(self):
        """ Simple pi/2 rotation test.
        """
        test = R3(jnp.pi / 2.)
        target = jnp.array(
            [[0., -1., 0.],
             [1., 0., 0.],
             [0., 0., 1.]]
        )
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                self.assertAlmostEqual(
                    test.dcm[i, j], target[i, j], places=7, msg='Incorrect R3 rotation matrix output.'
                )
    
    def test_call(self):
        """ Make sure __call__ for primitive objects returns dcm attibute.
        """
        test = Primitive()
        # test.dcm should be 3x3 identity matrix
        self.assertAlmostEqual(test.dcm[0, 0], 1., msg='Incorrect call output.')
        self.assertAlmostEqual(test.dcm[1, 1], 1., msg='Incorrect call output.')
        self.assertAlmostEqual(test.dcm[2, 2], 1., msg='Incorrect call output.')


class TestEuler(unittest.TestCase):
    """ Test for Euler angle functionality.
    """
    def test_euler_conversion(self):
        test_angles = (0.3490658700466156, 0.1745329350233078, -0.1745329350233078)
        test = EulerAngle(test_angles)
        test_out = test.get_eulerangles('313')
        target_out = (2.7129145, 0.24619713, -2.3485403)
        for i in range(3):
            self.assertAlmostEqual(
                test_out[i], target_out[i], msg='Error in Euler angle calculation from dcm.'
            )
    

class TestPRV(unittest.TestCase):
    """ Test for PRV functionality. 
    """
    def test_PRV_from_euler(self):
        test_angles = (0.3490658700466156, -0.1745329350233078, 2.094395160675049)
        test = EulerAngle(test_angles)
        test_theta, test_e = test.get_eulerangles('313')
        target_theta = 2.146152973175049
        target_e = jnp.array(
            [[-0.97555053],
            [-0.12165573],
            [-0.18303284]]
        )
        self.assertAlmostEqual(test_theta, target_theta, msg='Error in PRV calcuation from dcm.')
        for i in range(3):
            self.assertAlmostEqual(test_e.flatten()[i], target_e.flatt()[i], msg='Error in PRV calcuation from dcm.')


class TestQuaternions(unittest.TestCase):
    """ Test for quaternion functionality.  
    """
    def test_quaternion(self):
        q = jnp.array([0.235702, 0.471405, -0.471405, 0.707107])
        q /= jnp.linalg.norm(q)
        q = tuple(q.tolist())

        test = Quaternion(q)
        target = jnp.array(
            [[-0.44444445, -0.11111217,  0.88888884],
             [-0.7777776 , -0.44444445, -0.44444492],
             [ 0.44444492, -0.88888884,  0.11111029]]
        )
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(test()[i, j], target[i, j], msg='Error in quaternion calcuation.')
    
    def test_get_q(self):
        R = [[-0.529403, -0.474115, 0.703525], [-0.467056, -0.529403, -0.708231], [0.708231, -0.703525, 0.0588291]]
        test = DCM(R)
        test_q = test.get_q_short()
        target_q = (0.002425426384434104, 0.4850696623325348, -0.4850696623325348, 0.7276048064231873)
        for i in range(4):
            self.assertAlmostEqual(test_q[i], target_q[i], msg='Error in get q calcuation.')