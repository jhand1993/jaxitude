import unittest

import jax.numpy as jnp
from jax import config

from jaxitude.base import R1, R2, R3, Primitive, DCM, MiscUtil
from jaxitude.eulerangles import EulerAngle
from jaxitude.quaternions import Quaternion
from jaxitude.rodrigues import CRP, MRP
from jaxitude.operations.composition import compose_quat, relative_crp, compose_mrp
from jaxitude.determination.triad import get_triad_dcm
from jaxitude.determination.davenport import get_K
from jaxitude.determination.quest import quest_get_CRPq
from jaxitude.determination.olae import olae_get_CRPq

# Double precision needed for testing.
config.update("jax_enable_x64", True)


class TestPrimitives(unittest.TestCase):
    """ Test for basic rotation matrices.
    """
    def test_R1(self):
        """ Simple pi/2 rotation test.
        """
        test = R1(jnp.pi / 2.)
        target = jnp.array(
            [[1., 0., 0.],
             [0., 0., 1.],
             [0., -1., 0.]]
        )
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                self.assertAlmostEqual(
                    test.dcm[i, j], target[i, j],
                    msg='Incorrect R1 rotation matrix output.'
                )

    def test_R2(self):
        """ Simple pi/2 rotation test.
        """
        test = R2(jnp.pi / 2.)
        target = jnp.array(
            [[0., 0., -1.],
             [0., 1., 0.],
             [1., 0., 0.]]
        )
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                self.assertAlmostEqual(
                    test.dcm[i, j], target[i, j],
                    msg='Incorrect R2 rotation matrix output.'
                )

    def test_R3(self):
        """ Simple pi/2 rotation test.
        """
        test = R3(jnp.pi / 2.)
        target = jnp.array(
            [[0., 1., 0.],
             [-1., 0., 0.],
             [0., 0., 1.]]
        )
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                self.assertAlmostEqual(
                    test.dcm[i, j], target[i, j],
                    msg='Incorrect R3 rotation matrix output.'
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
    def test_euler_conversion_proper(self):
        test_angles = jnp.array(
            [[0.3490658700466156],
             [-0.1745329350233078],
             [2.094395160675049]]
        )
        test1 = EulerAngle(test_angles, '313')
        test1_out = MiscUtil.swapEuler_proper(test1.get_eulerangles('313'))
        target_out = test_angles.copy()
        for i in range(3):
            self.assertAlmostEqual(
                test1_out[i], target_out[i],
                msg='Error in proper Euler angle calculation from dcm.'
            )

    def test_euler_conversion_tb(self):
        test_angles = jnp.array(
            [[0.3490658700466156],
             [-0.1745329350233078],
             [2.094395160675049]]
        )
        test2 = EulerAngle(test_angles, '321')
        test2_out = test2.get_eulerangles('321')
        target_out = test_angles.copy()
        for i in range(3):
            self.assertAlmostEqual(
                test2_out[i], target_out[i],
                msg='Error in TB Euler angle calculation from dcm.'
            )


class TestPRV(unittest.TestCase):
    """ Test for PRV functionality.
    """
    def test_PRV_from_euler(self):
        test_angles = jnp.array(
            [[0.17453293],
             [0.43633231],
             [-0.26179939]]
        )
        test = EulerAngle(test_angles, '321')
        test_theta, test_e = test.get_prv()
        target_theta = 0.55459931377
        target_e = jnp.array(
            [[-0.532035],
             [0.740302],
             [0.410964]]
        )
        self.assertAlmostEqual(
            test_theta, target_theta,
            places=3,
            msg='Error in PRV calcuation from dcm.'
        )
        for i in range(3):
            self.assertAlmostEqual(
                test_e.flatten()[i], target_e.flatten()[i],
                places=3,
                msg='Error in PRV calcuation from dcm.'
            )


class TestQuaternions(unittest.TestCase):
    """ Test for quaternion functionality.
    """
    def test_quaternion(self):
        # Make sure to respect the normalization condition.
        b = jnp.array(
            [[0.235702],
             [0.471405],
             [-0.471405],
             [0.707107]]
        )
        b /= jnp.linalg.norm(b)

        test = Quaternion(b)
        target = jnp.array(
            [[-0.44444445, -0.11111217, 0.88888884],
             [-0.7777776, -0.44444445, -0.44444492],
             [0.44444492, -0.88888884, 0.11111029]]
        )
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(
                    test()[i, j], target[i, j],
                    places=6,
                    msg='Error in quaternion calcuation.'
                )

    def test_get_b(self):
        R = [[-0.529403, -0.474115, 0.703525],
             [-0.467056, -0.529403, -0.708231],
             [0.708231, -0.703525, 0.0588291]]
        test = DCM(R)
        test_b = test.get_b_short()
        target_b = jnp.array(
            [[0.002425426384434104],
             [0.4850696623325348],
             [-0.4850696623325348],
             [0.7276048064231873]]
        )
        for i in range(4):
            self.assertAlmostEqual(
                test_b[i], target_b[i],
                msg='Error in get quaternion b calculation.'
            )


class TestCRP(unittest.TestCase):
    """ Tests for CRP operations
    """
    def test_q_from_dcm(self):
        test_r = jnp.asarray(
            [[0.333333, 0.871795, -0.358974],
             [-0.666667, 0.487179, 0.564103],
             [0.666667, 0.0512821, 0.74359]]
        ).T
        test_dcm = DCM(test_r)
        test_q = test_dcm.get_q()
        target_q = jnp.array(
            [[-0.20000021],
             [-0.40000013],
             [-0.6000004]])
        for i in range(3):
            self.assertAlmostEqual(
                test_q[i], target_q[i],
                places=6,
                msg='Error in CVR q calculation.'
            )

    def test_dcm_from_q(self):
        q = jnp.array(
            [[0.1],
             [0.2],
             [0.3]]
        )
        test_dcm = CRP(q).dcm
        target_dcm = jnp.array(
            [[0.7719298, 0.5614036, -0.2982456],
             [-0.49122807, 0.82456136, 0.28070176],
             [0.40350878, -0.07017544, 0.9122807]]
        )
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(
                    test_dcm[i, j], target_dcm[i, j],
                    places=6,
                    msg='Error in DCM calculation from q.'
                )

    def test_dcm_from_s(self):
        s = jnp.array(
            [[0.1],
             [0.2],
             [0.3]]
        )
        test_dcm = MRP(s).dcm
        target_dcm = jnp.array(
            [[0.19975376, 0.91720533, -0.34472147],
             [-0.6709757, 0.38442597, 0.63404125],
             [0.7140659, 0.10464759, 0.692213]]
        )
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(
                    test_dcm[i, j], target_dcm[i, j],
                    places=6,
                    msg='Error in DCM calculation from s.'
                )

    def test_s_from_dcm(self):
        test_r = jnp.array(
            [[0.763314, 0.568047, -0.307692],
             [-0.0946746, -0.372781, -0.923077],
             [-0.639053, 0.733728, -0.230769]]
        )
        test_s = DCM(test_r).get_s()
        target_s = jnp.array(
            [[-0.4999999],
             [-0.09999997],
             [0.19999984]]
        )
        for i in range(3):
            self.assertAlmostEqual(
                test_s[i], target_s[i], msg='Error in MVR s calculation.'
            )


class TestCompositions(unittest.TestCase):
    """ Test for special compositions operations.
    """
    def test_compose_quat(self):
        b_p = jnp.array(
            [[0.774597],
             [0.258199],
             [0.516398],
             [0.258199]]
        )
        b_p /= jnp.linalg.norm(b_p)
        b_pp = jnp.array(
            [[0.359211],
             [-0.898027],
             [-0.179605],
             [-0.179605]])
        b_pp /= jnp.linalg.norm(b_pp)

        test_b = compose_quat(b_p, b_pp)
        target_b = jnp.array(
            [[0.64923435],
             [-0.64923435],
             [-0.13912137],
             [0.37099165]]
        )
        for i in range(4):
            self.assertAlmostEqual(
                test_b[i], target_b[i],
                msg='Error in direct quaternion composition.'
            )

    def test_compose_CRP(self):
        q1 = jnp.array(
            [[0.1],
             [0.2],
             [0.3]]
        )
        q2 = jnp.array(
            [[-0.3],
             [0.3],
             [0.1]]
        )
        test_q = relative_crp(q2, q1)
        target_q = jnp.array(
            [[-0.31132078],
             [0.18867928],
             [-0.27358493]])
        for i in range(3):
            self.assertAlmostEqual(
                test_q[i], target_q[i],
                msg='Error in direct CRP q composition.'
            )

    def test_compose_MRP(self):
        s2 = jnp.array(
            [[0.1],
             [0.2],
             [0.3]]
        )
        s1 = -jnp.array(
            [[0.5],
             [0.3],
             [0.1]]
        )
        test_s = compose_mrp(s1, s2)
        target_s = jnp.array(
            [[-0.37998495],
             [0.11437171],
             [-0.02332581]])
        for i in range(3):
            self.assertAlmostEqual(
                test_s[i], target_s[i],
                msg='Error in direct MRP s composition.'
            )


class TestTriad(unittest.TestCase):
    """ Test triad functionality.
    """
    def test_get_dcm(self):
        v1_b = jnp.array(
            [[0.8273],
             [0.5541],
             [-0.0920]]
        )
        v2_b = jnp.array(
            [[-0.8285],
             [0.5522],
             [-0.0955]]
        )

        v1_n = jnp.array(
            [[-0.1517],
             [-0.9669],
             [0.2050]]
        )
        v2_n = jnp.array(
            [[-0.8393],
             [0.4494],
             [-0.3044]])
        test_dcm = get_triad_dcm(v1_b, v2_b, v1_n, v2_n)
        target_dcm = jnp.array(
            [[0.4155587, -0.85509086, 0.3100492],
             [-0.83393234, -0.49427602, -0.24545473],
             [0.36313602, -0.15655923, -0.9184887]]
        )
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(
                    test_dcm[i, j], target_dcm[i, j],
                    places=5,
                    msg='Error in Triad DCM calculation.'
                )


class TestDavenport(unittest.TestCase):
    """ Test Davenport's q method functionality.
    """
    def test_get_K(self):
        vb = jnp.array(
            [[0.8273, 0.5541, -0.092],
             [-0.8285, 0.5522, -0.0955]]
        ).T
        vn = jnp.array(
            [[-0.1517, -0.9669, 0.205],
             [-0.8393, 0.4494, -0.3044]]
        ).T
        w = jnp.array([1., 0.5])
        test_K = get_K(w, vb, vn)
        target_K = jnp.array(
            [[-0.19382623, -0.0379503, -0.24166122, -0.6702926],
             [-0.0379503, 0.6381834, -1.3018681, 0.34972718],
             [-0.24166122, -1.3018681, -0.62953365, 0.0970416],
             [-0.6702926, 0.34972718, 0.0970416, 0.18517643]]
        )
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(
                    test_K[i, j], target_K[i, j],
                    places=5,
                    msg='Error in Davenport get_K calculation.'
                )


class TestQuest(unittest.TestCase):
    """ Test QUEST functionality
    """
    def test_get_q(self):
        vb = jnp.array(
            [[0.8273, 0.5541, -0.092],
             [-0.8285, 0.5522, -0.0955]]
        ).T
        vn = jnp.array(
            [[-0.1517, -0.9669, 0.205],
             [-0.8393, 0.4494, -0.3044]]
        ).T
        w = jnp.array([1., 0.5])
        test_q = quest_get_CRPq(w, vb, vn)
        target_q = jnp.array(
            [[-31.83776],
             [19.00673],
             [-7.576603]]
        )

        for i in range(3):
            self.assertAlmostEqual(
                test_q[i], target_q[i],
                places=2,
                msg='Error in quest_get_CRPq calculation.'
            )


class TestOLAE(unittest.TestCase):
    """ Test OLAE functionality.
    """
    def test_olae(self):
        vb = jnp.array(
            [[0.8273, 0.5541, -0.092],
             [-0.8285, 0.5522, -0.0955]]
        ).T
        vn = jnp.array(
            [[-0.1517, -0.9669, 0.205],
             [-0.8393, 0.4494, -0.3044]]
        ).T
        w = jnp.array([1., 0.5])
        test_q = olae_get_CRPq(w, vb, vn)
        target_q = jnp.array(
            [[-31.83776],
             [19.00673],
             [-7.576603]]
        )
        for i in range(3):
            self.assertAlmostEqual(
                test_q[i], target_q[i],
                places=1,
                msg='Error in olae_get_CRPq calculation.'
            )
