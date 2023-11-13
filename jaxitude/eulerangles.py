from typing import Tuple

import jax.numpy as jnp

from jaxitude.base import R1, R2, R3, Primitive, swapEuler_proper


class EulerAngle(Primitive):
    """ Euler angle rotation sequence class.  Rotations are intrinsic.
    """
    def __init__(self, angles: jnp.ndarray, order: str) -> None:
        """
        Attributes:
            angles (jnp.ndarray): 3x1 matrix of Euler angles alpha, beta,
                and gamma.
            order (str): Order of rotations as a string.  An example includes
                '121', 'xyx', or 'XYX for proper x-axis, y-axis, x-axis
                Euler angles.
            self.R_alpha (PrimitiveR): Rotation object for first rotation.
            self.R_beta (PrimitiveR): Rotation object for second rotation.
            self.R_gamma (PrimitiveR): Rotation object for third rotation.
            self.dcm (jnp.ndarray): Euler angle DCM.
        """
        # Define attributes.
        self.angles = angles.reshape((3, 1))
        self.order = self._order_decipher(order)
        f_alpha, f_beta, f_gamma = self._order_rotations(order)
        self.R_alpha = f_alpha(angles[0, 0])
        self.R_beta = f_beta(angles[1, 0])
        self.R_gamma = f_gamma(angles[2, 0])
        self.dcm = self.R_gamma() @ self.R_beta() @ self.R_alpha()

    def _order_decipher(self, order: str) -> str:
        """ Function to map input string to proper string format.

        Arguments:
            order (str): Order of rotations as a string.  An example includes
                '121', 'xyx', or 'XYX for proper x-axis, y-axis, x-axis
                Euler angles.

        Returns:
            str: order but in proper format.
        """
        mapper = {
            'x': '1',
            'y': '2',
            'z': '3',
            '1': '1',
            '2': '2',
            '3': '3'
        }

        return ''.join([mapper[i] for i in order.lower()])

    def _order_rotations(self, order: str) -> Tuple:
        """ Method to map input string to rotation order.

        Arguments:
            order (_type_): Order of rotations as a string.  An example
                includes '121', 'xyx', or 'XYX for proper x-axis, y-axis,
                x-axis Euler angles.

        Returns:
            Tuple: rotations of proper type per order.
        """
        order_proper = self._order_decipher(order)
        mapper = {
            '1': R1,
            '2': R2,
            '3': R3
        }

        return (mapper[i] for i in order_proper)

    def swapEuler_proper(self):
        """ Method to swap proper Euler angles with their valid counterparts:
            angle1 -> angle1 - pi, angle2 -> -angle2,
            angle3 -> angle3 - pi.  Returns a new EulerAngle object.

        Returns:
            EulerAngle: New EulerAngle object with swapped Euler angles.
        """
        # First, make sure the angle order is a proper Euler angle set.
        try:
            assert self.order[0] == self.order[2]

        except AssertionError:
            raise TypeError(
                f'Order {self.order} is not a proper Euler angle set.'
            )

        return self.__class__(
            swapEuler_proper(self.angles),
            self.order
        )
