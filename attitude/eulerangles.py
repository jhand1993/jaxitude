from attitude.primitives import R1, R2, R3, Primitive

class EulerAngle(Primitive):
    """ Euler angle rotation sequence class.  Rotations are intrinsic. 
    """
    def __init__(self, angles, order) -> None:
        """
        Attributes:
            alpha (float): First rotation angle in radians. 
            beta (float): Second rotation angle in radians.
            gamma (float): Third rotation angle in radians
            order (str): Order of rotations as a string.  An example includes 
                '121', 'xyx', or 'XYX for proper x-axis, y-axis, x-axis 
                Euler angles. 
            self.R_alpha (PrimitiveR): Rotation object for first rotation. 
            self.R_beta (PrimitiveR): Rotation object for second rotation. 
            self.R_gamma (PrimitiveR): Rotation object for third rotation. 
            self.dcm (jnp.ndarray): Euler angle DCM.
        """
        self.alpha, self.beta, self.gamma = angles
        self.order = self._order_decipher(order)
        f_alpha, f_beta, f_gamma = self._order_rotations(order)
        self.R_alpha, self.R_beta, self.R_gamma = f_alpha(angles[0]), f_beta(angles[1]), f_gamma(angles[2])
        self.dcm = self.R_alpha() @ self.R_beta() @ self.R_gamma()
    
    def _order_decipher(self, order) -> str:
        """ Function to map input string to proper string format.

        Args:
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

    def _order_rotations(self, order) -> tuple:
        """ Function to map input string to rotation order. 

        Args:
            order (_type_): Order of rotations as a string.  An example includes 
                '121', 'xyx', or 'XYX for proper x-axis, y-axis, x-axis 
                Euler angles. 
        Returns:
            tuple: rotations of proper type per order. 
        """
        order_proper = self._order_decipher(order)
        mapper = {
            '1': R1,
            '2': R2,
            '3': R3
        }
        return (mapper[i] for i in order_proper)