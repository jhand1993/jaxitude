""" Functions for adding noise to data.
"""
from typing import Tuple

import jax.numpy as jnp
from jax import jit
from jax.random import split, normal, uniform

from jaxitude.prv import PRV
from jaxitude.base import MiscUtil


class HeadingErrorModel():
    """ Heading vector error model for producing headings measurements with
        errors from some true heading measurement.

        A heading vector v is necessarily a unit vector, which complicates
        slightly trying to add error to a heading vector v.  For example,
        additive white noise of the form v_obs = v + err, where err~N(0, Cov_v)
        does not result in a proper heading vector v_obs, since v + err is
        not guaranteed to be a heading vector.
    """
    @staticmethod
    def addnoise(
        key: int,
        v: jnp.ndarray,
        sigma_angle: float
    ) -> jnp.ndarray:
        """ Add noise to heading vector v. Heading vector should be unit vector.

        Args:
            key (int): Random key.
            v (jnp.ndarray): 3x1 matrix, unit vector to add noise to.
            sigma_angle (float): Standard deviation for noise PDF in radians.

        Returns:
            jnp.ndarray: 3x1 matrix, noise-corrupted unit vector.
        """

        # Get thetas and random rotation axes.
        key, subkey = split(key)
        theta = HeadingErrorModel.theta_pdf(subkey)
        u, w = HeadingErrorModel.get_perp_vectors(v)
        e = HeadingErrorModel.combination(u, w, theta)

        # Get phi and randomly rotate.
        key, subkey = split(key)
        phi = HeadingErrorModel.phi_pdf(subkey, sigma_angle)
        return jnp.matmul(PRV(phi, e)(), v)

    @staticmethod
    def combination(
        x: jnp.ndarray,
        y: jnp.ndarray,
        theta: float,
    ) -> jnp.ndarray:
        """ Given two orthogonal unit vectors x and y, return a unit vector
            which is a linear combination of the two with coefficients
            u = cos(theta) * x + sin(theta) * y.

        Args:
            x (jnp.ndarray): Nx1 matrix, unit column vector.
            y (jnp.ndarray): Nx1 matrix, unit column vector.
            theta (float): Coefficient for basis vector x.

        Returns:
            jnp.ndarray: 3x1 matrix, unit column vector combination of x and y.
        """
        u = jnp.cos(theta) * x + jnp.sin(theta) * y
        return u / jnp.linalg.norm(u)

    @staticmethod
    def get_perp_vectors(
        v: jnp.ndarray,
    ) -> Tuple[jnp.ndarray]:
        """ Generates a set of orthogonal vector compliments to v using cross
            products.

            First, a candidate vector u_c is selected by permuting the indices
            of v: u_c = [v[2], v[0], v[1]].T.  u is the cross product of v and
            u_c.  A third orthogonal vector w is then chosen by crossing v and
            u.

        Args:
            v (jnp.ndarray): 3x1 matrix, unit vector.

        Returns:
            Tuple[jnp.ndarray]: Tuple of 3x1 matrices, two orthogonal unit
                vectors that are also orthogonal to v.
        """
        # First, get the candidate vector.
        u_c = jnp.array(
            [[v[2, 0]],
             [v[0, 0]],
             [v[1, 0]]]
        )

        # Second, get first orthogonal compliment and normalize it.
        u = MiscUtil.colvec_cross(v, u_c)
        u = u / jnp.linalg.norm(u)

        # third, get second orthogonal compliment and normalize it.
        w = MiscUtil.colvec_cross(v, u)
        w = w / jnp.linalg.norm(w)

        return u, w

    @staticmethod
    @jit
    def theta_pdf(
        key: int,
    ) -> jnp.ndarray:
        """ Uniform PDF from 0 to 2pi for calculating rotation axis vector.

        Args:
            key (int): Random key.

        Returns:
            jnp.ndarray: jnp.ndarray of sampled uniform elements with shape
                'shape'.
        """
        return uniform(key, maxval=2. * jnp.pi)

    @staticmethod
    @jit
    def phi_pdf(
        key: int,
        sigma_phi: float
    ) -> jnp.ndarray:
        """ Normal PDF centered at zero with standard deviation sigma_phi.
            Used to sample random rotation angles.

        Args:
            key (int): Random key.
            sigma_phi (float): Standard deviation of phi_pdf in radians.

        Returns:
            jnp.ndarray: jnp.ndarray of normally-sampled elements with shape
                'shape'.
        """
        return normal(key) * sigma_phi
