""" Functions for adding noise to data.
"""
from typing import Tuple

import jax.numpy as jnp
from jax import jit
from jax.random import split, normal, uniform

from jaxitude.prv import PRV
from jaxitude.quaternions import Quaternion, quat_expm_angleaxis
from jaxitude.quaternions import compose_quat
from jaxitude.base import colvec_cross


@jit
def theta_pdf(
    key: int,
) -> float:
    """ Uniform PDF from 0 to 2pi used when calculating perpendicular
        rotation axis vector.

    Args:
        key (int): Random key.

    Returns:
        float: Sampled theta.
    """
    return uniform(key, maxval=2. * jnp.pi)


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
    u = colvec_cross(v, u_c)
    u = u / jnp.linalg.norm(u)

    # third, get second orthogonal compliment and normalize it.
    w = colvec_cross(v, u)
    w = w / jnp.linalg.norm(w)

    return u, w


class HeadingNoise():
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
        sigma_dphi: float
    ) -> jnp.ndarray:
        """ Add noise to heading vector v. Heading vector should be unit vector.

        Args:
            key (int): Random key.
            v (jnp.ndarray): 3x1 matrix, unit vector to add noise to.
            sigma_dphi (float): Standard deviation for noise PDF in radians.

        Returns:
            jnp.ndarray: 3x1 matrix, noise-corrupted unit vector.
        """
        # Get thetas and random rotation axes.
        key, subkey = split(key)
        theta = theta_pdf(subkey)
        u, w = get_perp_vectors(v)
        e = combination(u, w, theta)

        # Get phi and randomly rotate.
        key, subkey = split(key)
        dphi = HeadingNoise.dphi_pdf(subkey, sigma_dphi)
        return jnp.matmul(PRV(dphi, e)(), v)

    @staticmethod
    @jit
    def dphi_pdf(
        key: int,
        sigma_dphi: float
    ) -> float:
        """ Normal PDF centered at zero with standard deviation sigma_phi.
            Used to sample random rotation angles.

        Args:
            key (int): Random key.
            sigma_dphi (float): Standard deviation of dphi_pdf in radians.

        Returns:
            float: Sampled phi.
        """
        return normal(key) * sigma_dphi


class QuatNoise():
    """ This quaternion noise model uses quaternion exponentiation and
        composition to inject error into a 'true' quaternion set b.

        For a given input quaternion b (the 'true' b), the observed quaterion
        b_obs is defined as the composition of b with the exponential map of
        perturbing rotations dv = dtheta * e, where e is the PRV axis of b:
        b_obs = composition(b, exp([0, dv].T))

        It is possible to consider a quaternion perturbation where the principal
        axis vector e is also perturbed by some small amount de:
        dv = dtheta * e + theta * de,
        but this has the effect of necessarily reducing the actual perturbed
        principal rotation angle Delta(theta) = theta(b) - theta(b_obs).

        The option is provided to also perturb the vector e, but note that the
        resulting injected error will be systematically biased lower.  For small
        perturbations (dtheta < 0.2 rad, arccos(dot(e, e + de)) < 0.1 rad), the
        bias should be no greater than 20%.
    """
    @staticmethod
    def addnoise(
        key: int,
        b: jnp.ndarray,
        sigma_dtheta: float
    ) -> jnp.ndarray:
        """ Add noise to quaternion set b.  Only a principal rotation
            perturbation dtheta ~ Norm(0, sigma_angle) is used to corrupt b.

            sigma_dtheta should be no greater than 0.1 radians, with less being
            safer.

        Args:
            key (int): Random key.
            v (jnp.ndarray): 4x1 matrix, 'true' quaternion set b.
            sigma_dtheta (float): Standard deviation for the principal rotation
                angle PDF.

        Returns:
            jnp.ndarray: 4x1 matrix, noise-corrupted quaternion set b.
        """
        # Get principal rotation axis from b.
        e_b = Quaternion(b).get_PRV_from_b()[1]

        # Build perturbing quaternion.
        key, subkey = split(key)
        dtheta = QuatNoise.dx_pdf(subkey, sigma_dtheta)
        b_err = quat_expm_angleaxis(dtheta, e_b)

        # Compose input b with b_err
        return compose_quat(b, b_err)

    @staticmethod
    def addnoise_perturbaxis(
        key: int,
        b: jnp.ndarray,
        sigma_dtheta: float,
        sigma_dphi: float
    ) -> jnp.ndarray:
        """ Add noise to quaternion set b.  Both a principal rotation
            perturbation dtheta ~ Norm(0, sigma_dtheta) and a principal rotation
            axis perturbation dphi ~ Norm(0, sigma_dphi) are used to corrupt b.

            sigma_dtheta should be no greater than 0.1 radians, with less being
            safer. sigma_dphi should be less, at around 0.05 radians.

        Args:
            key (int): Random key.
            v (jnp.ndarray): 4x1 matrix, 'true' quaternion set b.
            sigma_dtheta (float): Standard deviation for the principal rotation
                angle PDF.
            sigma_dphi (float): Standard deviation for the principal rotation
                axis perturbing angle PDF.

        Returns:
            jnp.ndarray: 4x1 matrix, noise-corrupted quaternion set b.
        """
        # Get principal rotation axis from b.
        e_b = Quaternion(b).get_PRV_from_b()[1]

        # Get theta and perpendicular rotation axis choice.  Note that this
        # theta is totally different than 'dtheta', instead being a uniformly
        # sampled angle for a vector residing in a plane perpendicular to e_b.
        key, subkey = split(key)
        theta = theta_pdf(subkey)
        u, w = get_perp_vectors(e_b)
        e_b_perp = combination(u, w, theta)

        # Build perturbed rotation axis.
        key, subkey = split(key)
        dphi = QuatNoise.dx_pdf(subkey, sigma_dphi)
        de_dphi = combination(e_b, e_b_perp, dphi)

        # Build perturbing quaternion.
        key, subkey = split(key)
        dtheta = QuatNoise.dx_pdf(subkey, sigma_dtheta)
        b_err = quat_expm_angleaxis(dtheta, de_dphi)

        # Compose input b with b_err
        return compose_quat(b, b_err)

    @staticmethod
    @jit
    def dx_pdf(
        key: int,
        sigma: float
    ):
        """ PDF that samples a normal distribution centered at zero with
            standard deviation sigma.

        Args:
            key (int): Random key.
            sigma (float): Standard deviation of dx_pdf in radians.

        Returns:
            float: Sampled dx.
        """
        return normal(key) * sigma
