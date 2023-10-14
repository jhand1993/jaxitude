""" Extended Kalman filter algorithms for quaternion sets b and MRP sets s.
    See https://link.springer.com/article/10.1007/BF03321529 for EKF MRP,

"""
from typing import Callable, Tuple

import jax.numpy as jnp

from jaxitude.rodrigues import MRP
from jaxitude.operations import evolution as ev
from jaxitude.operations.integrator import autonomous_euler
from jaxitude.operations.linearization import linearize


class MRPEKF(object):
    """ Container class for MRP EKF algorithm functionality.  The dynamics
        model is dxdt = f(x|w) + g(x,eta), with f(x|w) describing the system's
        kinematics and g(x,eta) describing the noise evolution.
    """
    @staticmethod
    def f(
        x: jnp.ndarray,
        w: jnp.ndarray,
    ) -> jnp.ndarray:
        """ Kinematics equation f to evolve MRPs.  Note that vector w is
            corrected for by gyro bias b: w_pred = w - x[3:, :] = w - b.

        Args:
            x (jnp.ndarray): 6x1 matrix, state vector.  First three
                components are MRP values, second three are bias measurements.
            w (jnp.ndarray): 3x1 matrix, measured attitude rate vector.

        Returns:
            jnp.ndarray: Kinematics component of state rate vector matrix
                with shape 6x1.
        """
        return jnp.vstack(
            [
                ev.evolve_MRP(x[:3, :], w - x[3:, :]),  # bias correction.
                jnp.zeros((3, 1))
            ]
        )

    @staticmethod
    def g(
        x: jnp.ndarray,
        eta: jnp.ndarray,
    ) -> jnp.ndarray:
        """ Noise evolution equation g to evolve MRPs.

        Args:
            s (jnp.ndarray): 6x1 matrix, state vector.  First three
                components are MRP values, second three are gyro bias
                measurements.
            eta (jnp.ndarray): 6x1 matrix, noise vectors. First three
                components are MRP noise, second three are gyro bias noise.

        Returns:
            jnp.ndarray: Noise component of state rate vector matrix
                with shape 6x1.
        """
        return jnp.vstack(
            [
                -ev.evolve_MRP(x[:3, :], eta[:3, :]),
                eta[3:, :]
            ]
        )

    @staticmethod
    def linearize_f(
        x_ref: jnp.ndarray,
        w_obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """ Linearizes kinematics equation about x_ref:
            F ~ Jac(f(x, w=w_obs))(x_ref).

        Args:
            x_ref (jnp.ndarray): 6x1 matrix, state vector estimate to linearize
                at.
            w_obs (jnp.ndarray): 3x1 matrix, measured attitude rate vector.

        Returns:
            Callable: Linearized kinematics system matrix F.
        """
        # Linearize f(x, w=w_obs) about x_ref.
        return linearize(
            lambda x: MRPEKF.f(x, w_obs),
            6, 0, x_ref
        )(x_ref)

    @staticmethod
    def linearize_g(
        x_ref: jnp.ndarray,
    ) -> Callable:
        """ Linearizes noise equations about x_ref and eta=0:
            G ~ Jac(g(x=x_ref, eta))(eta=0).

        Args:
            x_ref (jnp.ndarray): 6x1 matrix, state vector to linearize at.

        Returns:
            Callable: Linearized noise system matrix G.
        """
        # Linearize g(x=x_ref, eta) about eta=0.
        return linearize(
            lambda eta: MRPEKF.g(x_ref, eta),
            6, 1, jnp.zeros((6, 1))
        )(jnp.zeros((6, 1)))

    @staticmethod
    def pred_x(
        x_prior: jnp.ndarray,
        w_obs: jnp.ndarray,
        dt: float,
    ) -> jnp.ndarray:
        """ Predicts new state vector x_post from x_prior and w_obs along time
            interval dt.

        Args:
            x_prior (jnp.ndarray): 6x1 matrix, prior state vector estimate.
            w_obs (jnp.ndarray): 6x1 matrix, observed attitude rate vector.
            dt (float): Integration time interval.

        Returns:
            jnp.ndarray: Posterior state vector estimate.
        """
        return autonomous_euler(
            MRPEKF.f,
            x_prior,
            dt,
            w_obs
        )

    @staticmethod
    def pred_P(
        P_prior: jnp.ndarray,
        x_prior: jnp.ndarray,
        w_obs: jnp.ndarray,
        Lambda: jnp.ndarray,
        Q: jnp.ndarray,
        dt: float
    ) -> jnp.ndarray:
        """ Predict P_post from x_prior and covariance structures using Ricatti
            equation integrated with Euler's method.

        Args:
            P_prior (jnp.ndarray): 6x6 matrix, prior P estimate.
            x_prior (jnp.ndarray): 6x1 matrix, prior state vector estimate.
            w_obs (jnp.ndarray): 6x1 matrix, observed attitude rate vector.
            Lambda (jnp.ndarray): 6x6 matrix, MRP measurement covariance.
            Q (jnp.ndarray): 6x6 matrix, process noise covariance.
            dt (float): Integration time interval.

        Returns:
            jnp.ndarray: Posterior P estimate.
        """
        return autonomous_euler(
            ev.evolve_MRP_Pmatrix,
            P_prior,
            dt,
            MRPEKF.linearize_f(x_prior, w_obs),
            MRPEKF.linearize_g(x_prior),
            Lambda,
            Q
        )

    @staticmethod
    def x_shadow(
        x: jnp.ndarray
    ) -> jnp.ndarray:
        """ Calculates and returns MRP shadow set of input state vector x.

        Args:
            x (jnp.ndarray): 6x1 matrix, state vector.

        Returns:
            jnp.ndarray: 6x1 matrix, state vector with MRP shadow set.
        """
        return jnp.vstack(
            MRP.shadow(x[:3, :]),
            jnp.zeros((3, 1))
        )

    @staticmethod
    def P_shadow(
        s: jnp.ndarray,
        P: jnp.ndarray
    ) -> jnp.ndarray:
        """ Calculates and returns MRP shadow set state covariance matrix P.

        Args:
            s (jnp.ndarray): 3x1 matrix, MRP s set from state vector x.
            P (jnp.ndarray): 6x6 matrix, state covariance matrix.

        Returns:
            jnp.ndarray: 6x6 matrix, state covariance matrix for MRP shadow set.
        """
        s2_inv = 1. / (s.T @ s)[0, 0]
        S_mat = 2. * s2_inv**2. * (s @ s.T) - s2_inv
        return jnp.block(
            [[S_mat @ P[:3, :3]] @ S_mat.T, S_mat @ P[:3, 3:]
             [P[3:, :3] @ S_mat.T, P[3:, 3:]]]
        )

    @staticmethod
    def shadow_check(
        x: jnp.ndarray,
        P: jnp.ndarray
    ) -> Tuple[jnp.ndarray]:
        """ Checks magnitude of MRP set from state vector x and swaps to
            corresponding x, P shadow sets if needed.

        Args:
            x (jnp.ndarray): 6x1 matrix, state vector.
            P (jnp.ndarray): 6x6 matrix, state vector covariance.

        Returns:
            Tuple: shadow set versions of x and P.
        """
        return jnp.where(
            jnp.linalg.norm(x[:3, :]) < 1.,
            (x, P),
            (MRPEKF.x_shadow(x), MRPEKF.P_shadow(x, P))
        )
