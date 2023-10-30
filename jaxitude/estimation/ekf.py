""" Extended Kalman filter algorithms for quaternions b and MRPs s.
    See https://link.springer.com/article/10.1007/BF03321529 for MRP EKF
    algorithm, and https://matthewhampsey.github.io/blog/2020/07/18/mekf for
    an example Multiplicative EKF example (your blog is dope, Matthew).

    Note for the curious: these classes try to break down a single filter step
    into individual calculatable parts, with the resulting modularity being
    used to 'easily' build up the filter step calculations. This is on purpose,
    since it improves code readibility.

    Each class of EKF algorithms here uses its own Kalman gain matrix K
    calculator, since the shape and components are dependent upon the shape and
    properties the state space itself.

"""
from typing import Tuple

import jax.numpy as jnp
from jax.scipy.linalg import expm

from jaxitude.rodrigues import shadow
from jaxitude.operations import evolution as ev
from jaxitude.operations.integrator import autonomous_euler
from jaxitude.operations.linearization import tangent


class MRPEKF(object):
    """ Container class for MRP EKF algorithm functionality.

        The dynamics model here is assumed be dxdt = f(x|w) + g(x,eta), with
        f(x|w) describing the system's kinematics and g(x,eta) describing the
        noise evolution.  The underlying process is time-continuous and
        measurements are time-discrete.  The variable name convention is:

        x: 6x1 matrix, state vector with components [s, bias], where s is the
            MRP set and bias are the gyroscope biases.
        P: 6x6 matrix, state (process) covariance matrix estimate.
        w_obs: 3x1 matrix, observed attitude rates.
        s_obs: 3x1 matrix, observed MRP s set.
        R_w: 3x3 matrix, attitude rate measurement covariance.
        R_s: 3x3 matrix, MRP measurement covariance.
        Q: 6x6 matrix, (process) noise covariance.
        dt: time interval of filter step.
    """

    # The measurement matrix H for this system is 6x3 matrix.
    H: jnp.ndarray = jnp.block([jnp.eye(3), jnp.zeros((3, 3))])

    @staticmethod
    def filter_step(
        x_prior: jnp.ndarray,
        P_prior: jnp.ndarray,
        w_obs: jnp.ndarray,
        s_obs: jnp.ndarray,
        R_w: jnp.ndarray,
        R_s: jnp.ndarray,
        Q: jnp.ndarray,
        dt: float,
        H: jnp.ndarray,
        P_propogation_method: str = 'ricatti'
    ) -> Tuple[jnp.ndarray]:
        """ A single MRP EKF timestep

        Args:
            x_prior (jnp.ndarray): 6x1 matrix, prior state vector estimate.
            P_prior (jnp.ndarray): 6x6 matrix, prior state covariance estimate.
            w_obs (jnp.ndarray): 3x1 matrix, observed attitude rate vector.
            s_obs (jnp.ndarray): 3x1 matrix, observed attitude, represented with
                MRP set s.
            R_w (jnp.ndarray): 3x3 matrix, attitude vector w covariance.
            R_s (jnp.ndarray): 3x3 matrix, MRP s set covariance.
            Q (jnp.ndarray): 6x6 matrix, process noise covariance for attitude
                rates and gyroscopic bias.
            dt (float): Integration time interval.
            H (jnp.ndarray): 3x6 matrix, measurement model matrix.
            P_propogation_method (str): State covariance propogation method.
                Either directely using the system transition matrix ('stm') or
                via numerically integrating the Ricatti ODE ('ricatti').
                Defaults to 'stm'.

        Returns:
            Tuple[jnp.ndarray]: 6x1 matrix, 6x1 matrix, 6x6 matrix, updated
                state vector, noise vector, and process covariance estimates.
        """
        # Select proper alias for P propogation.
        P_prop = {
            'stm': MRPEKF.pred_P_stm,
            'ricatti': MRPEKF.pred_P_ricatti
        }[P_propogation_method.lower()]

        # Get linearized kinematics and noise matrices.
        F_prior = MRPEKF.tangent_f(x_prior, w_obs)
        G_prior = MRPEKF.tangent_g(x_prior)

        # Get posterior predictions for state vector and state covariance
        # matrix.
        x_post = MRPEKF.pred_x(x_prior, w_obs, dt)
        P_post = P_prop(P_prior, F_prior, G_prior, R_w, Q, dt)

        # First shadow set check.
        x_post, P_post = MRPEKF.shadow_propogation_check(x_post, P_post)

        # Get state prediction error and Kalman gain. Note that only the first
        # three components of state vector x_post need to be provided -- those
        # are the posterior prediction MRP s values.  Also, the second shadow
        # check is done within MRPEKF.get_y().
        y = MRPEKF.get_y(s_obs, x_post[:3, :])
        K = MRPEKF.get_K(P_post, R_s)

        # Update state vector and state covariance estimate with Kalman gain.
        x_new = x_post + K @ y
        P_new = (jnp.eye(6) - K @ H) @ P_post

        return x_new, P_new

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
                components are attitude rate noise, second three are gyro bias
                noise.

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
    def tangent_f(
        x_ref: jnp.ndarray,
        w_obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """ Gets the linearized kinematics equation matrix at x=x_ref:
            F = Jac(f(x, w=w_obs))(x_ref).

        Args:
            x_ref (jnp.ndarray): 6x1 matrix, state vector estimate to linearize
                at.
            w_obs (jnp.ndarray): 3x1 matrix, measured attitude rate vector.

        Returns:
            jnp.ndarray: Linearized kinematics system matrix F.
        """
        # Linearize f(x, w=w_obs) about x_ref.
        return tangent(
            lambda x: MRPEKF.f(x, w_obs),
            6, 0, x_ref
        )

    @staticmethod
    def tangent_g(
        x_ref: jnp.ndarray,
    ) -> jnp.ndarray:
        """ Gets the linearized kinematics equation matrix at eta=0:
            G = Jac(g(x=x_ref, eta))(eta=0).

        Args:
            x_ref (jnp.ndarray): 6x1 matrix, state vector to linearize at.

        Returns:
            jnp.ndarray: Linearized noise system matrix G.
        """
        # Linearize g(x=x_ref, eta) about eta=0.
        return tangent(
            lambda eta: MRPEKF.g(x_ref, eta),
            6, 0, jnp.zeros((6, 1))
        )

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
    def pred_P_ricatti(
        P_prior: jnp.ndarray,
        F_prior: jnp.ndarray,
        G_prior: jnp.ndarray,
        Q: jnp.ndarray,
        dt: float
    ) -> jnp.ndarray:
        """ Predict P_post from x_prior and covariance structures using Ricatti
            equation integrated with Euler's method.

        Args:
            P_prior (jnp.ndarray): 6x6 matrix, prior P estimate.
            F_prior (jnp.ndarray): 6x6 matrix, linearized kinematics model
                evaluated at x_prior and w_obs.
            G_prior (jnp.ndarray): 6x6 matrix, linearized noise model evaluated
                at w_obs. Not used, but maintained for API consistency with
                pred_P_stm().
            Q (jnp.ndarray): 6x6 matrix, process noise covariance.
            dt (float): Integration time interval.

        Returns:
            jnp.ndarray: Posterior P estimate from Ricatti equation integration.
        """
        return autonomous_euler(
            ev.evolve_P_ricatti,
            P_prior,
            dt,
            F_prior,
            Q
        )

    @staticmethod
    def pred_P_stm(
        P_prior: jnp.ndarray,
        F_prior: jnp.ndarray,
        G_prior: jnp.ndarray,
        Q: jnp.ndarray,
        dt: float
    ) -> jnp.ndarray:
        """ Predict P_post from x_prior and covariance structures using the
            the state transition matrix, as calculated from the linearized
            dynamics around the current state estimate.

        Args:
            P_prior (jnp.ndarray): 6x6 matrix, prior P estimate.
            F_prior (jnp.ndarray): 6x6 matrix, linearized kinematics model
                evaluated at x_prior and w_obs.
            G_prior (jnp.ndarray): 6x6 matrix, linearized noise model evaluated
                at w_obs.
            Q (jnp.ndarray): 6x6 matrix, process noise covariance.
            dt (float): Integration time interval.

        Returns:
            jnp.ndarray: Posterior P estimate from system transition matrix.
        """
        A = expm(
            dt * jnp.block(
                [[-F_prior, G_prior @ Q @ G_prior.T],
                 [jnp.zeros((6, 6)), F_prior.T]]
            )
        )
        Phi = A[6:, 6:].copy().T
        Q_tilde = Phi @ A[:6, 6:].copy()
        return Phi @ P_prior @ Phi.T + Q_tilde

    @staticmethod
    def get_y(
        s_obs: jnp.ndarray,
        s_post: jnp.ndarray,
    ) -> jnp.ndarray:
        """ Calculate MRP error between predicted and observed states.  This
            Also checks if the magnitude of y is less than 1/3.  If it is
            greater than 1/3, then another check via method
            'shadow_update_check(y, y_shadow)' is performed, where y_shadow is
            the prediction error calculated with the posterior predicted s's
            shadow set.

        Args:
            s_obs (jnp.ndarray): 3x1 matrix, observed MRP s set.
            s_post (jnp.ndarray): 3x1 matrix, predicted MRP s set.

        Returns:
            jnp.ndarray: Error between prediction and observation.
        """
        y = s_obs - s_post

        return jnp.where(
            jnp.linalg.norm(y) < 0.33333,
            y,
            MRPEKF.shadow_update_check(
                y,
                s_obs - MRPEKF.H[:, :3] @ shadow(s_post)
            )
        )

    @staticmethod
    def get_K(
        P_post: jnp.ndarray,
        R_s: jnp.ndarray,
        H: jnp.ndarray
    ) -> jnp.ndarray:
        """ Calculate the Kalman gain matrix K.

        Args:
            P_post (jnp.ndarray): 6x6 matrix, predicted state covariance.
            R_s (jnp.ndarray): 3x3 matrix, observed MRP s covariance.
            H (jnp.ndarray): 3x6 matrix, measurement model matrix.

        Returns:
            jnp.ndarray: 6x3 matrix, Kalman gain that also maps MRP s prediction
                error back to state space.
        """
        return P_post @ H.T @ jnp.linalg.inv(
            H @ P_post @ H.T + R_s
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
            [shadow(x[:3, :]),
             x[3:, :]]
        )

    @staticmethod
    def P_shadow(
        x: jnp.ndarray,
        P: jnp.ndarray
    ) -> jnp.ndarray:
        """ Calculates and returns MRP shadow set state covariance matrix P.

        Args:
            x (jnp.ndarray): 3x1 matrix, state vector x.
            P (jnp.ndarray): 6x6 matrix, state covariance matrix.

        Returns:
            jnp.ndarray: 6x6 matrix, state covariance matrix for MRP shadow set.
        """
        s2_inv = 1. / jnp.vdot(x[:3, :], x[:3, :])
        S_mat = 2. * s2_inv**2. * (x[:3, :] @ x[:3, :].T) - s2_inv

        return jnp.block(
            [[S_mat @ P[:3, :3] @ S_mat.T, S_mat @ P[:3, 3:]],
             [P[3:, :3] @ S_mat.T, P[3:, 3:]]]
        )

    @staticmethod
    def shadow_propogation_check(
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
        # Turns out you can't have tuples as arguments for jnp.where().
        # This is a workaround for now.
        result = jnp.where(
            jnp.linalg.norm(x[:3, :]) < 1.,
            jnp.hstack([x, P]),
            jnp.hstack([MRPEKF.x_shadow(x), MRPEKF.P_shadow(x, P)])
        )
        return result[:, :1], result[:, 1:]

    @staticmethod
    def shadow_update_check(
        y: jnp.ndarray,
        y_shadow: jnp.ndarray
    ) -> jnp.ndarray:
        """ Given the errors between the observed and predicted MRP s and shadow
            s sets:
                {y = s_obs - H @ s_post, y_shadow = s_obs_sahdow - H @ s_post},
            this method returns the error with the smaller norm.

        Args:
            y (jnp.ndarray): 6x1 matrix, MRP vector error.
            y_shadow (jnp.ndarray): 6x1 matrix, MRP vector error calculated
                with the s_obs shadow set.

        Returns:
            jnp.ndarray: Error vector with smaller norm.
        """
        return jnp.where(
            jnp.linalg.norm(y) < jnp.linalg.norm(y_shadow),
            y,
            y_shadow
        )


class MEKF():
    """ Container class for Multiplicative extended kalman filter (MEKF)
        calculations.

        The dynamics model here is assumed be dxdt = f(x|w) + g(x,eta), with
        f(x|w) describing the system's kinematics and g(x,eta) describing the
        noise evolution.  The underlying process is time-continuous and
        measurements are time-discrete.

        Because unit quaternions represent rotations, care needs to be taken
        when predicting quaternion values from observation and past estimates.
        MEKF does this by tracking an error quaternion dq = exp([0, a/2].T),
        where a is a 3-vector with compoents much less than one (small angle
        approximation for the win). Quaternion predictions is done utizling dq.

        The variable name convention is:

        x: 6x1 matrix, state vector with components [a, bias], where a is the
            error quaternion set and bias are the gyroscope biases.
        P: 6x6 matrix, state covariance matrix estimate.
        w_obs: 3x1 matrix, observed attitude rates.
        q_obs: 4x1 matrix, observed quaternion set.
        R_w: 3x3 matrix, attitude rate measurement covariance.
        R_q: 4x4 matrix, MRP measurement covariance.
        Q: 6x6 matrix, (process) noise covariance.
        dt: time interval of filter step.
    """
