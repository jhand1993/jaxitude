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
from typing import Tuple, Callable

import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import expm

from jaxitude.rodrigues import shadow, evolve_MRP
from jaxitude.quaternions import xi_op, evolve_quat
from jaxitude.base import colvec_cross
from jaxitude.operations.evolution import evolve_P_ricatti2, evolve_P_ricatti
from jaxitude.operations.integrator import autonomous_euler, autonomous_rk4
from jaxitude.operations.integrator import quat_integrator
from jaxitude.operations.linearization import tangent


@jit
def calculate_K(
    P_new: jnp.ndarray,
    R: jnp.ndarray,
    H: jnp.ndarray
) -> jnp.ndarray:
    """ Calculate the Kalman gain matrix K.

    Args:
        P_new (jnp.ndarray): NxN matrix, predicted state covariance.
        R (jnp.ndarray): MxM matrix, measurement covariance.
        H (jnp.ndarray): MxN matrix, measurement model matrix.

    Returns:
        jnp.ndarray: NxM matrix, Kalman gain matrix
    """
    return P_new @ H.T @ jnp.linalg.inv(
        H @ P_new @ H.T + R
    )


@jit
def joseph_P(
    P: jnp.ndarray,
    K: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray
) -> jnp.ndarray:
    """ Update step for state covariance estimate P using Joseph form.

    Args:
        P (jnp.ndarray): NxN matrix, state covariance estimate.
        K (jnp.ndarray): NxM matrix, Kalman gain matrix.
        H (jnp.ndarray): MxN matrix, measurement model matrix.
        R (jnp.ndarray): MxM matrix, measurement covariance.

    Returns:
        jnp.ndarray: NxN matrix, updated state covariance estimate.
    """
    # Get state vector dimension.
    n = P.shape[0]
    return (jnp.eye(n) - K @ H) @ P @ (jnp.eye(n) - K @ H).T + K @ R @ K.T


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
            Tuple[jnp.ndarray]: 6x1 matrix, 6x6 matrix, updated state vector and
                state covariance matrix estimates.
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
        P_new = joseph_P(P_post, K, H, R_s)

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
                evolve_MRP(x[:3, :], w - x[3:, :]),  # bias correction.
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
                -evolve_MRP(x[:3, :], eta[:3, :]),
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
            evolve_P_ricatti2,
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


class MEKF:
    """ Container class for Multiplicative extended kalman filter (MEKF)
        calculations.

        The dynamics model here is assumed be dxdt = f(x|w) + g(x,eta), with
        f(x|w) describing the system's kinematics and g(x,eta) describing the
        noise evolution.  The underlying process is time-continuous and
        measurements are time-discrete.

        The variable name convention is:

        x_aug: 10x1 matrix, state vector with components [a, b, bias], where a
            are the local attitude error parameters.
        P: 6x6 matrix, state covariance matrix estimate.
        w_obs: 3x1 matrix, observed attitude rates.
        b_obs: 4x1 matrix, observed quaternion set.
        R_w: 3x3 matrix, attitude rate measurement covariance.
        R_q: 4x4 matrix, MRP measurement covariance.
        Q_i: 6x6 matrix, internal process noise covariance.
        Q_a: 6x6 matrix, additive process noise covariance.
        dt: time interval of filter step.
    """
    # Define the measurement matrix.
    H = jnp.hstack([jnp.eye(4), jnp.zeros((4, 3))])

    @staticmethod
    def f(
        x_est: jnp.ndarray,
        w: jnp.ndarray,
    ) -> jnp.ndarray:
        """ Dynamical system equations relating current state and measured rates
            w to dx/dt.

        Args:
            x_est (jnp.ndarray): 7x1 matrix, state vector estimate.
            w (jnp.ndarray): 3x1 matrix, measured rate vector.

        Returns:
            jnp.ndarray: 4x1 matrix, quaternion component of dx/dt.
        """
        return evolve_quat(x_est[:4, :], w - x_est[4:, :])

    @staticmethod
    def f_a(
        x_est: jnp.ndarray,
        w: jnp.ndarray,
    ) -> jnp.ndarray:
        """ Dynamical system equations for propagating covariance matrix.

        Args:
            x_est (jnp.ndarray): 6x1 matrix, state vector estimate.
            w (jnp.ndarray): 3x1 matrix, measured rate vector.

        Returns:
            jnp.ndarray: 6x1 matrix, dx/dt.
        """
        return jnp.vstack(
            [
                colvec_cross(x_est[:3, :], w - x_est[3:, :]),
                jnp.zeros((3, 1))
            ]
        )

    @staticmethod
    def g_a(
        x_est: jnp.ndarray,
        eta: jnp.ndarray,
    ) -> jnp.ndarray:
        """ System noise equations calculated from state and noise vector.

        Args:
            x_est (jnp.ndarray): 6x1 matrix, state vector estimate.
            eta (jnp.ndarray): 6x1 matrix, noise vector.

        Returns:
            jnp.ndarray: 6x1 matrix, noise contribution to dx/dt.
        """
        return jnp.vstack(
            [
                -colvec_cross(x_est[:3, :], eta[:3, :]),
                eta[3:, :]
            ]
        )

    @staticmethod
    def tangent_f_a(
        x_est: jnp.ndarray,
        w_obs: jnp.ndarray
    ) -> jnp.ndarray:
        """ Gets the linearized kinematics equation matrix at x=x_ref:
            F = Jac(f(x, w=w_obs))(x_est).

        Args:
            x_est (jnp.ndarray): 6x1 matrix, state vector estimate to linearize
                at.
            w_obs (jnp.ndarray): 3x1 matrix, measured attitude rate vector.

        Returns:
            jnp.ndarray: 6x6 matrix, linearized kinematics system matrix F.
        """
        # Linearize f(x, w=w_obs) about x_est.
        return tangent(
            lambda x: MEKF.f_a(x, w_obs),
            6, 0, x_est
        )

    @staticmethod
    def tangent_g_a(
        x_est: jnp.ndarray,
    ) -> jnp.ndarray:
        """ Gets the linearized kinematics equation matrix at eta=0:
            G = Jac(g(x=x_est, eta))(eta=0).

        Args:
            x_est (jnp.ndarray): 6x1 matrix, state vector to linearize at.

        Returns:
            jnp.ndarray: 6x6 matrix, linearized noise system matrix G.
        """
        # Linearize g(x=x_est, eta) about eta=0.
        return tangent(
            lambda eta: MEKF.g_a(x_est, eta),
            6, 0, jnp.zeros((6, 1))
        )

    @staticmethod
    def pred_x(
        x_prior: jnp.ndarray,
        w_obs: jnp.ndarray,
        dt: float
    ) -> jnp.ndarray:
        """ Predicts new state vector x_new from x_prior and w_obs along time
            interval dt.

        Args:
            x_prior (jnp.ndarray): 7x1 matrix, prior state vector estimate.
            w_obs (jnp.ndarray): 3x1 matrix, current attitude rate vector
                measurement
            dt (float): time step to integrate.

        Returns:
            jnp.ndarray: 7x1 matrix, quaternion component of state vector.
        """
        return jnp.vstack(
            [
                quat_integrator(
                    x_prior[:4, :],
                    dt,
                    w_obs - x_prior[4:, :]
                ),
                x_prior[4:, :]
            ]
        )

    @staticmethod
    def pred_P_ricatti(
        P_prior: jnp.ndarray,
        F_a_prior: jnp.ndarray,
        G_a_prior: jnp.ndarray,
        Q_a: jnp.ndarray,
        Q_i: jnp.ndarray,
        dt
    ) -> jnp.ndarray:
        """ Predict P_post from x_prior and covariance structures using Ricatti
            equation integrated with the RK4 method.

        Args:
            P_prior (jnp.ndarray): 6x6 matrix, prior P estimate.
            F_a_prior (jnp.ndarray): 6x6 matrix, linearized dynamics model
                evaluated at x_prior and w_obs.
            G_a_prior (jnp.ndarray): 6x6 matrix, linearized noise model
                evaluated at x_prior and w_obs.
            Q_a (jnp.ndarray): 6x6 matrix, additive process noise covariance.
            Q_i (jnp.ndarray): 6x6 matrix, internal process noise covariance.
            dt (float): Integration time interval.

        Returns:
            jnp.ndarray: 6x6 matrix, posterior P estimate from Ricatti equation
                integration.
        """
        return autonomous_rk4(
            evolve_P_ricatti,
            P_prior,
            dt,
            F_a_prior,
            G_a_prior,
            Q_a,
            Q_i
        )

    @staticmethod
    def H_a(b: jnp.ndarray) -> jnp.ndarray:
        """ Generate MEKF measurement matrix from current quaternion estimate b.

        Args:
            b (jnp.ndarray): 4x1 matrix, current quaternion estimate.

        Returns:
            jnp.ndarray: 4x6 matrix, measurement matrix mapping state parameters
                a to measured quaternion.
        """
        return jnp.hstack([0.5 * xi_op(b), jnp.zeros((4, 3))])

    @staticmethod
    def pred_b(
        b_est: jnp.ndarray,
        w_obs: jnp.ndarray,
        dt: float
    ) -> jnp.ndarray:
        """ Predict the global attitude quaternion using quaternion integration.

        Args:
            b_est (jnp.ndarray): 4x1 matrix, prior quaternion estimate.
            w_obs (jnp.ndarray): 3x1 matrix, measured attitude rate.
            dt (float): Integration time interval.

        Returns:
            jnp.ndarray: 4x1 matrix, Propagated global attitude quaternion estimate.
        """
        return quat_integrator(
            b_est,
            dt,
            w_obs
        )

    @staticmethod
    def update_b(
        b_new: jnp.ndarray,
        a_post: jnp.ndarray
    ) -> jnp.ndarray:
        """ Updates the propagated quaternion estimate with the local attitude
            error.

        Args:
            b_est (jnp.ndarray): 4x1 matrix, propagated quaternion estimate.
            b_est (jnp.ndarray): 3x1 matrix, updated state vector estimate.

        Returns:
            jnp.ndarray: 4x1 matrix, updated quaternion estimate.
        """
        b_temp = b_new + 0.5 * xi_op(b_new) @ a_post
        return b_temp / jnp.linalg.norm(b_temp)

    @staticmethod
    def filter_step(
        x_aug_prior: jnp.ndarray,
        P_prior: jnp.ndarray,
        w_obs: jnp.ndarray,
        b_obs: jnp.ndarray,
        R_b: jnp.ndarray,
        Q_a: jnp.ndarray,
        Q_i: jnp.ndarray,
        H: jnp.ndarray,
        H_a: Callable,
        dt: float,
    ) -> Tuple[jnp.ndarray]:
        """ A single MEKF timestep.

        Args:
            x_aug_prior (jnp.ndarray): 10x1 matrix, augmented prior state vector
                estimate.
            P_prior (jnp.ndarray): 6x6 matrix, prior state covariance estimate.
            w_obs (jnp.ndarray): 3x1 matrix, observed attitude rate vector.
            b_obs (jnp.ndarray): 4x1 matrix, observed attitude, represented with
                quaternion b.
            R_b (jnp.ndarray): 4x4 matrix, quaternion measurement covariance.
            Q_a (jnp.ndarray): 6x6 matrix, additive process noise covariance.
            Q_i (jnp.ndarray): 6x6 matrix, internal process noise covariance.
            H (jnp.ndarray):  6x6 matrix, state measurement matrix model.    
            H_a (Callable): Callable that produces a 6x6 matrix, covariance
                measurement matrix model.
            dt (float): Integration time interval.

        Returns:
            Tuple[jnp.ndarray]: 10x1 matrix, 6x6 matrix: updated augmented state
                vector and state covariance estimates.
        """
        # Break up x_aug.
        x_prior = x_aug_prior[3:, :]

        # Reset local attitude error estimates to zero.
        a_prior = jnp.zeros((3, 1))

        # Build x_a vector.
        x_a_prior = jnp.vstack(
            [a_prior, x_prior[4:, :]]
        )

        # Get linearized kinematics and noise matrices.
        F_a_prior = MEKF.tangent_f_a(x_a_prior, w_obs)
        G_a_prior = MEKF.tangent_g_a(x_a_prior)

        # Get posterior predictions for global attitude quaternion, state vector,
        # and state covariance matrix.
        x_new = MEKF.pred_x(x_prior, w_obs, dt)
        P_new = MEKF.pred_P_ricatti(
            P_prior,
            F_a_prior,
            G_a_prior,
            Q_a,
            Q_i,
            dt
        )
        # Get posterior error and Kalman gain using calculated H matrix.
        r_b = b_obs - H @ x_new

        # Calculate the H_a matrix used to calculate K using current b estimate.
        H_a_b_est = H_a(x_new[:4, :])

        K = calculate_K(P_new, R_b, H_a_b_est)

        # Update global attitude quaternion, state vector, and state covariance
        # estimates.
        update = K @ r_b
        a_post = update[:3, :]
        bias_post = x_new[4:, :] + update[3:, :]

        # Use Joseph form for covariance update.
        P_post = joseph_P(P_new, K, H_a_b_est, R_b)

        # Remember to update global attitude estimate.
        b_post = MEKF.update_b(x_new[:4, :], a_post)

        # Build x_aug_post
        x_aug_post = jnp.vstack(
            [a_post, b_post, bias_post]
        )

        return x_aug_post, P_post
