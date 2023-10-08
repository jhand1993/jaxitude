""" QUEST Method implementation
"""
from jax import grad
from jax.lax import while_loop
import jax.numpy as jnp
from attitude.determination.davenport import get_B
from attitude.primitives import MiscUtil

def K_eig_eq(x: float, K: jnp.ndarray) -> float:
    """ Eigenvalue equation for K matrix.  To be optimized with 
        respect to lam.

    Args:
        x (float): eigenvalue.
        K (jnp.ndarray): K matrix from Davenport's method.

    Returns:
        float: Value of eigenvalue polynomial equation. 
    """
    return jnp.linalg.det(K - x * jnp.eye(K.shape[0]))


def get_lam(
    w: jnp.ndarray,
    K: jnp.ndarray,
    e: float=1e-15
) -> float:
    """ Use simple Newton-Raphson method to iterate for lam with initial guess 
        equal to the sum of the weights w. Function should be jitable. 

    Args:
        w (jnp.ndarray): N matrix with weights.
        K (jnp.ndarray): K matrix from Davenport's method
        e (float, optional): Desired precision. Defaults to 1e-15.

    Returns:
        float: Largest eigenvalue of K.
    """
    lam0 = jnp.array([jnp.sum(w), jnp.sum(w) + 1.])
    grad_K_eig_eq  = grad(K_eig_eq)
    body_func = lambda x: jnp.array([x[0] - K_eig_eq(x[0], K) / grad_K_eig_eq(x[0], K), x[0]]) 
    cond_func = lambda x: jnp.abs(x[0] - x[1]) > e
    return while_loop(cond_func, body_func, lam0)[0]


def get_CRPq(
    w: jnp.ndarray,
    v_b_set: jnp.ndarray,
    v_n_set: jnp.ndarray,
    e: float=1e-15
) -> jnp.ndarray:
    """ Get CRP q parameters from sensor heading and weights. 

    Args:
        w (jnp.ndarray): N matrix with weights.
        v_b_set (jnp.ndarray): Nx3 matrix of N body frame headings from each sensor.
        v_n_set (jnp.ndarray): Nx3 matrix of N inertial frame headings from each sensor.
        e (float, optional): Desired precision. Defaults to 1e-16.

    Returns:
        jnp.ndarray: 1x3 q parameter matrix.
    """
    B = get_B(w, v_b_set, v_n_set)
    S = B + B.T
    sigma = jnp.trace(B)
    Z = jnp.expand_dims(MiscUtil.antisym_dcm_vector(B), axis=-1)
    K = jnp.block([[sigma, Z.T], [Z, S - jnp.eye(3) * sigma]])
    lam = get_lam(w, K, e=e)
    return jnp.matmul(jnp.linalg.inv((lam + sigma) * jnp.eye(3) - S), Z).flatten()


def get_Quatb(
    w: jnp.ndarray,
    v_b_set: jnp.ndarray,
    v_n_set: jnp.ndarray,
    e: float=1e-15
) -> jnp.ndarray:
    """ Get Euler parameters b from sensor heading and weights. 

    Args:
        w (jnp.ndarray): N matrix with weights.
        v_b_set (jnp.ndarray): Nx3 matrix of N body frame headings from each sensor.
        v_n_set (jnp.ndarray): Nx3 matrix of N inertial frame headings from each sensor.
        e (float, optional): Desired precision. Defaults to 1e-16.

    Returns:
        jnp.ndarray: 1x4 b parameter matrix.
    """
    q = get_CRPq(w, v_b_set, v_n_set, e=e)
    q2 = jnp.dot(q, q)
    return jnp.array(1., q[0], q[1], q[2]) / jnp.sqrt(1. + q2)