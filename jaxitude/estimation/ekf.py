""" Extended Kalman filter algorithms for quaternion sets b and MRP sets s.
    See https://link.springer.com/article/10.1007/BF03321529 for EKF MRP,

"""
import jax.numpy as jnp

from jaxitude.operations import evolution as ev
from jaxitude.operations.linearize import linearize_dynamics
from jaxitude.operations.linearize import linearize_dynamics_control
from jaxitude.operations.integrator import autonomous_rk4


def mrp_ekf(
    t: jnp.ndarray,
    w_obs: jnp.ndarray,
    Lambda: jnp.ndarray,
    Q: jnp.ndarray,
    bias: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    pass


def mrp_ekf_propogate_x(
    x_old: jnp.ndarray,
    w_obs: jnp.ndarray,
    bias: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """ Integrate MRP prediction 

    Args:
        x_old (jnp.ndarray): _description_
        w_obs (jnp.ndarray): _description_
        bias (jnp.ndarray): _description_
        dt (float): _description_

    Returns:
        jnp.ndarray: _description_
    """
    s_old = x_old[:3, :]
    s_pred = autonomous_rk4(ev.evolve_MRP, s_old, dt)
    return jnp.vstack(s_pred, jnp.zeros_like(s_pred))
