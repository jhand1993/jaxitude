import jax.numpy as jnp
from jax import jit

@jit
def compose_quat(b_p: jnp.ndarray, b_pp: jnp.ndarray) -> jnp.ndarray:
    """ Adds Euler parameters directly via matrix multiplication:
        Q(b) = Q(b_pp)Q(b_p) for quaternion rotation matrix Q. 

    Args:
        b_p (jnp.ndarray): First rotation parameters 1x4 matrix.
        b_pp (jnp.ndarray): second rotation parameters 1x4 matrix.

    Returns:
        jnp.ndarray: Addition of rotation parameters as 1x4 matrix.
    """
    matrix = jnp.array(
        [[b_pp[0], -b_pp[1], -b_pp[2], -b_pp[3]],
         [b_pp[1], b_pp[0], b_pp[3], -b_pp[2]],
         [b_pp[2], -b_pp[3], b_pp[0], b_pp[1]],
         [b_pp[3], b_pp[2], -b_pp[1], b_pp[0]]]
    )
    return jnp.matmul(matrix, b_p.reshape((4, 1))).flatten()

@jit
def compose_crp(q_p: jnp.ndarray, q_pp: jnp.ndarray) -> jnp.ndarray:
    """ Compose CRP parameters directly in the following order:
        R(q) = R(q_pp)R(q_p) for CRP rotation matrix R. 

    Args:
        q_p (jnp.ndarray): First CRP parameters as 1x3 matrix.
        q_pp (jnp.ndarray): second CRP parameters as 1x3 matrix.

    Returns:
        jnp.ndarray: Composed CRP parameters as 1x3 matrix.
    """
    num = jnp.subtract(jnp.add(q_pp, q_p), jnp.cross(q_pp, q_p))
    denom = 1. - jnp.dot(q_p, q_pp)
    return num / denom

def relative_crp(q: jnp.ndarray, q_p: jnp.ndarray) -> jnp.ndarray:
    """ Compose CRP parameters directly in the following order:
        R(q_pp) = R(q_p)R(q)^-1 for CRP rotation matrix R. 
    Args:
        q (jnp.ndarray): First CRP parameters as 1x3 matrix.
        q_p (jnp.ndarray): second CRP parameters as 1x3 matrix.

    Returns:
        jnp.ndarray: Relative CRP parameters as 1x3 matrix.
    """
    return compose_crp(-q_p, q)