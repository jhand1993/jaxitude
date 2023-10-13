""" Triad attitude estimation technique
"""
import jax.numpy as jnp
from jax import jit


def get_triad_frame(s: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
    """ Calculates the triad frame from vectors s and m.
        t1 = s, t2 = s x m / |s x m|, t3 = t1 x t2.

    Args:
        s (jnp.ndarray): First measurement heading vector as 3x1 matrix.
        m (jnp.ndarray): Second measurement heading vector as 3x1 matrix.

    Returns:
        jnp.ndarray: 3x3 stack of triad frame basis vectors:
            [t1.T, t2.T, t3.T].T.  Note that t1, t2, t3 are calculated as row
            vectors and then stacked vertically, hence the transpose.
    """
    # Normalize s
    t1 = s.T[0] / jnp.linalg.norm(s)
    # Also, normalize m.
    t2_raw = jnp.cross(t1, m.T[0] / jnp.linalg.norm(m))
    t2 = t2_raw / jnp.linalg.norm(t2_raw)
    t3_raw = jnp.cross(t1, t2)
    t3 = t3_raw / jnp.linalg.norm(t3_raw)
    return jnp.vstack([t1, t2, t3]).T


@jit
def get_triad_dcm(
    s_b: jnp.ndarray,
    m_b: jnp.ndarray,
    s_n: jnp.ndarray,
    m_n: jnp.ndarray
) -> jnp.ndarray:
    """ Calculates body to inertial frame DCM from heading vectors given in
        body (b) and inertial (n) frames.

    Args:
        s_b (jnp.ndarray): 3x1 matrix first heading measurement in b frame.
        m_b (jnp.ndarray): 3x1 matrix second heading measurement in b frame.
        s_n (jnp.ndarray): 3x1 matrix first heading measurement in n frame.
        m_n (jnp.ndarray): 3x1 matrix second heading measurement in n frame.

    Returns:
        jnp.ndarray: 3x3 array of body to inertial frame DCM.
    """
    BT = get_triad_frame(s_b, m_b)
    NT = get_triad_frame(s_n, m_n)
    return jnp.matmul(BT, NT.T)
