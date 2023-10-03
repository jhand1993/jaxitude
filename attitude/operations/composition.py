import jax.numpy as jnp


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


def compose_mrp(s_p: jnp.ndarray, s_pp: jnp.ndarray, tol=1e-2) -> jnp.ndarray:
    """ Compose MRP parameters direclty in the following order:
        R(s) = R(s_pp)R(s_p) for MRP rotation matrix R. If the denominator 
        is less than tol value, either s_p or s_pp are transformed to
        their shadow set (based on which is larger: |s_p| or |s_pp|).  

    Args:
        s_p (jnp.ndarray): First MRP parameters as 1x3 matrix.
        s_pp (jnp.ndarray): Second MRP parameters as 1x3 matrix.
        tol (float): Tolerance for shadow switch.

    Returns:
        jnp.ndarray: composed MRP parameters as 1x3 matrix.
    """
    dot_p = jnp.dot(s_p, s_p)
    dot_pp = jnp.dot(s_pp, s_pp)
    return ((1. - dot_pp) * s_p + (1. - dot_p) * s_pp - 2. * jnp.cross(s_pp, s_p)) /\
              (1. + dot_p * dot_pp - 2. * jnp.dot(s_p, s_pp))


def relative_mrp(s: jnp.ndarray, s_p: jnp.ndarray, tol=1e-2) -> jnp.ndarray:
    """ Compose MRP parameters directly in the following order:
        R(s_pp) = R(s_p)R(s)^-1 for MRP rotation matrix R. 
    Args:
        s (jnp.ndarray): First MRP parameters as 1x3 matrix.
        s_p (jnp.ndarray): Second MRP parameters as 1x3 matrix.
        tol (float): Tolerance for shadow switch.


    Returns:
        jnp.ndarray: Relative MRP parameters as 1x3 matrix.
    """
    return compose_mrp(-s_p, s, tol=tol)


def shadow_s(s: jnp.ndarray) -> jnp.ndarray:
    """ Return shadow set of s. 

    Args:
        s (jnp.ndarray): 1x3 matrix of MRP s parameters.

    Returns:
        jnp.ndarray: 1x3 matrix of s shadow parameters.
    """
    return -s / jnp.dot(s, s)