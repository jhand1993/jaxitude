import jax.numpy as jnp
from jax import jit

from jaxitude.base import MiscUtil


@jit
def compose_quat(b_p: jnp.ndarray, b_pp: jnp.ndarray) -> jnp.ndarray:
    """ Adds Euler parameters directly via matrix multiplication:
        Q(b) = Q(b_pp)Q(b_p) for quaternion rotation matrix Q.

    Args:
        b_p (jnp.ndarray): First rotation parameters 4x1 matrix.
        b_pp (jnp.ndarray): second rotation parameters 4x1 matrix.

    Returns:
        jnp.ndarray: Addition of rotation parameters as 4x1 matrix.
    """
    matrix = jnp.array(
        [[b_pp[0, 0], -b_pp[1, 0], -b_pp[2, 0], -b_pp[3, 0]],
         [b_pp[1, 0], b_pp[0, 0], b_pp[3, 0], -b_pp[2, 0]],
         [b_pp[2, 0], -b_pp[3, 0], b_pp[0, 0], b_pp[1, 0]],
         [b_pp[3, 0], b_pp[2, 0], -b_pp[1, 0], b_pp[0, 0]]]
    )
    return jnp.matmul(matrix, b_p)


@jit
def compose_crp(q_p: jnp.ndarray, q_pp: jnp.ndarray) -> jnp.ndarray:
    """ Compose CRP parameters directly in the following order:
        R(q) = R(q_pp)R(q_p) for CRP rotation matrix R.

    Args:
        q_p (jnp.ndarray): First CRP parameters as 3x1 matrix.
        q_pp (jnp.ndarray): second CRP parameters as 3x1 matrix.

    Returns:
        jnp.ndarray: Composed CRP parameters as 3x1 matrix.
    """
    num = jnp.subtract(
        jnp.add(q_pp, q_p),
        MiscUtil.colvec_cross(q_pp, q_p)
    )
    denom = 1. - jnp.vdot(q_p, q_pp)
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


def compose_mrp(s_p: jnp.ndarray, s_pp: jnp.ndarray) -> jnp.ndarray:
    """ Compose MRP parameters direclty in the following order:
        R(s) = R(s_pp)R(s_p) for MRP rotation matrix R.

    Args:
        s_p (jnp.ndarray): First MRP parameters as 1x3 matrix.
        s_pp (jnp.ndarray): Second MRP parameters as 1x3 matrix.

    Returns:
        jnp.ndarray: composed MRP parameters as 1x3 matrix.
    """
    dot_p = jnp.vdot(s_p, s_p)
    dot_pp = jnp.vdot(s_pp, s_pp)
    cross_spp_sp = MiscUtil.colvec_cross(s_pp, s_p)
    return ((1. - dot_pp) * s_p + (1. - dot_p) * s_pp - 2. * cross_spp_sp) /\
        (1. + dot_p * dot_pp - 2. * jnp.vdot(s_p, s_pp))

# NOTE: In Progress.
# def compose_mrp2(
#     s_p: jnp.ndarray,
#     s_pp: jnp.ndarray,
#     tol=1e-2
# ) -> jnp.ndarray:
#     """ Compose MRP parameters direclty in the following order:
#         R(s) = R(s_pp)R(s_p) for MRP rotation matrix R. If the denominator
#         is less than tol value, either s_p or s_pp are transformed to
#         their shadow set (based on which is larger: |s_p| or |s_pp|).

#     Args:
#         s_p (jnp.ndarray): First MRP parameters as 1x3 matrix.
#         s_pp (jnp.ndarray): Second MRP parameters as 1x3 matrix.
#         tol (float): Tolerance for shadow switch.

#     Returns:
#         jnp.ndarray: composed MRP parameters as 1x3 matrix.
#     """
#     dot_p = jnp.dot(s_p, s_p)
#     dot_pp = jnp.dot(s_pp, s_pp)
#     return ((1. - dot_pp) * s_p + (1. - dot_p) * s_pp -
#             - 2. * jnp.cross(s_pp, s_p)) /\
#         (1. + dot_p * dot_pp - 2. * jnp.dot(s_p, s_pp))


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
