"""Velocity-uncertainty-inflated velocity obstacles.

We inflate the collision cone with the relative-velocity covariance,

    w ~ N(w_hat, Sigma_v),   Sigma_v = Sigma_own + Sigma_target   (independent errors),

which carries both ships' velocity uncertainty. Minkowski sum the
fixed k-sigma ellipse E_v with the collision cone,

    CC (+) E_v.

The apex stays at v_target: the target's velocity uncertainty is incorporated into Sigma_v.
"""

import numpy as np
from dataclasses import dataclass

from minkowski_utils import Ellipse, MinkowskiSum
from .cone import _dir
from .obstacle import VelocityObstacle
from .uncertainty import covariance_ellipse


def _outer_arc(a_from, a_to, a_through, n):
    """Angles from a_from to a_to along the arc that passes through a_through."""
    span = (a_to - a_from) % (2 * np.pi)
    if (a_through - a_from) % (2 * np.pi) <= span:
        return np.linspace(a_from, a_from + span, n)              # CCW
    return np.linspace(a_from, a_from - (2 * np.pi - span), n)    # CW


@dataclass
class RobustVelocityObstacle:
    apex: np.ndarray                     # cone apex = target velocity
    k: float                             # confidence level (number of sigmas)
    ellipse: Ellipse                     # E_v, placed at the apex (for plotting)
    contains_origin: bool
    contacts: np.ndarray | None = None   # (2,2) tangent points p1,p2 where edges meet E_v
    edges: np.ndarray | None = None      # (2,2) unit edge rays e1,e2 (unchanged by inflation)
    cap: np.ndarray | None = None        # (n_cap,2) rounded-apex arc, from p1 to p2


def robust_velocity_obstacle(vo, Sigma_v, k=2.0, n_cap=64):
    """Inflate a VelocityObstacle by the k-sigma relative-velocity ellipse E_v.

    Sigma_v is the relative-velocity covariance (Sigma_own + Sigma_target), so a
    single ellipse accounts for both ships' velocity uncertainty.

    Returns a RobustVelocityObstacle holding the offset-edge tangent points, the
    unchanged edge directions and the rounded-apex arc -- all in velocity space.
    If the underlying cone contains the origin the geometry fields are None
    (every velocity is already unsafe).
    """
    a = np.asarray(vo.apex, dtype=float)
    E0 = covariance_ellipse(Sigma_v, k)                  # centered at the origin
    E = Ellipse(E0.a, E0.b, center=a, theta=E0.theta)    # placed at the apex

    if vo.cone.contains_origin:
        return RobustVelocityObstacle(a, k, E, contains_origin=True)

    (n1, n2), (e1, e2) = vo.cone.normals, vo.cone.edges
    contacts = np.array([a + E0.support_point(n1), a + E0.support_point(n2)])

    b = e1 + e2
    b = b / np.linalg.norm(b)                            # bisector, points into the cone
    angs = _outer_arc(np.arctan2(n1[1], n1[0]),
                      np.arctan2(n2[1], n2[0]),
                      np.arctan2(-b[1], -b[0]), n_cap)    # arc over the apex-outward side
    cap = np.array([a + E0.support_point(_dir(t)) for t in angs])

    return RobustVelocityObstacle(a, k, E, False, contacts,
                                  np.array([e1, e2]), cap)


def uncertain_velocity_obstacle(O, v_target, Sigma_pos, Sigma_vel,
                                k_pos=2.0, k_vel=2.0, n_grid=1440):
    """Velocity obstacle under both position and velocity uncertainty.
    """
    O_k = MinkowskiSum(O, covariance_ellipse(Sigma_pos, k_pos))
    vo = VelocityObstacle(O_k, v_target, n_grid=n_grid)
    return robust_velocity_obstacle(vo, Sigma_vel, k_vel)
