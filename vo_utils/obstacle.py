"""Velocity obstacle: the collision cone translated to the target velocity.

    VO(O, v_target) = { v_own : v_own - v_target  in  CollisionCone(O) }

i.e. the set of own-ship velocities on a collision course with a constant-
velocity target whose config-space obstacle is O.  The cone geometry is the
same as the collision cone; only the apex moves from the origin to v_target.
"""

import numpy as np

from minkowski_utils import Circle, MinkowskiSum
from .cone import collision_cone


def config_space_obstacle(own_domain, target_domain, relpos):
    """Config-space obstacle ``O = D_target (+) (-D_own)``, placed at ``relpos``.

    Parameters
    ----------
    own_domain : Shape
        Own-ship convex domain (reflected into O).
    target_domain : Shape
        Target-ship convex domain.
    relpos : array_like, shape (2,)
        Target position relative to own, where O is placed.

    Returns
    -------
    MinkowskiSum
    """
    return MinkowskiSum(target_domain, own_domain.reflect(),
                        Circle(0.0, center=np.asarray(relpos, dtype=float)))


class VelocityObstacle:
    def __init__(self, O, v_target, n_grid=1440):
        """
        Parameters
        ----------
        O : Shape
            Config-space obstacle (relative-position space).
        v_target : array_like, shape (2,)
            Target velocity (the cone apex).
        n_grid : int
            Angular resolution of the cone tangent search.
        """
        self.apex = np.asarray(v_target, dtype=float)   # cone apex = target velocity
        self.cone = collision_cone(O, n_grid=n_grid)    # relative-velocity geometry

    @classmethod
    def from_ships(cls, own, target, n_grid=1440):
        """Build the VO for an encounter from two ships (own vs target).

        Parameters
        ----------
        own, target : Ship
            The two ships.
        n_grid : int
            Angular resolution of the cone tangent search.

        Returns
        -------
        VelocityObstacle
        """
        O = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
        return cls(O, target.vel, n_grid=n_grid)

    @property
    def edges(self):
        """The two cone edge ray directions (unit, relative-velocity space)."""
        return self.cone.edges

    def contains(self, v_own):
        """True if own velocity ``v_own`` is on a collision course.

        Collides iff the relative velocity ``w = v_own - v_target`` lies inside the
        cone, i.e. ``w`` is within the half-angle of the bisector ``b = normalize(e1 + e2)``.

        Parameters
        ----------
        v_own : array_like, shape (2,)
            Candidate own velocity.

        Returns
        -------
        bool
        """
        if self.cone.contains_origin:
            return True                                 # every velocity unsafe

        w = np.asarray(v_own, dtype=float) - self.apex
        wn = np.linalg.norm(w)
        if wn < 1e-12:
            return False                                # no relative motion

        e1, e2 = self.cone.edges
        b = e1 + e2
        b = b / np.linalg.norm(b)                       # cone bisector
        return bool((w / wn) @ b >= e1 @ b - 1e-9)
