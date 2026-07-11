"""Velocity obstacle: the collision cone translated to the target velocity.

    VO(O, v_target) = { v_own : v_own - v_target  in  CollisionCone(O) }

i.e. the set of own-ship velocities on a collision course with a constant-
velocity target whose config-space obstacle is O.  The cone geometry is the
same as the collision cone; only the apex moves from the origin to v_target.
"""

import numpy as np

from .cone import collision_cone


class VelocityObstacle:
    def __init__(self, O, v_target, n_grid=1440):
        self.apex = np.asarray(v_target, dtype=float)   # cone apex = target velocity
        self.cone = collision_cone(O, n_grid=n_grid)    # relative-velocity geometry

    @property
    def edges(self):
        """The two cone edge ray directions (unit, relative-velocity space)."""
        return self.cone.edges

    def contains(self, v_own):
        """True if own velocity v_own is on a collision course.

        v_own collides iff the relative velocity w = v_own - v_target lies inside
        the collision cone.  Membership test: the cone bisector
        b = normalize(e1 + e2) points into the convex set, and w is inside when w's
        direction is within the half-angle of b
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
