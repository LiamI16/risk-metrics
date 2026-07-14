"""Closest-point-of-approach risk metrics for two vessels.

    DCPA - Distance at Closest Point of Approach
    TCPA - Time to Closest Point of Approach
    vo   - the velocity obstacle for the encounter (own vs target)
"""

import numpy as np

from minkowski_utils import Circle
from .ship import Ship
from .obstacle import VelocityObstacle


class Metrics:
    def __init__(self, ownship, targetship):
        if not isinstance(ownship, Ship) or not isinstance(targetship, Ship):
            raise TypeError("ownship and targetship must be Ship instances")
        self.ownship = ownship
        self.targetship = targetship

        # Relative quantities, target measured w.r.t. own ship.
        self.relpos = targetship.pos - ownship.pos      # line of sight
        self.relvel = targetship.vel - ownship.vel      # closing velocity

    def _require_circular(self):
        """DCPA/TCPA reduce ships to points with a scalar clearance, which only
        holds for disc domains.  For arbitrary convex shapes use the VO instead."""
        for role, ship in (("ownship", self.ownship), ("targetship", self.targetship)):
            if not isinstance(ship.domain, Circle):
                raise TypeError(
                    f"{role} has a {ship.domain} domain; DCPA/TCPA assume disc "
                    "footprints. Use the velocity obstacle (Metrics.vo) for "
                    "arbitrary convex shapes.")

    @property
    def safety_radius(self):
        """Combined disc clearance (own + target radii). Circular domains only."""
        self._require_circular()
        return self.ownship.domain.r + self.targetship.domain.r

    def TCPA(self):
        """Time to closest point of approach (s). Negative => already past it."""
        self._require_circular()
        speed_sq = float(np.dot(self.relvel, self.relvel))
        if speed_sq < 1e-12:              # parallel, no relative motion
            return 0.0
        return -float(np.dot(self.relpos, self.relvel)) / speed_sq

    def DCPA(self):
        """Distance at closest point of approach (m)."""
        self._require_circular()
        speed_sq = float(np.dot(self.relvel, self.relvel))
        if speed_sq < 1e-12:
            return float(np.linalg.norm(self.relpos))
        t = self.TCPA()
        return float(np.linalg.norm(self.relpos + self.relvel * t))

    def vo(self, n_grid=1440):
        """Velocity obstacle for this encounter (own ship vs target)."""
        return VelocityObstacle.from_ships(self.ownship, self.targetship, n_grid=n_grid)
