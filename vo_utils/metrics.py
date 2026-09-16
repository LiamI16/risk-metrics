"""Closest-point-of-approach risk metrics for two vessels (DCPA, TCPA, VO)."""

import numpy as np

from minkowski_utils import Circle
from .ship import Ship
from .obstacle import VelocityObstacle


def cpa_fields(relpos, relvel):
    """Vectorized TCPA and *signed* DCPA -- the array counterpart of :class:`Metrics`.

    ``relpos`` and ``relvel`` need only broadcast against each other to (..., 2), so this
    serves a cloud of realizations, a grid of candidate velocities, or one pair.

    Returns
    -------
    tcpa, dcpa_s : ndarray
        Degenerate ``relvel = 0`` takes the zero-speed branch of the definitions:
        ``TCPA = 0`` and ``DCPA = ||relpos||``.
    """
    speed_sq = np.einsum("...i,...i->...", relvel, relvel)
    moving = speed_sq > 1e-15
    safe = np.where(moving, speed_sq, 1.0)
    tcpa = np.where(moving, -np.einsum("...i,...i->...", relpos, relvel) / safe, 0.0)
    cross = relvel[..., 0] * relpos[..., 1] - relvel[..., 1] * relpos[..., 0]
    dcpa_s = np.where(moving, cross / np.sqrt(safe), np.linalg.norm(relpos, axis=-1))
    return tcpa, dcpa_s


def collides(relpos, relvel, R):
    """Is there a ``t >= 0`` with ``||relpos + relvel t|| <= R``? Vectorized, exact.

    Solves the quadratic ``a t^2 + b t + c <= 0`` directly rather than going through
    TCPA/DCPA, so anything comparing this against the CPA characterization of Theorem 1
    is testing that theorem rather than assuming it. Since ``a = ||relvel||^2 >= 0`` the
    parabola dips below zero exactly between its roots, so a non-negative solution exists
    iff the roots are real and the larger one is non-negative. Degenerate ``relvel = 0``
    reduces to ``||relpos|| <= R``.
    """
    a = np.einsum("...i,...i->...", relvel, relvel)
    b = 2.0 * np.einsum("...i,...i->...", relpos, relvel)
    c = np.einsum("...i,...i->...", relpos, relpos) - R ** 2
    moving = a > 1e-15
    disc = b ** 2 - 4.0 * a * c
    t_hi = (-b + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * np.where(moving, a, 1.0))
    return np.where(moving, (disc >= 0.0) & (t_hi >= 0.0), c <= 0.0)


class Metrics:
    def __init__(self, ownship, targetship):
        """
        Parameters
        ----------
        ownship : Ship
            Own ship.
        targetship : Ship
            Target ship, measured relative to own.
        """
        if not isinstance(ownship, Ship) or not isinstance(targetship, Ship):
            raise TypeError("ownship and targetship must be Ship instances")
        self.ownship = ownship
        self.targetship = targetship

        # Relative quantities, target measured w.r.t. own ship.
        self.relpos = targetship.pos - ownship.pos      # line of sight
        self.relvel = targetship.vel - ownship.vel      # closing velocity

    def _require_circular(self):
        """Require disc domains; DCPA/TCPA assume point ships with scalar clearance."""
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
        """Signed distance at closest point of approach (m).
        """
        self._require_circular()
        r, v = self.relpos, self.relvel
        speed_sq = float(np.dot(v, v))
        if speed_sq < 1e-12:
            return float(np.linalg.norm(r))
        cross = float(v[0] * r[1] - v[1] * r[0])         # v x r (signed)
        return cross / float(np.sqrt(speed_sq))

    def vo(self, n_grid=1440):
        """Velocity obstacle for this encounter (own ship vs target).

        Parameters
        ----------
        n_grid : int
            Angular resolution of the cone tangent search.

        Returns
        -------
        VelocityObstacle
        """
        return VelocityObstacle.from_ships(self.ownship, self.targetship, n_grid=n_grid)
