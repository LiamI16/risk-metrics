"""Vessel model for velocity-obstacle scenarios.

A Ship has a 2-D position, constant velocity, and a convex ``domain`` (footprint /
safety region) as a minkowski_utils Shape.
"""

import numpy as np

from minkowski_utils import Circle


class Ship:
    def __init__(self, pos, vel, domain=None):
        """
        Parameters
        ----------
        pos : array_like, shape (2,)
            Position.
        vel : array_like, shape (2,)
            Constant velocity.
        domain : Shape, optional
            Convex footprint / safety region. Defaults to a radius-10 disc.
        """
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.domain = domain if domain is not None else Circle(10.0, center=(0.0, 0.0))

    def position_at(self, t):
        """Position after elapsed time ``t`` under constant velocity (non-mutating)."""
        return self.pos + self.vel * t

    def step(self, t):
        """Advance the position by elapsed time ``t`` (mutating)."""
        self.pos = self.pos + self.vel * t

    def __repr__(self):
        return f"Ship({self.domain}, pos={self.pos.tolist()}, vel={self.vel.tolist()})"
