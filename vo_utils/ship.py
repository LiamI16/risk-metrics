"""Vessel model for velocity-obstacle scenarios.

A Ship has a 2-D position and (constant) velocity, plus a convex *domain* -- its
footprint / safety region as a minkowski_utils Shape.  The domain defaults to a
disc of the given radius but may be any convex shape (e.g. an oriented Ellipse
ship domain), which is what the generalized VO consumes.
"""

import numpy as np

from minkowski_utils import Circle


class Ship:
    def __init__(self, pos, vel, radius=10.0, name="ship", domain=None):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.radius = float(radius)
        self.name = name
        self.domain = domain if domain is not None else Circle(radius, center=(0.0, 0.0))

    def position_at(self, t):
        """Position after time t under constant velocity (non-mutating)."""
        return self.pos + self.vel * t

    def step(self, t):
        self.pos = self.pos + self.vel * t

    def __repr__(self):
        return f"Ship(name={self.name!r}, pos={self.pos.tolist()}, vel={self.vel.tolist()})"
