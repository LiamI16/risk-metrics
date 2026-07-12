"""Probabilistic (position-uncertainty) collision cones.

Positional uncertainty on the relative position is Gaussian, relpos ~ N(mu, Sigma).
A collision cone at confidence level k is the deterministic cone of the obstacle
INFLATED by the k-sigma covariance ellipse:

    O_k = O  (+)  k * E_Sigma,     h_{O_k}(d) = h_O(d) + k * sqrt(d^T Sigma d).

Nested cones over increasing k are the level sets of the collision-probability
field -- a thin cone (small k) bounds high-probability collisions, a wide cone
(large k) bounds low-probability ones.  Parametrised by sigma level k, not by a
probability delta; that labelling is deferred (see docs/velocity_obstacles.md).
"""

import numpy as np
from dataclasses import dataclass

from minkowski_utils import Ellipse, MinkowskiSum
from .cone import collision_cone, CollisionCone


def covariance_ellipse(Sigma, k=1.0):
    """The k-sigma confidence ellipse of a 2-D Gaussian covariance Sigma.

    Returns an Ellipse centred at the origin whose support function equals
        h(d) = k * sqrt(d^T Sigma d)
    """

    vals, vecs = np.linalg.eigh(Sigma)
    theta = np.arctan2(vecs[1, 0], vecs[0, 0])
    a, b = k * np.sqrt(vals)

    return Ellipse(a, b, center=(0, 0), theta=theta)


@dataclass
class SigmaCone:
    k: float                     # confidence level (number of sigmas)
    cone: CollisionCone          # collision cone of the k-sigma-inflated obstacle


def probabilistic_collision_cone(O, Sigma, k_levels=(1.0, 2.0, 3.0)):
    """Nested collision cones for a Gaussian-position-uncertain obstacle.

    For each k in k_levels, inflate O by the k-sigma covariance ellipse (a
    Minkowski summand) and take its collision cone.  Returns a list of SigmaCone,
    one per level -- the level sets of the collision-probability field.
    """

    return [SigmaCone(k, collision_cone(MinkowskiSum(O, covariance_ellipse(Sigma, k)))) for k in k_levels]
