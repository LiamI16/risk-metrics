"""Probabilistic (position-uncertainty) collision cones.

For Gaussian relative position ``pos_rel ~ N(mu, Sigma)``, the confidence-level-k
cone is the deterministic cone of the obstacle inflated by its k-sigma ellipse:

    O_k = O (+) k * E_Sigma,     h_{O_k}(d) = h_O(d) + k * sqrt(d^T Sigma d).

Nested cones over increasing k are level sets of the collision-probability field.
"""

import numpy as np
from dataclasses import dataclass

from minkowski_utils import Ellipse, MinkowskiSum
from .cone import collision_cone, CollisionCone


def covariance_ellipse(Sigma, k=1.0):
    """The k-sigma confidence ellipse of a 2-D Gaussian covariance.

    Parameters
    ----------
    Sigma : array_like, shape (2, 2)
        Covariance matrix.
    k : float
        Confidence level (number of standard deviations).

    Returns
    -------
    Ellipse
        Centered at the origin, with support ``h(d) = k * sqrt(d^T Sigma d)``.
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

    For each ``k``, inflate ``O`` by its k-sigma covariance ellipse and take the
    collision cone -- the level sets of the collision-probability field.

    Parameters
    ----------
    O : Shape
        Convex obstacle in relative-position space.
    Sigma : array_like, shape (2, 2)
        Relative-position covariance.
    k_levels : sequence of float
        Confidence levels (sigmas), one per output cone.

    Returns
    -------
    list of SigmaCone
    """

    return [SigmaCone(k, collision_cone(MinkowskiSum(O, covariance_ellipse(Sigma, k)))) for k in k_levels]
