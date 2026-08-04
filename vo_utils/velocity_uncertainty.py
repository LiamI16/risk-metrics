"""Velocity-uncertainty-inflated velocity obstacles.

Velocity errors are distributions exposing convex confidence bodies
``VelocityUncertainty.body(k) -> Shape``; independent errors combine by
``own + target``. The geometry only sees the resulting Shape K and forms
``VO = v_target (+) CollisionCone (+) K``.
"""

from abc import ABC, abstractmethod

import numpy as np
from dataclasses import dataclass

from minkowski_utils import Shape, Polygon, MinkowskiSum
from .cone import _dir
from .obstacle import VelocityObstacle
from .uncertainty import covariance_ellipse


class VelocityUncertainty(ABC):
    """A velocity-error distribution as a family of convex confidence bodies."""

    @abstractmethod
    def body(self, k: float) -> Shape:
        """Centered convex confidence body at level ``k``.

        Parameters
        ----------
        k : float
            Confidence level; each distribution owns its meaning (e.g. sigmas for
            a Gaussian, support scale for a box).

        Returns
        -------
        Shape
        """

    def __add__(self, other):
        """Combine independent errors: ``own + target`` -> SumUncertainty."""
        if not isinstance(other, VelocityUncertainty):
            return NotImplemented
        return SumUncertainty(self, other)


class SumUncertainty(VelocityUncertainty):
    """Independent errors; ``body(k)`` is the Minkowski sum of parts at a shared level.

    Parameters
    ----------
    *parts : VelocityUncertainty
        The summands; nested sums are flattened.
    """

    def __init__(self, *parts):
        flat = []
        for p in parts:
            flat.extend(p.parts if isinstance(p, SumUncertainty) else [p])
        self.parts = tuple(flat)

    def body(self, k=2.0):
        """Minkowski sum of each part's ``body(k)`` at shared confidence level ``k``."""
        return MinkowskiSum(*(p.body(k) for p in self.parts))

    def __repr__(self):
        return "SumUncertainty(" + ", ".join(repr(p) for p in self.parts) + ")"


@dataclass
class Gaussian(VelocityUncertainty):
    """Gaussian error; ``body(k)`` is the k-sigma covariance ellipse.

    Attributes
    ----------
    Sigma : ndarray, shape (2, 2)
        Velocity-error covariance matrix.
    """

    Sigma: np.ndarray

    def body(self, k=2.0):
        """k-sigma confidence ellipse; ``k`` is the number of standard deviations."""
        return covariance_ellipse(self.Sigma, k)

    def __repr__(self):
        return f"Gaussian(Sigma={np.asarray(self.Sigma).tolist()})"


@dataclass
class UniformBox(VelocityUncertainty):
    """Uniform-interval error on a rotated box; ``body(k)`` scales the half-widths.

    Attributes
    ----------
    half_widths : ndarray, shape (2,)
        Per-axis half-extents of the error box.
    theta : float
        Box rotation from the axes, in radians.
    """

    half_widths: np.ndarray
    theta: float = 0.0

    def body(self, k=1.0):
        """Box scaled by ``k`` (support scale; ``k=1`` is the full support)."""
        hx, hy = k * np.asarray(self.half_widths, dtype=float)
        corners = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
        c, s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c, -s], [s, c]])
        return Polygon(corners @ R.T)


def _outer_arc(a_from, a_to, a_through, n):
    """Angles from a_from to a_to along the arc that passes through a_through."""
    span = (a_to - a_from) % (2 * np.pi)
    if (a_through - a_from) % (2 * np.pi) <= span:
        return np.linspace(a_from, a_from + span, n)
    return np.linspace(a_from, a_from - (2 * np.pi - span), n)


@dataclass
class RobustVelocityObstacle:
    apex: np.ndarray
    body: Shape                          # K, the uncertainty summand (centered at origin)
    contains_origin: bool
    contacts: np.ndarray | None = None   # tangent points where edges meet K
    edges: np.ndarray | None = None      # unit edge rays (unchanged by inflation)
    cap: np.ndarray | None = None        # rounded-apex arc, contacts[0] -> contacts[1]


def robust_velocity_obstacle(vo, K, n_cap=64):
    """Inflate a VelocityObstacle by a convex body ``K``, forming ``CollisionCone (+) K``.

    Parameters
    ----------
    vo : VelocityObstacle
        The obstacle to inflate.
    K : Shape
        Convex body centered at the origin, e.g. ``(own + target).body(k)``.
    n_cap : int
        Number of samples along the rounded-apex arc.

    Returns
    -------
    RobustVelocityObstacle
        If the cone contains the origin, the geometry fields are None (all unsafe).
    """
    a = np.asarray(vo.apex, dtype=float)

    if vo.cone.contains_origin:
        return RobustVelocityObstacle(a, K, contains_origin=True)

    (n1, n2), (e1, e2) = vo.cone.normals, vo.cone.edges
    contacts = np.array([a + K.support_point(n1), a + K.support_point(n2)])

    b = e1 + e2
    b = b / np.linalg.norm(b)
    angs = _outer_arc(np.arctan2(n1[1], n1[0]),
                      np.arctan2(n2[1], n2[0]),
                      np.arctan2(-b[1], -b[0]), n_cap)
    cap = np.array([a + K.support_point(_dir(t)) for t in angs])

    return RobustVelocityObstacle(a, K, False, contacts,
                                  np.array([e1, e2]), cap)


def uncertain_velocity_obstacle(O, v_target, pos_uncertainty, vel_uncertainty,
                                k_pos=2.0, k_vel=2.0, n_grid=1440):
    """VO under position and velocity uncertainty.

    Parameters
    ----------
    O : Shape
        Config-space obstacle (relative-position space).
    v_target : array_like, shape (2,)
        Target velocity (the cone apex).
    pos_uncertainty : VelocityUncertainty or Shape
        Position error inflating ``O``.
    vel_uncertainty : VelocityUncertainty or Shape
        Velocity error inflating the cone.
    k_pos, k_vel : float
        Confidence levels for the position and velocity bodies.
    n_grid : int
        Angular resolution of the cone tangent search.

    Returns
    -------
    RobustVelocityObstacle
    """
    K_pos = pos_uncertainty.body(k_pos) if isinstance(pos_uncertainty, VelocityUncertainty) \
        else pos_uncertainty
    K_vel = vel_uncertainty.body(k_vel) if isinstance(vel_uncertainty, VelocityUncertainty) \
        else vel_uncertainty

    vo = VelocityObstacle(MinkowskiSum(O, K_pos), v_target, n_grid=n_grid)
    return robust_velocity_obstacle(vo, K_vel)
