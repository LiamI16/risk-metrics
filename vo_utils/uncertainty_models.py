"""Uncertainty models, and the velocity obstacles they inflate.

An ``UncertaintyModel`` is a zero-mean 2-D error distribution with two views:
``body(k)`` returns a convex confidence Shape, and ``sample(n, rng)`` draws error
points. Independent errors add (``own + target``). Models describe either channel --
position error and velocity error use the same classes.

Each view serves one consumer. ``uncertain_velocity_obstacle`` takes the Shape and
inflates the geometry to ``VO = v_target (+) CollisionCone (+) K``; ``sampling.sample_cpa``
takes the draws instead. Neither needs the other view.
"""

from abc import ABC, abstractmethod

import numpy as np
from dataclasses import dataclass

from minkowski_utils import Shape, Ellipse, Polygon, MinkowskiSum
from .cone import _dir
from .obstacle import VelocityObstacle
from .uncertainty import covariance_ellipse


class UncertaintyModel(ABC):
    """A zero-mean 2-D error distribution, usable as either a set or a sampler.

    Subclasses supply both views of the same distribution: ``body(k)`` for the
    set-based geometry, ``sample(n, rng)`` for Monte Carlo. A consumer uses only the
    view it needs. The same model describes position or velocity error.
    """

    @abstractmethod
    def body(self, k: float) -> Shape:
        """Convex confidence body at level ``k``, centered at the origin.

        Parameters
        ----------
        k : float
            Level, interpreted per distribution: standard deviations for a Gaussian,
            a scale on the support for a bounded model (where ``k=1`` is the full
            support).

        Returns
        -------
        Shape
        """

    @abstractmethod
    def sample(self, n, rng):
        """Draw ``n`` zero-mean error samples.

        A bounded model's draws fill exactly ``body(1)``; a Gaussian's are unbounded.

        Parameters
        ----------
        n : int
            Number of samples.
        rng : numpy.random.Generator
            Random generator.

        Returns
        -------
        ndarray, shape (n, 2)
        """

    def __add__(self, other):
        """Combine independent errors: ``own + target`` -> IndependentSum."""
        if not isinstance(other, UncertaintyModel):
            return NotImplemented
        return IndependentSum(self, other)


class IndependentSum(UncertaintyModel):
    """Independent errors combined, as built by ``a + b``.

    ``body(k)`` is the Minkowski sum of the parts' bodies at a shared level; ``sample``
    adds one independent draw per part.

    Parameters
    ----------
    *parts : UncertaintyModel
        The summands; nested sums are flattened.
    """

    def __init__(self, *parts):
        flat = []
        for p in parts:
            flat.extend(p.parts if isinstance(p, IndependentSum) else [p])
        self.parts = tuple(flat)

    def body(self, k=2.0):
        """Minkowski sum of each part's ``body(k)`` at shared confidence level ``k``."""
        return MinkowskiSum(*(p.body(k) for p in self.parts))

    def sample(self, n, rng):
        """Sum of each part's independent samples."""
        return sum(p.sample(n, rng) for p in self.parts)

    def __repr__(self):
        return "IndependentSum(" + ", ".join(repr(p) for p in self.parts) + ")"


@dataclass
class Gaussian(UncertaintyModel):
    """Gaussian error ``N(0, Sigma)``; ``body(k)`` is the k-sigma covariance ellipse.

    Unbounded, so no ``body(k)`` contains all the draws -- ``k`` selects a confidence
    level rather than a support.

    Attributes
    ----------
    Sigma : ndarray, shape (2, 2)
        Error covariance.
    """

    Sigma: np.ndarray

    def body(self, k=2.0):
        """k-sigma confidence ellipse; ``k`` is the number of standard deviations."""
        return covariance_ellipse(self.Sigma, k)

    def sample(self, n, rng):
        """Draw from ``N(0, Sigma)`` (unbounded)."""
        return rng.multivariate_normal(np.zeros(2), np.asarray(self.Sigma, dtype=float), size=n)

    def __repr__(self):
        return f"Gaussian(Sigma={np.asarray(self.Sigma).tolist()})"


@dataclass
class UniformBox(UncertaintyModel):
    """Error uniform over a rotated box; ``body(k)`` scales the half-widths.

    Bounded: ``body(1)`` is the support, which the draws fill.

    Attributes
    ----------
    half_widths : ndarray, shape (2,)
        Per-axis half-extents.
    theta : float
        Rotation from the axes, in radians.
    """

    half_widths: np.ndarray
    theta: float = 0.0

    def body(self, k=1.0):
        """Box scaled by ``k`` (support scale; ``k=1`` is the full support)."""
        hx, hy = k * np.asarray(self.half_widths, dtype=float)
        corners = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
        return Polygon(corners @ self._rot().T)

    def sample(self, n, rng):
        """Uniform over the rotated box (fills ``body(1)``)."""
        pts = rng.uniform(-1.0, 1.0, size=(n, 2)) * np.asarray(self.half_widths, dtype=float)
        return pts @ self._rot().T

    def _rot(self):
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([[c, -s], [s, c]])


@dataclass
class UniformEllipse(UncertaintyModel):
    """Error uniform over a rotated ellipse; ``body(k)`` scales the semi-axes.

    Bounded: ``body(1)`` is the support, which the draws fill.

    Attributes
    ----------
    a, b : float
        Semi-axes.
    theta : float
        Rotation from the axes, in radians.
    """

    a: float
    b: float
    theta: float = 0.0

    def body(self, k=1.0):
        """Ellipse scaled by ``k`` (support scale; ``k=1`` is the full support)."""
        return Ellipse(k * self.a, k * self.b, center=(0, 0), theta=self.theta)

    def sample(self, n, rng):
        """Uniform over the ellipse area via the sqrt-radius unit-disk map (fills ``body(1)``)."""
        r = np.sqrt(rng.uniform(0.0, 1.0, size=n))       # sqrt makes it area-uniform
        phi = rng.uniform(0.0, 2 * np.pi, size=n)
        disk = np.column_stack((r * np.cos(phi), r * np.sin(phi)))   # uniform in unit disk
        return disk @ self.body(1.0).M.T                 # affine map onto the ellipse


def _outer_arc(a_from, a_to, a_through, n):
    """Angles from a_from to a_to along the arc that passes through a_through."""
    span = (a_to - a_from) % (2 * np.pi)
    if (a_through - a_from) % (2 * np.pi) <= span:
        return np.linspace(a_from, a_from + span, n)
    return np.linspace(a_from, a_from - (2 * np.pi - span), n)


@dataclass
class RobustVelocityObstacle:
    """A velocity obstacle inflated by an uncertainty body: ``CollisionCone (+) K``.

    Inflation offsets the cone's edges outward but does not rotate them, so ``edges``
    match the uninflated cone; the apex becomes a rounded ``cap`` joining the two
    ``contacts``. When the cone already contains the origin every velocity is unsafe
    and the geometry fields are None.
    """

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
    pos_uncertainty : UncertaintyModel or Shape
        Position error inflating ``O``.
    vel_uncertainty : UncertaintyModel or Shape
        Velocity error inflating the cone.
    k_pos, k_vel : float
        Confidence levels for the position and velocity bodies.
    n_grid : int
        Angular resolution of the cone tangent search.

    Returns
    -------
    RobustVelocityObstacle
    """
    K_pos = pos_uncertainty.body(k_pos) if isinstance(pos_uncertainty, UncertaintyModel) \
        else pos_uncertainty
    K_vel = vel_uncertainty.body(k_vel) if isinstance(vel_uncertainty, UncertaintyModel) \
        else vel_uncertainty

    vo = VelocityObstacle(MinkowskiSum(O, K_pos), v_target, n_grid=n_grid)
    return robust_velocity_obstacle(vo, K_vel)
