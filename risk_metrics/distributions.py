from dataclasses import dataclass
import numpy as np
from .geometry import Ellipse, MinkowskiSum, Polygon, _rotation, covariance_ellipse


def _principal_axes(Sigma):
    vals, vecs = np.linalg.eigh(np.asarray(Sigma, dtype=float))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = float(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return np.sqrt(np.maximum(vals, 0.0)), theta


class Distribution:
    """A zero-mean error distribution in R^2 (position or velocity noise); errors add: ``a + b``.

    To add a model, implement ``sample(n, rng)``, returning (n, 2) zero-mean draws; implement
    ``uncertainty_set(k)`` (a convex Shape centered at the origin) only for the uncertainty-aware VO.
    ``reflect()`` is correct as inherited; override it with ``return self`` if the model is
    centrally symmetric.
    """

    def sample(self, n, rng):
        """``n`` zero-mean draws, shape (n, 2), from a ``numpy.random.Generator``."""
        raise NotImplementedError

    def uncertainty_set(self, k):
        """Convex confidence set at level ``k``, centered at the origin; each model defines ``k``."""
        raise NotImplementedError

    def reflect(self):
        r"""The error set $-D$. An own-ship error enters a relative state reflected."""
        return Reflected(self)

    def __add__(self, other):
        if not isinstance(other, Distribution):
            return NotImplemented
        return IndependentSum(self, other)


class Reflected(Distribution):
    r"""The error set $-D$: draws and ``uncertainty_set(k)`` negated. Correct for any convex body, so
    ``Distribution.reflect()`` returns one by default. Symmetric models override it.
    """

    def __init__(self, d):
        self.d = d

    def sample(self, n, rng):
        return -self.d.sample(n, rng)

    def uncertainty_set(self, k):
        return self.d.uncertainty_set(k).reflect()

    def reflect(self):
        return self.d                                   # reflecting twice cancels

    def __repr__(self):
        return f"Reflected({self.d!r})"


class IndependentSum(Distribution):
    """Sum of independent errors (``a + b``); ``uncertainty_set(k)`` is the Minkowski sum of the parts' ``uncertainty_set(k)``."""

    def __init__(self, *parts):
        flat = []
        for p in parts:
            flat.extend(p.parts if isinstance(p, IndependentSum) else [p])
        self.parts = tuple(flat)

    def sample(self, n, rng):
        return sum(p.sample(n, rng) for p in self.parts)

    def uncertainty_set(self, k):
        return MinkowskiSum(*(p.uncertainty_set(k) for p in self.parts))

    def reflect(self):
        return IndependentSum(*(p.reflect() for p in self.parts))

    def __repr__(self):
        return "IndependentSum(" + ", ".join(repr(p) for p in self.parts) + ")"


@dataclass(eq=False)
class Gaussian(Distribution):
    r"""$N(0, \Sigma)$. ``uncertainty_set(k)`` is the $k\sigma$ ellipse."""

    Sigma: np.ndarray

    def sample(self, n, rng):
        return rng.multivariate_normal(np.zeros(2), np.asarray(self.Sigma, dtype=float), size=n)

    def uncertainty_set(self, k):
        return covariance_ellipse(self.Sigma, k)

    def reflect(self):
        return self                  # centrally symmetric


@dataclass(eq=False)
class UniformEllipse(Distribution):
    """Uniform over an ellipse with semi-axes ``a`` (along ``theta``, radians) and ``b``.
    ``uncertainty_set(k)`` scales the semi-axes, so ``uncertainty_set(1)`` is the support.
    """

    a: float
    b: float
    theta: float = 0.0

    @classmethod
    def from_covariance(cls, Sigma):
        r"""Equal-covariance uniform ellipse: semi-axes $2\sigma$, so its edge is the $2\sigma$ ellipse."""
        sig, theta = _principal_axes(Sigma)
        return cls(a=2.0 * sig[0], b=2.0 * sig[1], theta=theta)

    def sample(self, n, rng):
        r = np.sqrt(rng.uniform(0.0, 1.0, size=n))       # sqrt makes it uniform in area
        phi = rng.uniform(0.0, 2 * np.pi, size=n)
        pts = np.column_stack((r * np.cos(phi), r * np.sin(phi)))
        return pts @ Ellipse(self.a, self.b, theta=self.theta).M.T

    def uncertainty_set(self, k):
        return Ellipse(k * self.a, k * self.b, center=(0, 0), theta=self.theta)

    def reflect(self):
        return self                  # centrally symmetric


@dataclass(eq=False)
class UniformBox(Distribution):
    """Uniform over a box with ``half_widths`` (hx, hy), rotated by ``theta``.
    ``uncertainty_set(k)`` scales the half-widths, so ``uncertainty_set(1)`` is the support.
    """

    half_widths: np.ndarray
    theta: float = 0.0

    @classmethod
    def from_covariance(cls, Sigma):
        r"""Equal-covariance uniform box: half-widths $h = \sqrt{3}\,\sigma$, since variance is $h^2/3$."""
        sig, theta = _principal_axes(Sigma)
        return cls(half_widths=np.sqrt(3.0) * sig, theta=theta)

    def sample(self, n, rng):
        pts = rng.uniform(-1.0, 1.0, size=(n, 2)) * np.asarray(self.half_widths, dtype=float)
        return pts @ _rotation(self.theta).T

    def uncertainty_set(self, k):
        hx, hy = k * np.asarray(self.half_widths, dtype=float)
        corners = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
        return Polygon(corners @ _rotation(self.theta).T)

    def reflect(self):
        return self                  # centrally symmetric
