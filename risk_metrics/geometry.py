"""Convex shapes in the plane represented by their support functions."""

import numpy as np


def _rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


class Shape:
    """A closed convex set in R^2, defined by its support function.

    To add a shape, implement ``support(d)``, ``support_point(d)`` and ``reflect()``;
    Minkowski sums and velocity obstacles use only these three.
    """

    def support(self, d):
        r"""$h(d) = \max_{x \in S} d \cdot x$; $d$ does not need to be unit length."""
        raise NotImplementedError

    def support_point(self, d):
        r"""A point $x \in S$ attaining $h(d)$."""
        raise NotImplementedError

    def reflect(self):
        """The set $-S$, whose support is $h_{-S}(d) = h_S(-d)$."""
        raise NotImplementedError

    def sample_boundary(self, n=180):
        """``n`` boundary points: the support points at evenly spaced angles."""
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        dirs = np.column_stack([np.cos(theta), np.sin(theta)])
        return np.array([self.support_point(d) for d in dirs])

    def __str__(self):
        return type(self).__name__


class Circle(Shape):
    """Filled circle of radius ``r``."""

    def __init__(self, r, center=(0.0, 0.0)):
        self.r = float(r)
        self.center = np.asarray(center, dtype=float)

    def support(self, d):
        d = np.asarray(d, dtype=float)
        return float(self.center @ d + self.r * np.linalg.norm(d))

    def support_point(self, d):
        d = np.asarray(d, dtype=float)
        return self.center + self.r * d / np.linalg.norm(d)

    def reflect(self):
        return Circle(self.r, -self.center)

    def __repr__(self):
        return f"Circle(r={self.r}, center={self.center.tolist()})"


class Ellipse(Shape):
    """Filled ellipse with semi-axes ``a`` (along ``theta``, radians) and ``b``."""

    def __init__(self, a, b, center=(0.0, 0.0), theta=0.0):
        self.a = float(a)
        self.b = float(b)
        self.center = np.asarray(center, dtype=float)
        self.theta = float(theta)

    @property
    def rotation(self):
        return _rotation(self.theta)

    @property
    def M(self):
        return self.rotation @ np.diag([self.a, self.b])

    def support(self, d):
        d = np.asarray(d, dtype=float)
        return float(self.center @ d + np.linalg.norm(self.M.T @ d))

    def support_point(self, d):
        d = np.asarray(d, dtype=float)
        M = self.M
        w = M.T @ d
        w_norm = np.linalg.norm(w)
        if w_norm == 0.0:
            return self.center.copy()
        return self.center + (M @ w) / w_norm

    def reflect(self):
        return Ellipse(self.a, self.b, -self.center, self.theta)

    def __repr__(self):
        return (f"Ellipse(a={self.a}, b={self.b}, "
                f"center={self.center.tolist()}, theta={self.theta})")


class Polygon(Shape):
    """Filled convex polygon given by its vertices (convexity is assumed, not checked).

    When ``d`` is normal to an edge the first maximizing vertex is returned.
    """

    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=float)

    def support(self, d):
        d = np.asarray(d, dtype=float)
        return float(np.max(self.vertices @ d))

    def support_point(self, d):
        d = np.asarray(d, dtype=float)
        return self.vertices[int(np.argmax(self.vertices @ d))].copy()

    def reflect(self):
        return Polygon(-self.vertices)

    def sample_boundary(self, n=None):
        return self.vertices.copy()

    def __repr__(self):
        return f"Polygon({len(self.vertices)} vertices)"


class MinkowskiSum(Shape):
    r"""$A \oplus B \oplus \dots$; support functions and support points add.
    """

    def __init__(self, *shapes):
        parts = []
        for s in shapes:
            if isinstance(s, MinkowskiSum):
                parts.extend(s.parts)
            elif isinstance(s, Shape):
                parts.append(s)
            else:
                raise TypeError(f"MinkowskiSum parts must be Shape, got {type(s)}")
        if not parts:
            raise ValueError("MinkowskiSum needs at least one shape")
        self.parts = parts

    def support(self, d):
        return np.sum([p.support(d) for p in self.parts])

    def support_point(self, d):
        return np.sum([p.support_point(d) for p in self.parts], axis=0)

    def reflect(self):
        return MinkowskiSum(*[s.reflect() for s in self.parts])

    def __repr__(self):
        return "MinkowskiSum(" + ", ".join(repr(p) for p in self.parts) + ")"


def covariance_ellipse(Sigma, k=1.0):
    r"""The $k\sigma$ ellipse of a 2x2 covariance, centered at the origin.

    Its support is $h(d) = k \sqrt{d^T \Sigma d}$.
    """
    vals, vecs = np.linalg.eigh(Sigma)
    theta = np.arctan2(vecs[1, 0], vecs[0, 0])
    a, b = k * np.sqrt(vals)
    return Ellipse(a, b, center=(0, 0), theta=theta)
