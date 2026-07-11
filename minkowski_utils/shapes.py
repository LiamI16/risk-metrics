import numpy as np


class Shape:
    """Base class for closed, filled convex shapes in R^2.

    Subclasses must implement ``support``, ``support_point`` and ``reflect``.
    The base supplies ``sample_boundary``.
    """

    def support(self, d):
        """h_S(d) = max_{x in S} d . x.  Accepts non-unit d."""
        raise NotImplementedError

    def support_point(self, d):
        """argmax_{x in S} d . x"""
        raise NotImplementedError

    def reflect(self):
        """The reflected body -S = {-x : x in S}.  Note h_{-S}(d) = h_S(-d)."""
        raise NotImplementedError

    def sample_boundary(self, n=180):
        """Polygonal boundary samples from the support-point function.

        Sweeps directions uniformly in angle and reads off support points.
        """
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        dirs = np.column_stack([np.cos(theta), np.sin(theta)])
        return np.array([self.support_point(d) for d in dirs])


class Circle(Shape):
    """Filled disk of radius r centred at ``center``"""

    def __init__(self, r, center=(0.0, 0.0)):
        self.r = float(r)
        self.center = np.asarray(center, dtype=float)

    def support(self, d):
        d = np.asarray(d, dtype=float)
        return float(self.center @ d + self.r * np.linalg.norm(d))

    def support_point(self, d):
        d = np.asarray(d, dtype=float)
        return self.center + self.r * d / np.linalg.norm(d)

    def reflect(self):                     # a disk is centrally symmetric
        return Circle(self.r, -self.center)

    def __repr__(self):
        return f"Circle(r={self.r}, center={self.center.tolist()})"


class Ellipse(Shape):
    """Filled ellipse: semi-axes (a, b), centre c, rotation theta (rad).

    Written as an affine image of the unit disk.
    """

    def __init__(self, a, b, center=(0.0, 0.0), theta=0.0):
        self.a = float(a)
        self.b = float(b)
        self.center = np.asarray(center, dtype=float)
        self.theta = float(theta)

    @property
    def R(self):
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([[c, -s], [s, c]])

    @property
    def M(self):
        """Affine map sending the unit disk to the (centred) ellipse."""
        return self.R @ np.diag([self.a, self.b])

    def support(self, d):
        d = np.asarray(d, dtype=float)
        return float(self.center @ d + np.linalg.norm(self.M.T @ d))

    def support_point(self, d):
        d = np.asarray(d, dtype=float)
        M = self.M
        w = M.T @ d
        return self.center + (M @ w) / np.linalg.norm(w)

    def reflect(self):                     # an ellipse is centrally symmetric
        return Ellipse(self.a, self.b, -self.center, self.theta)

    def __repr__(self):
        return (f"Ellipse(a={self.a}, b={self.b}, "
                f"center={self.center.tolist()}, theta={self.theta})")


class Polygon(Shape):
    """Filled convex polygon.
    
    Vertices are assumed to describe a convex region; the support function reads
    only the extreme points, so a consistent tie-break (first argmax) is used
    when d is perpendicular to an edge.
    """

    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=float)
        self.sides = len(self.vertices)

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
        return f"Polygon({self.sides} vertices)"
