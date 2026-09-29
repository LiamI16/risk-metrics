"""Support-function properties of the shapes, Minkowski sums and covariance ellipses."""

import numpy as np
import pytest

from risk_metrics import Circle, Ellipse, Polygon, MinkowskiSum, covariance_ellipse


def _dirs(n=200, seed=0):
    """Random unit directions plus the four axis directions."""
    a = np.random.default_rng(seed).uniform(0, 2 * np.pi, n)
    return np.vstack([np.column_stack([np.cos(a), np.sin(a)]),
                      [[1, 0], [0, 1], [-1, 0], [0, -1]]])


SHAPES = [
    Circle(2.0, center=(1.0, -0.5)),
    Ellipse(3.0, 1.0, center=(-1.0, 2.0), theta=0.7),
    Polygon([(0, 0), (4, 0), (5, 2), (2, 3), (-1, 1)]),
]
SHAPES_AND_SUM = SHAPES + [MinkowskiSum(*SHAPES)]


@pytest.mark.parametrize("S", SHAPES_AND_SUM, ids=str)
def test_support_point_attains_support(S):
    r"""$d \cdot$ support_point(d) == support(d)."""
    for d in _dirs():
        assert np.isclose(d @ S.support_point(d), S.support(d), atol=1e-10)


@pytest.mark.parametrize("S", SHAPES_AND_SUM, ids=str)
def test_support_is_positively_homogeneous(S):
    r"""$h(t d) = t\,h(d)$ for $t > 0$."""
    for d in _dirs(50):
        for t in (0.3, 2.0, 7.5):
            assert np.isclose(S.support(t * d), t * S.support(d), atol=1e-10)


@pytest.mark.parametrize("S", SHAPES_AND_SUM, ids=str)
def test_reflect_support_identity(S):
    """$h_{-S}(d) = h_S(-d)$."""
    Sr = S.reflect()
    for d in _dirs():
        assert np.isclose(Sr.support(d), S.support(-d), atol=1e-12)


@pytest.mark.parametrize("S", SHAPES[:2], ids=str)
def test_support_point_is_gradient_on_smooth_bodies(S):
    """support_point(d) is the gradient of h at d, checked on the two smooth shapes.

    A polygon is left out: its h is not differentiable where d is normal to an edge.
    """
    eps = 1e-6
    for d in _dirs(50):
        g = [(S.support(d + e) - S.support(d - e)) / (2 * eps) for e in eps * np.eye(2)]
        assert np.allclose(g, S.support_point(d), atol=1e-4)


def test_circle_closed_form():
    r"""Circle: $h(d) = c \cdot d + r$ for unit d."""
    S = SHAPES[0]
    for d in _dirs():
        assert np.isclose(S.support(d), S.center @ d + S.r, atol=1e-12)


def test_ellipse_support_on_principal_axes():
    """Along each principal axis a centered ellipse's support is that semi-axis."""
    S = Ellipse(3.0, 1.0, theta=0.7)
    assert np.isclose(S.support(S.rotation[:, 0]), 3.0, atol=1e-12)
    assert np.isclose(S.support(S.rotation[:, 1]), 1.0, atol=1e-12)


def test_polygon_support_is_max_vertex():
    P = SHAPES[2]
    for d in _dirs():
        assert np.isclose(P.support(d), np.max(P.vertices @ d), atol=1e-12)


def test_sample_boundary_points_lie_on_boundary():
    r"""Every sampled boundary point x of an ellipse has $\|M^{-1}(x - c)\| = 1$."""
    E = SHAPES[1]
    xy = E.sample_boundary(64)
    assert xy.shape == (64, 2)
    assert np.allclose(np.linalg.norm(np.linalg.solve(E.M, (xy - E.center).T), axis=0), 1.0)


def test_minkowski_support_and_support_point_add():
    A, B, C = SHAPES
    S = MinkowskiSum(A, B, C)
    for d in _dirs():
        assert np.isclose(S.support(d), A.support(d) + B.support(d) + C.support(d), atol=1e-12)
        assert np.allclose(S.support_point(d),
                           A.support_point(d) + B.support_point(d) + C.support_point(d), atol=1e-12)


def test_minkowski_circle_plus_circle_is_circle():
    r"""Circle(r1) $\oplus$ Circle(r2) == Circle(r1 + r2), centers add."""
    A, B = Circle(2.0, (1.0, -0.5)), Circle(3.0, (0.5, 4.0))
    ref = Circle(5.0, A.center + B.center)
    for d in _dirs():
        assert np.isclose(MinkowskiSum(A, B).support(d), ref.support(d), atol=1e-12)


def test_minkowski_order_does_not_matter():
    A, B, C = SHAPES
    for d in _dirs(50):
        assert np.isclose(MinkowskiSum(A, B, C).support(d), MinkowskiSum(C, A, B).support(d))


def test_minkowski_nested_sums_flatten():
    A, B, C = SHAPES
    assert len(MinkowskiSum(MinkowskiSum(A, B), C).parts) == 3


def test_minkowski_rejects_bad_parts():
    with pytest.raises(TypeError):
        MinkowskiSum(Circle(1.0), np.eye(2))
    with pytest.raises(ValueError):
        MinkowskiSum()


def test_covariance_ellipse_support_identity():
    r"""$h(d) = k \sqrt{d^T \Sigma d}$ for the $k\sigma$ ellipse."""
    Sigma, k = np.array([[4.0, 1.0], [1.0, 2.0]]), 1.7
    E = covariance_ellipse(Sigma, k)
    for d in _dirs():
        assert np.isclose(E.support(d), k * np.sqrt(d @ Sigma @ d), atol=1e-9)


def test_isotropic_covariance_is_circle():
    r"""$\Sigma = s^2 I$ gives a circle of radius $k s$."""
    E = covariance_ellipse(4.0 * np.eye(2), k=3.0)
    for d in _dirs():
        assert np.isclose(E.support(d), 6.0, atol=1e-9)
