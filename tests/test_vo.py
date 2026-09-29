"""Velocity obstacles, deterministic and uncertainty-aware, against collides() by brute force."""

import numpy as np
import pytest

from risk_metrics import (
    Circle, Ellipse, Polygon, MinkowskiSum, Gaussian, UniformBox, Ship, collides,
    config_space_obstacle, VelocityObstacle, UncertaintyAwareVelocityObstacle,
    CANONICAL_SCENARIOS, CROSSING, HEAD_ON, NORTH_EAST,
)

# A triangle with its centroid at the origin; no triangle equals its own reflection.
TRIANGLE = Polygon([(1.5, 0.0), (-0.5, 1.0), (-1.0, -1.0)])


def _circle_pair(R=2.0, D=10.0, v_target=(0.0, 0.0)):
    """Own ship a point at the origin; circular target of radius R at distance D ahead (+y)."""
    return (Ship([0.0, 0.0], [0.0, 0.0], Circle(0.0)),
            Ship([0.0, D], v_target, Circle(R)))


def _fill(P, n_edge=200, n_int=300, seed=0):
    """Dense points of a convex polygon: vertices, along every edge, and interior."""
    V = P.vertices
    s = np.linspace(0, 1, n_edge, endpoint=False)[:, None]
    edges = np.vstack([V[i] + (V[(i + 1) % len(V)] - V[i]) * s for i in range(len(V))])
    w = np.random.default_rng(seed).dirichlet(np.ones(len(V)), n_int)
    return np.vstack([V, edges, w @ V])


def _circle_cone_margin(x, relpos, R, h):
    r"""Signed distance of own-space offsets x to cone(circle) $\oplus$ K, with K given by its support h."""
    # For a closed convex set it is $\max_{\|d\| = 1} d \cdot x - h_{set}(d)$; the cone's
    # support is 0 on its polar arc and infinite elsewhere.
    D = np.linalg.norm(relpos)
    alpha = np.arcsin(R / D)
    phi = np.arctan2(relpos[1], relpos[0]) + np.linspace(np.pi / 2 + alpha, 1.5 * np.pi - alpha, 2001)
    d = np.column_stack([np.cos(phi), np.sin(phi)])
    return np.max(x @ d.T - np.array([h(di) for di in d]), axis=1)


def test_circle_cone_half_angle_is_arcsin():
    """Circle of radius R at distance D: each edge is arcsin(R/D) off the axis, symmetrically."""
    vo = VelocityObstacle(*_circle_pair(R=2.0, D=10.0))
    assert not vo.already_in_collision
    assert np.allclose(vo.edges[:, 1], np.cos(np.arcsin(0.2)), atol=1e-6)
    assert np.isclose(vo.edges[0, 0], -vo.edges[1, 0], atol=1e-6)


@pytest.mark.parametrize("target_domain", [
    Circle(2.0), Ellipse(3.0, 1.0, theta=0.6),
    MinkowskiSum(Circle(1.0), Ellipse(2.0, 1.0, theta=0.3)), TRIANGLE])
def test_tangency_invariant(target_domain):
    """For convex C off the origin: two unit normals with $h_C(n) = 0$, distinct edges, tangent points on C."""
    own, target = Ship([0, 0], [0, 0], Ellipse(1.0, 0.5, theta=0.2)), Ship([1.0, 9.0], [0, 0], target_domain)
    vo = VelocityObstacle(own, target)
    C = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
    assert vo.normals.shape == vo.edges.shape == (2, 2)
    for n, e, p in zip(vo.normals, vo.edges, vo.tangent_points):
        assert np.isclose(C.support(n), 0.0, atol=1e-6)
        assert np.isclose(n @ p, 0.0, atol=1e-6) and np.isclose(np.linalg.norm(e), 1.0)
    assert not np.allclose(*vo.edges)


def test_vo_basic_membership():
    """Membership for own velocities built as v_target + offset, with v_target non-zero.

    Aiming straight at the circle collides. Aiming across it does not. Matching the target
    velocity does not collide either, because the separation never changes. The last two
    offsets straddle the cone edge, one just inside the half-angle and one just outside.
    """
    vt = np.array([3.0, -1.0])
    vo = VelocityObstacle(*_circle_pair(v_target=vt))
    ha = np.arcsin(0.2)
    assert vo.contains(vt + [0, 1]) and not vo.contains(vt + [1, 0]) and not vo.contains(vt)
    assert vo.contains(vt + [np.sin(0.9 * ha), np.cos(0.9 * ha)])
    assert not vo.contains(vt + [np.sin(1.1 * ha), np.cos(1.1 * ha)])


@pytest.mark.parametrize("sc", CANONICAL_SCENARIOS, ids=lambda s: s.name)
def test_vo_contains_matches_collides(sc):
    """VO.contains(v_own) == collides(relpos, v_target - v_own, R), away from a 1e-6 rad band at the edges."""
    own, target = sc.ships()
    vo = VelocityObstacle(own, target)
    assert not vo.already_in_collision
    relpos = target.pos - own.pos
    v = target.vel + np.random.default_rng(4).uniform(-12, 12, (4000, 2))
    w = v - target.vel                                  # = -relvel
    ang = np.arccos(np.clip(w @ relpos / (np.linalg.norm(w, axis=1) * np.linalg.norm(relpos)), -1, 1))
    keep = np.abs(ang - np.arcsin(sc.R / np.linalg.norm(relpos))) > 1e-6
    truth = collides(relpos, target.vel - v, sc.R)
    assert np.array_equal(vo.contains(v)[keep], truth[keep])
    assert truth.any()


def test_contains_shape_handling():
    """(2,) -> bool, (n, 2) -> (n,), (a, b, 2) -> (a, b), for both classes."""
    own, target = CROSSING.ships()
    for obj in (VelocityObstacle(own, target),
                UncertaintyAwareVelocityObstacle(own, target, K_v=TRIANGLE)):
        assert type(obj.contains(own.vel)) is bool
        assert obj.contains(np.zeros((7, 2))).shape == (7,)
        assert obj.contains(np.zeros((3, 5, 2))).shape == (3, 5)


def test_already_in_collision_when_ships_overlap():
    """Overlapping domains: every velocity collides and the cone geometry is None."""
    own, target = Ship([0, 0], [0, 0], Circle(5.0)), Ship([1.0, 0], [0, 0], Circle(5.0))
    vo = VelocityObstacle(own, target)
    uvo = UncertaintyAwareVelocityObstacle(own, target, K_v=Gaussian(np.eye(2)).uncertainty_set(1.0))
    assert vo.already_in_collision and vo.edges is None and vo.tangent_points is None
    assert uvo.already_in_collision and uvo.contacts is None and uvo.cap is None
    v = np.random.default_rng(0).normal(size=(10, 2))
    assert vo.contains(v).all() and uvo.contains(v).all() and vo.contains([1.0, 0.0]) is True


def test_uvo_without_uncertainty_equals_vo():
    own, target = NORTH_EAST.ships()
    vo, uvo = VelocityObstacle(own, target), UncertaintyAwareVelocityObstacle(own, target)
    v = target.vel + np.random.default_rng(1).uniform(-10, 10, (2000, 2))
    assert np.allclose(uvo.edges, vo.edges) and np.array_equal(uvo.contains(v), vo.contains(v))


def test_uvo_velocity_inflation_geometry():
    """K_v keeps the edges, puts contacts at apex + support_point(K_v, n), caps away from the cone."""
    own, target = _circle_pair(D=20.0, v_target=(0.0, -6.0))
    vo = VelocityObstacle(own, target)
    for K in (Gaussian(2.25 * np.eye(2)).uncertainty_set(2.0), UniformBox([1.0, 2.0]).uncertainty_set(1.0),
              (Gaussian(np.diag([1.0, 0.4])) + UniformBox([0.5, 0.5], theta=0.3)).uncertainty_set(2.0)):
        uvo = UncertaintyAwareVelocityObstacle(own, target, K_v=K)
        assert np.allclose(uvo.edges, vo.edges, atol=1e-12)
        for p, n in zip(uvo.contacts, vo.normals):
            assert np.allclose(p, uvo.apex + K.support_point(n), atol=1e-12)
        assert np.allclose(uvo.cap[[0, -1]], uvo.contacts, atol=1e-9)
        b = vo.edges.sum(axis=0)
        assert (uvo.cap[len(uvo.cap) // 2] - uvo.apex) @ b < 0
    # isotropic K_v = circle of radius 3: each contact sits 3 from the apex along its normal
    uvo = UncertaintyAwareVelocityObstacle(own, target, K_v=Circle(3.0))
    assert np.allclose(uvo.contacts - uvo.apex, 3.0 * vo.normals, atol=1e-9)


@pytest.mark.parametrize("sc", [CROSSING, HEAD_ON, NORTH_EAST], ids=lambda s: s.name)
def test_uvo_velocity_convention_asymmetric_kv(sc):
    r"""contains(v_own) == "some $k \in K_v$ gives collides(relpos, v_target - v_own + k, R)"."""
    # K_v is the set of errors k on relvel (true relvel = v_target - v_own + k), checked by brute
    # force over a dense fill of an asymmetric triangle; the reflected set $-K_v$ must disagree.
    own, target = sc.ships()
    relpos = target.pos - own.pos
    uvo = UncertaintyAwareVelocityObstacle(own, target, K_v=TRIANGLE)
    v = target.vel + np.random.default_rng(3).uniform(-5, 5, (3000, 2))
    x, ks = v - target.vel, _fill(TRIANGLE)
    truth = {s: collides(relpos, (target.vel - v)[:, None] + s * ks[None], sc.R).any(axis=1) for s in (1, -1)}
    keep = np.ones(len(v), dtype=bool)
    for K in (TRIANGLE, TRIANGLE.reflect()):
        keep &= np.abs(_circle_cone_margin(x, relpos, sc.R, K.support)) > 0.05
    got = uvo.contains(v)
    assert keep.mean() > 0.95
    assert np.array_equal(got[keep], truth[1][keep])
    differ = keep & (truth[1] != truth[-1])
    assert differ.sum() > 20 and not np.any(got[differ] == truth[-1][differ])


def test_uvo_position_uncertainty_matches_brute_force():
    r"""With K_p only: contains(v_own) == "some $k \in K_p$ gives collides(relpos + k, relvel, R)"."""
    # Brute force over a dense fill of a triangle K_p.
    own, target = CROSSING.ships()
    relpos, K_p = target.pos - own.pos, Polygon(10.0 * TRIANGLE.vertices)
    uvo = UncertaintyAwareVelocityObstacle(own, target, K_p=K_p)
    v = target.vel + np.random.default_rng(5).uniform(-10, 10, (3000, 2))
    w = v - target.vel
    wn = w / np.linalg.norm(w, axis=1, keepdims=True)
    keep = np.all(np.abs(wn[:, 0, None] * uvo.edges[:, 1] - wn[:, 1, None] * uvo.edges[:, 0]) > 2e-3, axis=1)
    ks = _fill(K_p)
    truth = collides(relpos + ks[None], (target.vel - v)[:, None], CROSSING.R).any(axis=1)
    assert keep.mean() > 0.95 and truth.mean() > 0.05
    assert np.array_equal(uvo.contains(v)[keep], truth[keep])


def test_uvo_circle_position_inflation_closed_form_and_nested():
    r"""Isotropic $k\sigma$ circle K_p on a circular obstacle: half-angle $\arcsin((R + k\sigma)/D)$, growing with k."""
    R, D, sigma = 2.0, 20.0, 1.5
    own, target = _circle_pair(R=R, D=D)
    halves = [np.arccos(VelocityObstacle(own, target).edges[:, 1])]
    for k in (1.0, 2.0, 3.0):
        uvo = UncertaintyAwareVelocityObstacle(own, target, K_p=Gaussian(sigma ** 2 * np.eye(2)).uncertainty_set(k))
        halves.append(np.arccos(uvo.edges[:, 1]))
        assert np.allclose(halves[-1], np.arcsin((R + k * sigma) / D), atol=1e-6)
    assert np.all(np.diff(np.array(halves)[:, 0]) > 0)


def test_zero_size_body_is_the_origin():
    r"""Gaussian($\Sigma$).uncertainty_set(0) is the point {0}: UVO with it as K_p or K_v equals the plain VO."""
    own, target = CROSSING.ships()
    K0 = Gaussian(np.eye(2)).uncertainty_set(0.0)
    assert np.allclose(K0.support_point([1.0, 0.0]), 0.0)
    v = target.vel + np.random.default_rng(2).uniform(-10, 10, (500, 2))
    expect = VelocityObstacle(own, target).contains(v)
    for kw in ({"K_p": K0}, {"K_v": K0}):
        assert np.array_equal(UncertaintyAwareVelocityObstacle(own, target, **kw).contains(v), expect)


def test_half_angle_and_outline():
    """Circle half-angle is arcsin(R/D); outline starts and ends on the edges and passes the apex or cap."""
    own, target = CROSSING.ships()
    vo = VelocityObstacle(own, target)
    D = np.linalg.norm(target.pos - own.pos)
    assert np.isclose(vo.half_angle, np.arcsin(CROSSING.R / D))
    line = vo.outline(10.0)
    assert np.allclose(line[1], vo.apex)
    assert np.allclose(line[[0, 2]], vo.apex + 10.0 * vo.edges)
    uvo = UncertaintyAwareVelocityObstacle(own, target, K_v=Circle(1.0))
    line = uvo.outline(10.0)
    assert np.allclose(line[1:-1], uvo.cap)
    assert np.allclose(line[[0, -1]], uvo.contacts + 10.0 * uvo.edges)
