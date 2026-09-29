"""Noise distributions and Monte Carlo CPA sampling."""

import numpy as np
import pytest

from risk_metrics import (
    Circle, Distribution, IndependentSum, MinkowskiSum, Polygon, Reflected, Gaussian, UniformEllipse,
    UniformBox, Ship, covariance_ellipse, cpa, sample_cpa,
)

# Anisotropic and rotated: principal variances 25 and 4 at 30 degrees.
_c, _s = np.cos(np.pi / 6), np.sin(np.pi / 6)
SIGMA = np.array([[_c, -_s], [_s, _c]]) @ np.diag([25.0, 4.0]) @ np.array([[_c, _s], [-_s, _c]])


def _dirs(n=100):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(a), np.sin(a)])


def _encounter():
    """A generic crossing (TCPA > 0, DCPA != 0)."""
    return (Ship([0.0, 0.0], [1.0, 0.0], Circle(1.0)),
            Ship([30.0, 8.0], [-1.0, 0.0], Circle(1.0)))


class Skew(Distribution):
    """A model that is not centrally symmetric, so reflecting it is observable."""

    def sample(self, n, rng):
        return rng.exponential(1.0, size=(n, 2))

    def uncertainty_set(self, k):
        return Polygon(k * np.array([[1.0, 0.0], [-0.5, 0.8], [-0.5, -0.3]]))


@pytest.mark.parametrize("cls", [Gaussian, UniformEllipse, UniformBox])
def test_sample_covariance_matches_sigma(cls):
    r"""Gaussian($\Sigma$) and X.from_covariance($\Sigma$) draws have mean 0 and covariance $\Sigma$ (to 2%)."""
    dist = Gaussian(SIGMA) if cls is Gaussian else cls.from_covariance(SIGMA)
    x = dist.sample(200_000, np.random.default_rng(0))
    assert x.shape == (200_000, 2)
    assert np.allclose(x.mean(axis=0), 0.0, atol=0.05)
    assert np.allclose(np.cov(x.T), SIGMA, rtol=0.02, atol=0.02 * 25.0)


@pytest.mark.parametrize("cls", [UniformEllipse, UniformBox])
def test_uniform_samples_lie_in_body_one(cls):
    r"""uncertainty_set(1) is the support: every draw satisfies $d \cdot x \le h(d)$."""
    dist = cls.from_covariance(SIGMA)
    x, B = dist.sample(5000, np.random.default_rng(1)), dist.uncertainty_set(1.0)
    for d in _dirs(60):
        assert np.all(x @ d <= B.support(d) + 1e-9)


def test_uniform_ellipse_edge_is_gaussian_two_sigma_ellipse():
    r"""UniformEllipse.from_covariance($\Sigma$).uncertainty_set(1) has the support of covariance_ellipse($\Sigma$, 2)."""
    B, E = UniformEllipse.from_covariance(SIGMA).uncertainty_set(1.0), covariance_ellipse(SIGMA, 2.0)
    for d in _dirs():
        assert np.isclose(B.support(d), E.support(d), atol=1e-9)


def test_gaussian_body_is_covariance_ellipse():
    for d in _dirs():
        assert np.isclose(Gaussian(SIGMA).uncertainty_set(1.5).support(d), 1.5 * np.sqrt(d @ SIGMA @ d))


def test_uniform_box_from_covariance_half_widths():
    """A uniform interval of half-width h has variance $h^2/3$."""
    assert np.allclose(UniformBox.from_covariance(np.diag([3.0, 3.0])).half_widths, 3.0)


def test_sum_samples_add_and_bodies_minkowski_add():
    """a + b draws a then b from the same generator and adds; the supports add too."""
    a, b, c = Gaussian(np.diag([1.0, 0.4])), UniformBox([0.5, 0.5], theta=0.3), UniformEllipse(2.0, 1.0, 0.4)
    s = a + b + c
    assert isinstance(s, IndependentSum) and len(s.parts) == 3
    rng = np.random.default_rng(5)
    expect = a.sample(100, rng) + b.sample(100, rng) + c.sample(100, rng)
    assert np.array_equal(s.sample(100, np.random.default_rng(5)), expect)
    for d in _dirs():
        assert np.isclose(s.uncertainty_set(2.0).support(d),
                          a.uncertainty_set(2.0).support(d) + b.uncertainty_set(2.0).support(d) + c.uncertainty_set(2.0).support(d))



def test_reflect_is_identity_for_symmetric_models_and_a_true_reflection_otherwise():
    r"""``reflect()`` returns self for the shipped models; for a skewed one, $-D$ negates the
    draws and satisfies $h_{-D}(u) = h_D(-u)$."""
    for D in (Gaussian(SIGMA), UniformEllipse(2.0, 1.0, 0.4), UniformBox([0.5, 1.5], theta=0.3)):
        assert D.reflect() is D
    s = Gaussian(SIGMA) + UniformBox([1.0, 0.5])
    assert all(a is b for a, b in zip(s.reflect().parts, s.parts))

    D = Skew()
    R = D.reflect()
    assert isinstance(R, Reflected) and R.reflect() is D
    assert np.array_equal(R.sample(50, np.random.default_rng(1)),
                          -D.sample(50, np.random.default_rng(1)))
    for u in _dirs():
        assert np.isclose(R.uncertainty_set(2.0).support(u), D.uncertainty_set(2.0).support(-u))


def test_reflecting_a_distribution_or_its_body_gives_the_same_set():
    """Reflecting a Distribution or its uncertainty set gives the same result, and doing both cancels."""
    own, target = Skew(), UniformBox([0.4, 0.2], theta=0.2)
    at_distribution = (target + own.reflect()).uncertainty_set(2.0)
    at_shape = MinkowskiSum(target.uncertainty_set(2.0), own.uncertainty_set(2.0).reflect())
    unreflected = MinkowskiSum(target.uncertainty_set(2.0), own.uncertainty_set(2.0))
    reflected_twice = MinkowskiSum(target.uncertainty_set(2.0), own.reflect().uncertainty_set(2.0).reflect())
    for u in _dirs():
        assert np.isclose(at_distribution.support(u), at_shape.support(u))
        assert np.isclose(reflected_twice.support(u), unreflected.support(u))
    assert any(not np.isclose(at_shape.support(u), unreflected.support(u)) for u in _dirs())


def test_minimal_subclass_works_in_sample_cpa():
    """A Distribution implementing only sample() drives sample_cpa; uncertainty_set() is left unimplemented."""
    class Ring(Distribution):
        def __init__(self, r):
            self.r = r

        def sample(self, n, rng):
            phi = rng.uniform(0, 2 * np.pi, n)
            return self.r * np.column_stack([np.cos(phi), np.sin(phi)])

    own, target = _encounter()
    s = sample_cpa(own, target, pos_noise=Ring(2.0), vel_noise=Ring(0.1) + Gaussian(0.01 * np.eye(2)),
                   n=1000, rng=np.random.default_rng(0))
    assert np.allclose(np.linalg.norm(s.relpos - (target.pos - own.pos), axis=1), 2.0)
    assert s.tcpa.shape == (1000,) and np.isfinite(s.dcpa).all()
    with pytest.raises(NotImplementedError):
        Ring(1.0).uncertainty_set(1.0)


def test_zero_noise_equals_nominal_cpa():
    own, target = _encounter()
    s = sample_cpa(own, target, n=256, rng=np.random.default_rng(0))
    tcpa, dcpa = cpa(target.pos - own.pos, target.vel - own.vel)
    assert np.all(s.tcpa == tcpa) and np.all(s.dcpa == dcpa)


def test_sample_cpa_shapes_signed_dcpa_and_consistency():
    """Outputs are (n,) and (n, 2), the metrics are cpa() of the returned states, and DCPA takes both signs."""
    own, target = _encounter()
    s = sample_cpa(own, target, Gaussian(np.eye(2)), Gaussian(0.1 * np.eye(2)), n=1000,
                   rng=np.random.default_rng(1))
    assert s.tcpa.shape == s.dcpa.shape == (1000,) and s.relpos.shape == s.relvel.shape == (1000, 2)
    t, d = cpa(s.relpos, s.relvel)
    assert np.array_equal(t, s.tcpa) and np.array_equal(d, s.dcpa)
    assert s.dcpa.min() < 0.0 < s.dcpa.max()


def test_sample_cpa_seed_reproducible_and_draw_order():
    """Same seed, same draws; positions are drawn before velocities."""
    own, target = _encounter()
    P, V = Gaussian(np.eye(2)), UniformBox([0.2, 0.3])
    a = sample_cpa(own, target, P, V, n=500, rng=np.random.default_rng(0))
    b = sample_cpa(own, target, P, V, n=500, rng=np.random.default_rng(0))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    rng = np.random.default_rng(0)
    dp, dv = P.sample(500, rng), V.sample(500, rng)
    assert np.array_equal(a.relpos, target.pos - own.pos + dp)
    assert np.array_equal(a.relvel, target.vel - own.vel + dv)


def test_small_noise_mean_near_nominal():
    own, target = _encounter()
    tcpa, dcpa = cpa(target.pos - own.pos, target.vel - own.vel)
    s = sample_cpa(own, target, Gaussian(1e-3 * np.eye(2)), Gaussian(1e-3 * np.eye(2)), n=20000,
                   rng=np.random.default_rng(7))
    assert abs(s.tcpa.mean() - tcpa) < 0.2 and abs(s.dcpa.mean() - dcpa) < 0.2


def test_velocity_only_noise_spreads_both_metrics():
    own, target = _encounter()
    s = sample_cpa(own, target, vel_noise=Gaussian(0.25 * np.eye(2)), n=5000, rng=np.random.default_rng(7))
    assert s.dcpa.std() > 0.0 and s.tcpa.std() > 0.0
    assert np.all(s.relpos == target.pos - own.pos)
