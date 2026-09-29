"""cpa() and collides() against each other, brute force, and the named scenarios."""

import numpy as np
import pytest

from risk_metrics import cpa, collides, CANONICAL_SCENARIOS


def _draws(n, seed):
    r"""Random (relpos, relvel, R); about a fifth start with $\|relpos\| \le R$."""
    rng = np.random.default_rng(seed)
    R = rng.uniform(5.0, 30.0, n)
    ang = rng.uniform(0, 2 * np.pi, (2, n))
    dist = np.where(rng.uniform(size=n) < 0.2, rng.uniform(0, 1, n) * R, rng.uniform(0, 200, n))
    speed = rng.uniform(0, 10, n)
    relpos = (dist * [np.cos(ang[0]), np.sin(ang[0])]).T
    relvel = (speed * [np.cos(ang[1]), np.sin(ang[1])]).T
    return relpos, relvel, R


def test_cpa_head_on_example():
    """A target closing head-on from 141 m at 7.1 m/s: TCPA 20 s, DCPA 0."""
    tcpa, dcpa = cpa([100.0, 100.0], [-5.0, -5.0])
    assert np.isclose(tcpa, 20.0) and np.isclose(dcpa, 0.0)


def test_cpa_zero_relvel():
    r"""relvel = 0: TCPA = 0 and DCPA = $\|relpos\|$."""
    tcpa, dcpa = cpa([3.0, 4.0], [0.0, 0.0])
    assert tcpa == 0.0 and dcpa == 5.0


def test_signed_dcpa_follows_relative_track_not_heading():
    """Own at 5 m/s north; DCPA is signed by the relative track, not by the own heading.

    A target 10 m east of the own track gives +10 whether it is met head-on or overtaken, and
    -10 when it overtakes from astern; 10 m west of the track gives -10.
    """
    own_vel = np.array([0.0, 5.0])
    head_on = cpa([10.0, 100.0], np.array([0.0, -5.0]) - own_vel)       # target ahead, opposite course
    overtaken = cpa([10.0, 100.0], np.array([0.0, 2.0]) - own_vel)      # slower target ahead
    overtaking = cpa([10.0, -100.0], np.array([0.0, 8.0]) - own_vel)    # faster target from astern
    for tcpa, dcpa, expected in (head_on + (10.0,), overtaken + (10.0,), overtaking + (-10.0,)):
        assert tcpa > 0 and np.isclose(dcpa, expected)
    assert np.isclose(cpa([-10.0, 100.0], np.array([0.0, -5.0]) - own_vel)[1], -10.0)


def test_cpa_broadcasts():
    tcpa, dcpa = cpa([0.0, 100.0], np.zeros((3, 4, 2)) + [0.0, -1.0])
    assert tcpa.shape == dcpa.shape == (3, 4)


def test_collides_matches_cpa_characterization_infinite_horizon():
    r"""collides == (tcpa > 0 & $|dcpa| \le R$) | (tcpa <= 0 & $\|r\| \le R$)."""
    r, v, R = _draws(20000, seed=1)
    tcpa, dcpa = cpa(r, v)
    rn = np.linalg.norm(r, axis=1)
    # drop exact ties, where the quadratic and the CPA rule can round either way
    keep = (np.abs(np.abs(dcpa) - R) > 1e-6 * R) & (np.abs(rn - R) > 1e-6 * R) & (np.abs(tcpa) > 1e-9)
    assert keep.mean() > 0.99
    expect = ((tcpa > 0) & (np.abs(dcpa) <= R)) | ((tcpa <= 0) & (rn <= R))
    got = collides(r, v, R)
    assert np.array_equal(got[keep], expect[keep])
    assert 0.1 < expect.mean() < 0.9            # both outcomes well represented


def test_collides_finite_horizon_matches_brute_force_and_cpa():
    r"""With horizon $\tau$, collides matches a dense t-grid minimum and the three-branch CPA rule."""
    r, v, R = _draws(10000, seed=2)
    tau = 25.0
    t = np.linspace(0.0, tau, 1001)
    grid_min = np.concatenate([
        np.linalg.norm(r[i:i + 1000, None] + v[i:i + 1000, None] * t[:, None], axis=2).min(axis=1)
        for i in range(0, len(r), 1000)])
    tcpa, dcpa = cpa(r, v)
    t_star = np.clip(tcpa, 0.0, tau)
    exact_min = np.linalg.norm(r + v * t_star[:, None], axis=1)
    speed = np.linalg.norm(v, axis=1)
    # drop what the grid cannot resolve (closest approach within one step of R) and exact ties
    keep = ((np.abs(exact_min - R) > speed * (t[1] - t[0]) + 1e-6 * R)
            & (np.abs(tcpa) > 1e-9) & (np.abs(tcpa - tau) > 1e-9))
    assert keep.mean() > 0.95
    got = collides(r, v, R, horizon=tau)
    assert np.array_equal(got[keep], (grid_min <= R)[keep])

    rn = np.linalg.norm(r, axis=1)
    r_tau = np.linalg.norm(r + v * tau, axis=1)
    three = (((tcpa > 0) & (tcpa <= tau) & (np.abs(dcpa) <= R))
             | ((tcpa <= 0) & (rn <= R)) | ((tcpa > tau) & (r_tau <= R)))
    assert np.array_equal(got[keep], three[keep])
    # the horizon removes some infinite-horizon collisions and adds none
    inf = collides(r, v, R)
    assert np.all(inf[got]) and (inf & ~got).sum() > 100


def test_collides_simple_cases():
    assert collides([100.0, 100.0], [-5.0, -5.0], R=20.0)
    assert not collides([100.0, 100.0], [-5.0, -5.0], R=20.0, horizon=10.0)
    assert collides([10.0, 0.0], [0.0, 0.0], R=20.0)       # already overlapping, at rest


# Nominal (TCPA s, signed DCPA m) as written in the comments of scenarios.py.
NOMINAL = {
    "head_on": (20.0, 0.0), "head_on_faster": (15.4, 0.0), "crossing": (20.0, 0.0),
    "crossing_wide": (20.0, -141.4), "north_east": (16.0, -89.4), "overtaking": (30.0, 0.0),
    "collinear_slower": (100.0, 0.0), "collinear_equal": (0.0, 200.0),
    "collinear_falling_behind": (-100.0, 0.0), "diverging": (-10.0, 0.0),
    "stationary": (40.0, 0.0), "low_closing": (240.0, 2.0), "beam": (0.0, 50.0),
}


@pytest.mark.parametrize("sc", CANONICAL_SCENARIOS, ids=lambda s: s.name)
def test_scenario_comments_match_cpa(sc):
    """The (TCPA, DCPA) in each scenario's comment agrees with cpa() to its rounding."""
    own, target = sc.ships()
    tcpa, dcpa = cpa(target.pos - own.pos, target.vel - own.vel)
    assert np.allclose([tcpa, dcpa], NOMINAL[sc.name], atol=0.05)


def test_all_scenarios_listed():
    assert {s.name for s in CANONICAL_SCENARIOS} == set(NOMINAL)
