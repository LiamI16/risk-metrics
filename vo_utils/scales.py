"""Common plot scales shared across a whole catalog of scenarios.

``compute_scales`` does one pass over a scenario x noise-model catalog and returns a
single :class:`Scales`, so color and axis extents mean the same thing in every figure.

Two kinds of shared scale, which are not the same thing:

* WIDTHS (``vel_half``, ``pos_half``) -- the velocity and position panels are centered
  per scenario, so what is shared is the span, applied around each figure's own center.
* ABSOLUTE EXTENTS (``tau_lim``, ``d_lim``) -- zero is meaningful on both axes of the
  (TCPA, DCPA) panel, so those limits are identical everywhere. With |d| <= R at a fixed
  width, the collision fraction is comparable between scenarios by eye.

Color limits are computed on the BINNED values the panels draw, not the raw samples: a
bin median has far less spread, so a raw-sample percentile leaves every panel washed out.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from .sampling import sample_cpa

GRIDSIZE = 40                # hexbin resolution; must match _plot_vo_metric's default
COLOR_Q = 99.0               # percentile for symmetric color limits, over pooled bins
EXTENT_Q = (0.5, 99.5)       # per-scenario robust extent rule, as in _plot_cpa_joint
FRAME_Q = (0.5, 99.5)        # per-scenario cloud extent for the spatial panels
FRAME_PAD = 1.2              # margin so the k=3 confidence ring clears the frame
DEGENERATE_SNR = 3.0         # velocity_snr below this = heavy-tailed TCPA


def velocity_snr(scenario):
    """Signal-to-noise ratio of the relative-velocity estimate: ||v_rel|| / sigma_vel.

    Parameters
    ----------
    scenario : Scenario

    Returns
    -------
    float
        ``inf`` when the velocity channel is exact (no noise to be swamped by).
    """
    own, target = scenario.ships()
    v_rel = np.asarray(target.vel, float) - np.asarray(own.vel, float)
    closing_speed = float(np.linalg.norm(v_rel))

    _, Sigma_vel = scenario.covariances()
    # Per-axis sigma of an isotropic 2x2 covariance: trace = sigma_x^2 + sigma_y^2.
    sigma_vel = float(np.sqrt(np.trace(np.asarray(Sigma_vel, float)) / 2.0))

    return np.inf if sigma_vel == 0 else closing_speed / sigma_vel


def is_degenerate(scenario, snr=DEGENERATE_SNR):
    """True if the encounter's closing speed is small next to its velocity noise.

    Below this ratio the relative-velocity DIRECTION is poorly determined, so
    TCPA = -(r.v)/|v|^2 picks up Cauchy tails orders of magnitude wider than a
    well-conditioned encounter's. Such scenarios are excluded from scale-setting --
    pooling them flattens everything else to a vertical line -- but are still rendered on
    the resulting scale, overflowing it, with the panel reporting the fraction off frame.

    Note the test is only meaningful when sigma_vel is set independently of the closing
    speed. Anchoring it to that speed makes the ratio a constant, identical for every
    encounter, and nothing is ever selected.

    Parameters
    ----------
    scenario : Scenario
    snr : float
        Threshold on :func:`velocity_snr`.

    Returns
    -------
    bool
    """
    return velocity_snr(scenario) < snr


def iqr(a):
    """Interquartile range -- the spread reduction for the binned metric panels."""
    a = np.asarray(a, dtype=float)
    return float(np.subtract(*np.percentile(a, [75, 25]))) if a.size else 0.0


def _binned(x, y, values, reduce, gridsize=GRIDSIZE):
    """The per-bin reduction hexbin would draw, without drawing it.

    Detached Figure, not pyplot, so nothing touches global figure state.
    """
    ax = Figure().subplots()
    hb = ax.hexbin(x, y, C=values, reduce_C_function=reduce, gridsize=gridsize, mincnt=1)
    return np.asarray(hb.get_array(), dtype=float)


@dataclass(frozen=True)
class Scales:
    """Catalog-wide plot limits. See module docstring for widths vs absolute extents."""

    dcpa: float                     # symmetric color limit, DCPA median panel [m]
    tcpa: float                     # symmetric color limit, TCPA median panel [s]
    spread_dcpa: float              # upper color limit, DCPA IQR panel [m]
    spread_tcpa: float              # upper color limit, TCPA IQR panel [s]
    vel_half: float                 # half-WIDTH of the velocity-frame panels [m/s]
    pos_half: float                 # half-WIDTH of the position panels [m]
    tau_lim: tuple[float, float]    # absolute TCPA extent, common (tau, d) panel [s]
    d_lim: float                    # absolute symmetric DCPA extent, same panel [m]


def compute_scales(scenarios, noise_models, seed=0, gridsize=GRIDSIZE,
                   skip_degenerate=True):
    """Pool one catalog into a single :class:`Scales`.

    Parameters
    ----------
    scenarios : sequence of Scenario
        Usually ``CANONICAL_SCENARIOS``.
    noise_models : dict
        ``name -> f(Sigma_pos, Sigma_vel) -> (pos_spec, vel_spec)``. Pooling across all
        models keeps the matched-covariance Gaussian/uniform variants comparable.
    seed : int
        Must match the render seed, so the limits describe the figures that get drawn.
    skip_degenerate : bool
        Exclude low-relative-speed encounters from scale-setting -- see
        :func:`is_degenerate`. They are still rendered on the resulting scale.

    Returns
    -------
    Scales
    """
    if skip_degenerate:
        scenarios = [sc for sc in scenarios if not is_degenerate(sc)]
        if not scenarios:
            raise ValueError("every scenario is degenerate; pass skip_degenerate=False")

    med_d, med_t, iqr_d, iqr_t = [], [], [], []
    tau_lo, tau_hi, d_ext, vel_half, pos_half = [], [], [], [], []

    for make_specs in noise_models.values():
        for sc in scenarios:
            own, target = sc.ships()
            pos_spec, vel_spec = make_specs(*sc.covariances())
            s = sample_cpa(own, target, pos_spec, vel_spec, n=sc.n,
                           rng=np.random.default_rng(seed))

            vown = np.asarray(target.vel, dtype=float) - s.relvel
            for arr, values in ((med_d, s.dcpa), (med_t, s.tcpa)):
                arr.append(_binned(vown[:, 0], vown[:, 1], values, np.median, gridsize))
            for arr, values in ((iqr_d, s.dcpa), (iqr_t, s.tcpa)):
                arr.append(_binned(vown[:, 0], vown[:, 1], values, iqr, gridsize))

            # Per-scenario robust extents; the catalog takes the widest of them.
            lo, hi = np.percentile(s.tcpa, EXTENT_Q)
            tau_lo.append(float(lo)); tau_hi.append(float(hi))
            d_ext.append(float(np.percentile(np.abs(s.dcpa), EXTENT_Q[1])))
            vel_half.append(_half_width(vown))
            pos_half.append(_half_width(s.relpos))

    pool = lambda parts: np.concatenate([p[np.isfinite(p)] for p in parts])
    tau_span = (min(tau_lo), max(tau_hi))
    pad = 0.05 * (tau_span[1] - tau_span[0] + 1e-9)

    return Scales(
        dcpa=float(np.percentile(np.abs(pool(med_d)), COLOR_Q)),
        tcpa=float(np.percentile(np.abs(pool(med_t)), COLOR_Q)),
        spread_dcpa=float(np.percentile(pool(iqr_d), COLOR_Q)),
        spread_tcpa=float(np.percentile(pool(iqr_t), COLOR_Q)),
        vel_half=max(vel_half),
        pos_half=max(pos_half),
        tau_lim=(tau_span[0] - pad, tau_span[1] + pad),
        d_lim=max(d_ext) * 1.05,
    )


def _half_width(points):
    """Half the larger axis span of a robust bounding box around ``points``."""
    lo, hi = np.percentile(np.asarray(points, dtype=float), FRAME_Q, axis=0)
    return float(np.max(hi - lo)) / 2.0 * FRAME_PAD
