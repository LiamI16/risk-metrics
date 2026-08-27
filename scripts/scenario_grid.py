"""Every encounter in a catalog as (TCPA, DCPA) small multiples on one shared scale.

Flags are independent; each combination writes its own PNG to artifacts/comparison:

    (none)                   scenario_grid.png
    --constant-cov           scenario_grid_constant_cov.png
    --symlog                 scenario_grid_symlog.png
    --constant-cov --symlog  scenario_grid_constant_cov_symlog.png

--constant-cov  Fix (sigma_pos, sigma_vel) at FIXED_SIGMA_* for every scenario, rather
                than anchoring sigma_vel to each one's own closing speed.
--symlog        Symmetric-log TCPA axis, and nothing is excluded from scale-setting.
                On the linear axis, low-closing-speed encounters (see
                vo_utils.scales.is_degenerate) do not set the limits and overflow the
                frame; the panel then reports the fraction off frame.

Each panel: green/red samples (red = inside |d| <= R with tau >= 0), P = that fraction,
black dot = nominal CPA, * = low closing speed.

Run:  python3 scripts/scenario_grid.py [--constant-cov] [--symlog]
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so vo_utils imports

import numpy as np
import matplotlib.pyplot as plt

from vo_utils import CANONICAL_SCENARIOS, uniform_like, sample_cpa
from vo_utils.plotting import _apply_style
from vo_utils.scales import compute_scales, is_degenerate

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "comparison"
SEED = 0
NCOLS = 5
FIXED_SIGMA_POS = 5.0    # [m]
FIXED_SIGMA_VEL = 1.0    # [m/s]
LINTHRESH = 10.0                                        # [s]
SYMLOG_TICKS = (-1000, -100, -10, 0, 10, 100, 1000)     # [s]

NOISE_MODELS = {
    "gaussian":        lambda Sp, Sv: (Sp, Sv),
    "uniform_ellipse": lambda Sp, Sv: (uniform_like(Sp, "ellipse"), uniform_like(Sv, "ellipse")),
    "uniform_box":     lambda Sp, Sv: (uniform_like(Sp, "box"), uniform_like(Sv, "box")),
}


def constant_cov(scenarios, sigma_pos=FIXED_SIGMA_POS, sigma_vel=FIXED_SIGMA_VEL):
    """Retune ``scenarios`` so every one carries the same (Sigma_pos, Sigma_vel).

    Parameters
    ----------
    scenarios : sequence of Scenario
        Usually ``CANONICAL_SCENARIOS``.
    sigma_pos, sigma_vel : float
        Position [m] and velocity [m/s] standard deviations to impose on every scenario.

    Returns
    -------
    tuple of Scenario
    """
    out = []
    for sc in scenarios:
        own, target = sc.ships()
        R = own.domain.r + target.domain.r
        out.append(replace(sc, alpha=sigma_pos / R, vel_scale=sigma_vel / sc.beta))
    return tuple(out)


def panel(ax, sc, scales, model="gaussian", symlog=False):
    """Draw one scenario's (TCPA, DCPA) cloud on the shared frame.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    sc : Scenario
        The encounter to sample and draw.
    scales : Scales
        Catalog-wide limits from ``compute_scales``; supplies ``tau_lim`` and ``d_lim``.
    model : {'gaussian', 'uniform_ellipse', 'uniform_box'}
        Which noise shape to draw at the scenario's covariance.
    symlog : bool
        Put TCPA on a symmetric-log axis (see module docstring).
    """
    own, target = sc.ships()
    pos, vel = NOISE_MODELS[model](*sc.covariances())
    s = sample_cpa(own, target, pos, vel, n=sc.n, rng=np.random.default_rng(SEED))
    R = own.domain.r + target.domain.r
    tau, d = s.tcpa, s.dcpa
    hit = (np.abs(d) <= R) & (tau >= 0) & (True if sc.horizon is None else tau <= sc.horizon)

    (xlo, xhi), ylim = scales.tau_lim, scales.d_lim
    off = float(((tau < xlo) | (tau > xhi) | (np.abs(d) > ylim)).mean())

    ax.fill_between([0, xhi], -R, R, facecolor="#d62728", alpha=0.07, lw=0, zorder=0)
    ax.axvline(0.0, color="k", lw=0.7, ls="--", alpha=0.45)
    for y in (R, -R):
        ax.axhline(y, color="#d62728", lw=0.8, ls="--", alpha=0.7)
    ax.scatter(tau[~hit], d[~hit], s=1.5, alpha=0.10, color="#2ca02c", lw=0)
    ax.scatter(tau[hit], d[hit], s=1.5, alpha=0.18, color="#d62728", lw=0)
    m = sc.metrics()
    ax.plot(m.TCPA(), m.DCPA(), "o", color="k", ms=3.5, zorder=6)

    if symlog:
        ax.set_xscale("symlog", linthresh=LINTHRESH, linscale=0.8)
        ax.set_xticks([t for t in SYMLOG_TICKS if xlo <= t <= xhi])
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
        ax.set_xticks([], minor=True)
    ax.set_xlim(xlo, xhi); ax.set_ylim(-ylim, ylim)
    flag = "  *" if is_degenerate(sc) else ""
    sub = f"P={hit.mean():.2f}" + (f",  {off:.0%} off" if off > 0.005 else "")
    ax.set_title(f"{sc.name}{flag}\n{sub}", fontsize=9)
    ax.tick_params(labelsize=7)


def figure(scenarios, scales, model="gaussian", constant=False, symlog=False):
    """Lay the whole catalog out as a grid of ``panel`` calls, one per scenario.

    ``constant`` and ``symlog`` only affect the annotation and the axis type; the
    scenarios and scales handed in must already reflect them.
    """
    nrows = int(np.ceil(len(scenarios) / NCOLS))
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(3.1 * NCOLS, 3.0 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, sc in zip(axes, scenarios):
        panel(ax, sc, scales, model, symlog)
    for ax in axes[len(scenarios):]:
        ax.axis("off")
    for i, ax in enumerate(axes[:len(scenarios)]):
        if i % NCOLS == 0:
            ax.set_ylabel("DCPA [m]", fontsize=9)
        if i >= len(scenarios) - NCOLS:            # bottom of its own column, not of the grid
            ax.set_xlabel("TCPA [s]" + (" (symlog)" if symlog else ""), fontsize=9)
            ax.tick_params(labelbottom=True)       # sharex hides these on a ragged last row
    noise = (f"sigma_pos = {FIXED_SIGMA_POS:g} m, sigma_vel = {FIXED_SIGMA_VEL:g} m/s "
             "held constant" if constant else "noise anchored per scenario")
    star = ("* = low closing speed (heavy-tailed TCPA)" if symlog else
            "* = excluded from scale-setting (low closing speed)")
    fig.suptitle(f"(TCPA, DCPA) across all {len(scenarios)} encounters, one shared scale "
                 f"({model}; {noise})\n"
                 r"band = $|d| \leq R$;  " + star, fontsize=12.5)
    fig.tight_layout()
    return fig


def main():
    args = sys.argv[1:]
    constant, symlog = "--constant-cov" in args, "--symlog" in args
    OUT.mkdir(parents=True, exist_ok=True)
    _apply_style()
    scenarios = constant_cov(CANONICAL_SCENARIOS) if constant else CANONICAL_SCENARIOS
    scales = compute_scales(scenarios, NOISE_MODELS, seed=SEED, skip_degenerate=not symlog)
    fig = figure(scenarios, scales, constant=constant, symlog=symlog)
    stem = "scenario_grid" + ("_constant_cov" if constant else "") + ("_symlog" if symlog else "")
    path = OUT / f"{stem}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    main()
