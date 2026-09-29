"""All canonical scenarios as (TCPA, DCPA) small multiples on one shared scale.

TCPA is the time until two ships are nearest (negative if that is past) and DCPA how far
apart they are then, signed by which side the target passes on: positive when it passes to
the left of the relative track, the direction the target moves as seen from the own ship.
The sign only records the side; the separation itself is |DCPA|. R, per scenario, is the two
ships' radii added: a collision is |DCPA| <= R with TCPA >= 0, the red band in every panel.

Two ships holding the same velocity have no relative motion and so no track direction. The
sign is then undefined and DCPA is the current separation, as in collinear_equal below.

Run from the repo root:  python examples/04_scenario_grid.py [--constant-cov] [--symlog]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_metrics import CANONICAL_SCENARIOS, Gaussian, collides, cpa, sample_cpa

OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "examples"
N, SEED, NCOLS = 20000, 0, 5
ALPHA, BETA = 0.25, 0.10          # sigma_pos = ALPHA * R, sigma_vel = BETA * closing speed
# $\|v_{rel}\|$ is zero or tiny here, so anchoring $\sigma_{vel}$ to it would give no velocity noise.
VEL_SCALE_OVERRIDE = {"collinear_equal": 5.0, "low_closing": 5.0}     # m/s
CONSTANT_SIGMA_POS, CONSTANT_SIGMA_VEL = 5.0, 1.0                     # m, m/s
MIN_SNR = 3.0                     # $\|v_{rel}\| / \sigma_{vel}$ below this: heavy-tailed TCPA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--constant-cov", action="store_true",
                        help=f"sigma_pos = {CONSTANT_SIGMA_POS:g} m, sigma_vel = "
                             f"{CONSTANT_SIGMA_VEL:g} m/s for every scenario")
    parser.add_argument("--symlog", action="store_true",
                        help="symmetric-log TCPA axis; no scenario is excluded from the limits")
    args = parser.parse_args()
    plt.style.use(Path(__file__).resolve().parent / "risk_metrics.mplstyle")

    # 1. Noise per scenario, isotropic on the relative state, then sample.
    runs = []
    for sc in CANONICAL_SCENARIOS:
        own, target = sc.ships()
        v_rel = np.linalg.norm(target.vel - own.vel)
        if args.constant_cov:
            sigma_pos, sigma_vel = CONSTANT_SIGMA_POS, CONSTANT_SIGMA_VEL
        else:
            sigma_pos = ALPHA * sc.R
            sigma_vel = BETA * VEL_SCALE_OVERRIDE.get(sc.name, v_rel)
        s = sample_cpa(own, target, Gaussian(sigma_pos ** 2 * np.eye(2)),
                       Gaussian(sigma_vel ** 2 * np.eye(2)), n=N, rng=np.random.default_rng(SEED))
        # When the closing speed is small next to its noise, the direction of v_rel is
        # poorly known and TCPA $= -(r \cdot v)/\|v\|^2$ gets very heavy tails.
        low_closing = v_rel / sigma_vel < MIN_SNR
        runs.append((sc, own, target, s, collides(s.relpos, s.relvel, sc.R), low_closing))

    # 2. Shared limits, pooled over the scenarios allowed to set them.
    setters = [r for r in runs if args.symlog or not r[5]]
    tau_lo = min(np.percentile(r[3].tcpa, 0.5) for r in setters)
    tau_hi = max(np.percentile(r[3].tcpa, 99.5) for r in setters)
    pad = 0.05 * (tau_hi - tau_lo)
    tau_lim = (tau_lo - pad, tau_hi + pad)
    d_lim = 1.05 * max(np.percentile(np.abs(r[3].dcpa), 99.5) for r in setters)
    print(f"shared limits: TCPA {tau_lim[0]:.1f} to {tau_lim[1]:.1f} s, |DCPA| <= {d_lim:.1f} m")

    # 3. One panel per scenario.
    nrows = int(np.ceil(len(runs) / NCOLS))
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(3.1 * NCOLS, 3.0 * nrows),
                             sharex=True, sharey=True)
    axes = axes.ravel()
    for i, (ax, (sc, own, target, s, hit, low_closing)) in enumerate(zip(axes, runs)):
        R = sc.R
        off = ((s.tcpa < tau_lim[0]) | (s.tcpa > tau_lim[1]) | (np.abs(s.dcpa) > d_lim)).mean()
        print(f"  {sc.name:>25}{' *' if low_closing else '  '}  P[collision] = {hit.mean():.3f}"
              f"  off frame = {off:.1%}")
        ax.fill_between([0, tau_lim[1]], -R, R, fc="tab:red", alpha=0.07, lw=0, zorder=0)
        ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.45)
        for y in (R, -R):
            ax.axhline(y, color="tab:red", lw=0.8, ls="--", alpha=0.7)
        ax.scatter(s.tcpa[~hit], s.dcpa[~hit], s=1.5, alpha=0.10, lw=0, color="tab:green")
        ax.scatter(s.tcpa[hit], s.dcpa[hit], s=1.5, alpha=0.18, lw=0, color="tab:red")
        ax.plot(*cpa(target.pos - own.pos, target.vel - own.vel), "o", color="k", ms=3.5, zorder=6)
        if args.symlog:
            ax.set_xscale("symlog", linthresh=10.0, linscale=0.8)
            ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}".replace("-", "\N{MINUS SIGN}"))
        sub = f"P={hit.mean():.2f}" + (f",  {off:.0%} off" if off > 0.005 else "")
        ax.set_title(f"{sc.name}{'  *' if low_closing else ''}\n{sub}", fontsize=9)
        ax.tick_params(labelsize=7)
        if i % NCOLS == 0:
            ax.set_ylabel("DCPA [m]", fontsize=9)
        if i >= len(runs) - NCOLS:            # bottom of its own column; sharex hides a ragged row's labels
            ax.set_xlabel("TCPA [s]" + (" (symlog)" if args.symlog else ""), fontsize=9)
            ax.tick_params(labelbottom=True)
    for ax in axes[len(runs):]:
        ax.axis("off")
    axes[0].set_xlim(*tau_lim)
    axes[0].set_ylim(-d_lim, d_lim)

    noise = (rf"$\sigma_{{pos}}$ = {CONSTANT_SIGMA_POS:g} m, $\sigma_{{vel}}$ = {CONSTANT_SIGMA_VEL:g} m/s held constant"
             if args.constant_cov
             else rf"$\sigma_{{pos}}$ = {ALPHA:g} R, $\sigma_{{vel}}$ = {BETA:g} $\times$ closing speed")
    star = "low closing speed (heavy-tailed TCPA)" if args.symlog else "excluded from scale-setting (low closing speed)"
    fig.suptitle(f"(TCPA, DCPA) across all {len(runs)} scenarios, one shared scale (Gaussian; {noise}; n = {N})\n"
                 rf"band = $|d| \leq R$, TCPA $\geq 0$ (red = collision);  black = nominal;  * = {star}",
                 fontsize=12.5)
    fig.tight_layout()

    stem = "04_scenario_grid" + ("_constant_cov" if args.constant_cov else "") + \
        ("_symlog" if args.symlog else "")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png")
    print("saved", OUT_DIR / f"{stem}.png")
