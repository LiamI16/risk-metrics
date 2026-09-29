"""Monte Carlo (TCPA, DCPA) and P[collision] under Gaussian vs. uniform-ellipse noise at equal covariance.

TCPA is the time until two ships are nearest and DCPA how far apart they are then. DCPA is
signed by which side the target passes on: positive when it passes to the left of the relative
track, the direction the target moves as seen from the own ship. So the scatter below spans
both signs, and with R the two ships' radii added, a collision is |DCPA| <= R with TCPA >= 0.

Same crossing encounter and noise as the paper's stochastic-comparison figure, which uses
n = 5000 and seed 0. This script draws more samples, so its P[collision] can differ from
that figure's in the second decimal.

Run from the repo root:  python examples/03_uncertainty_sampling.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from risk_metrics import CROSSING, Gaussian, UniformEllipse, collides, cpa, sample_cpa

OUT = Path(__file__).resolve().parent.parent / "output" / "examples" / "03_uncertainty_sampling.png"
N = 20000
SEED = 0

if __name__ == "__main__":
    plt.style.use(Path(__file__).resolve().parent / "risk_metrics.mplstyle")

    # 1. Encounter and noise. Noise is on the relative state, so $\Sigma = \Sigma_{own} + \Sigma_{target}$.
    sc = CROSSING                                       # ships on perpendicular tracks
    own, target = sc.ships()
    R = sc.R
    sigma_pos, sigma_vel = 5.0, 0.5                     # m, m/s, isotropic
    Sigma_p = sigma_pos ** 2 * np.eye(2)
    Sigma_v = sigma_vel ** 2 * np.eye(2)
    tcpa0, dcpa0 = cpa(target.pos - own.pos, target.vel - own.vel)
    print(f"{sc.name}: nominal TCPA = {tcpa0:.1f} s, DCPA = {dcpa0:.1f} m, R = {R:.0f} m")

    # 2. Two models with the same covariance: the uniform ellipse's edge is the $2\sigma$ ellipse,
    # so it has no tails.
    models = {
        "Gaussian": (Gaussian(Sigma_p), Gaussian(Sigma_v)),
        "uniform ellipse": (UniformEllipse.from_covariance(Sigma_p),
                            UniformEllipse.from_covariance(Sigma_v)),
    }

    # 3. Sample and score. The same seed makes each run reproducible, but it does not pair the
    # draws across models, which consume the generator differently. The two P[collision]
    # values below differ by about 0.010; changing the seed moves that difference by about
    # 0.003, so it is a real effect of the model rather than sampling noise.
    iqr = lambda a: np.subtract(*np.percentile(a, [75, 25]))
    results = {}
    for name, (pos_noise, vel_noise) in models.items():
        s = sample_cpa(own, target, pos_noise, vel_noise, n=N, rng=np.random.default_rng(SEED))
        hit = collides(s.relpos, s.relvel, R)
        print(f"  {name:>15}: P[collision] = {hit.mean():.3f}, "
              f"TCPA IQR = {iqr(s.tcpa):.2f} s, DCPA IQR = {iqr(s.dcpa):.2f} m")
        results[name] = (s, hit)

    # 4. Figure: one joint plot per model, (TCPA, DCPA) scatter with marginal histograms,
    # on limits pooled over both models so the panels compare directly.
    tau_all = np.concatenate([s.tcpa for s, _ in results.values()])
    d_all = np.concatenate([s.dcpa for s, _ in results.values()])
    xlo, xhi = np.percentile(tau_all, [0.5, 99.5])
    xlo, xhi = xlo - 0.05 * (xhi - xlo), xhi + 0.05 * (xhi - xlo)
    ymax = 1.05 * np.percentile(np.abs(d_all), 99.5)
    fig = plt.figure(figsize=(14, 6.3))
    for sub, (name, (s, hit)) in zip(fig.subfigures(1, 2, wspace=0.04), results.items()):
        gs = sub.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.04, hspace=0.04)
        ax = sub.add_subplot(gs[1, 0])
        ax_top = sub.add_subplot(gs[0, 0], sharex=ax)
        ax_right = sub.add_subplot(gs[1, 1], sharey=ax)
        # The band is the collision region, $|DCPA| \leq R$. Its other condition, TCPA $\geq 0$,
        # holds for every sample here, so the band's left edge at TCPA = 0 sits off the axes.
        ax.add_patch(Rectangle((0.0, -R), xhi, 2 * R, fc="tab:red", alpha=0.07, lw=0, zorder=0))
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.3)
        for y in (R, -R):
            ax.axhline(y, color="tab:red", lw=1.0, ls="--", alpha=0.7, label=f"$|d| = R = {R:g}$" if y > 0 else None)
        ax.scatter(s.tcpa[~hit], s.dcpa[~hit], s=3, alpha=0.15, color="tab:green", lw=0, label="safe sample")
        ax.scatter(s.tcpa[hit], s.dcpa[hit], s=3, alpha=0.25, color="tab:red", lw=0, label="collision sample")
        ax.plot(tcpa0, dcpa0, "o", color="k", ms=5, zorder=6, label="deterministic CPA")
        ax.set(xlim=(xlo, xhi), ylim=(-ymax, ymax), xlabel="TCPA [s]", ylabel="DCPA [m]")
        leg = ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        for h in leg.legend_handles[1:3]:
            h.set_sizes([20]); h.set_alpha(1)
        ax_top.hist(s.tcpa, bins=80, range=(xlo, xhi), density=True, color="tab:gray", alpha=0.65)
        ax_top.axvline(tcpa0, color="k", ls=":", lw=1.0)
        ax_top.tick_params(labelbottom=False); ax_top.set_yticks([])
        ax_right.hist(s.dcpa, bins=80, range=(-ymax, ymax), density=True, orientation="horizontal",
                      color="tab:gray", alpha=0.65)
        ax_right.axhline(dcpa0, color="k", ls=":", lw=1.0)
        ax_right.tick_params(labelleft=False); ax_right.set_xticks([])
        ax_top.set_title(rf"{name}   |   $P[\mathrm{{collision}}]={hit.mean():.3f}$", fontsize=11.5)
    fig.suptitle(rf"{sc.name}: $\sigma_{{pos}}$ = {sigma_pos:g} m, $\sigma_{{vel}}$ = {sigma_vel:g} m/s, "
                 f"n = {N}, same covariance", fontsize=12.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print("saved", OUT)
