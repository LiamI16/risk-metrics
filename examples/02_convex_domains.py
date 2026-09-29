"""Velocity obstacles for polygon and ellipse agents, and why equal covariance hides the model.

A velocity obstacle is the set of own-ship velocities that lead to a collision if both ships
hold their course, so a safe course is one outside it. Uncertainty inflates it: K_p is the
error set on the relative position, K_v on the relative velocity.

A Gaussian and a uniform ellipse of the same covariance give the same K_v, so they give the
same uncertainty-aware VO. Example 03 samples the two and shows what that set-based view
cannot see.

Run from the repo root:  python examples/02_convex_domains.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_metrics import (
    Ellipse, Gaussian, MinkowskiSum, Polygon, Ship, UncertaintyAwareVelocityObstacle, UniformEllipse,
    VelocityObstacle, config_space_obstacle, cpa, sample_cpa,
)

OUT = Path(__file__).resolve().parent.parent / "output" / "examples" / "02_convex_domains.png"
LIGHT_RED, MID_RED = "#fbe9e9", "#f2bebe"       # opaque tints

if __name__ == "__main__":
    plt.style.use(Path(__file__).resolve().parent / "risk_metrics.mplstyle")

    # 1. A ship's domain is the area it occupies, its hull or a safety region around it,
    # given in the ship's own frame (m). It can be any convex Shape.
    hull = Polygon([(0.0, 3.0), (1, 1), (1, -2), (-1, -2), (-1, 1)])
    own = Ship([0.0, 0.0], [0.0, 5.0], domain=hull)
    target = Ship([0.0, 70.0], [0.0, -6.0], domain=Ellipse(3.0, 2.0, theta=np.deg2rad(20)))

    # 2. TCPA/DCPA use the reference positions only, so they ignore the domains entirely and
    # work for any shape. (The scalar collision test, in example 03, needs circles.)
    tcpa, dcpa = cpa(target.pos - own.pos, target.vel - own.vel)
    print(f"TCPA = {tcpa:.2f} s, DCPA = {dcpa:.2f} m")

    # 3. Configuration-space obstacle $C = D_{target} \oplus (-D_{own})$, placed at relpos.
    C = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
    vo = VelocityObstacle(own, target)
    print(f"own velocity in VO: {vo.contains(own.vel)}")

    # 4. Uncertainty: K_p inflates the obstacle C, K_v grows the VO out in velocity space.
    # Covariances are of the relative state: $\Sigma_{own} + \Sigma_{target}$.
    Sigma_p = np.array([[30.0, 12.0], [12.0, 20.0]])    # m^2
    Sigma_v = np.array([[0.5, 0.15], [0.15, 0.3]])      # (m/s)^2
    k = 2.0
    K_p = Gaussian(Sigma_p).uncertainty_set(k)
    uvo = UncertaintyAwareVelocityObstacle(own, target, K_p=K_p, K_v=Gaussian(Sigma_v).uncertainty_set(k))

    # 5. A uniform ellipse of equal covariance: its support, uncertainty_set(1), is the Gaussian's $2\sigma$ ellipse,
    # so the two UVOs coincide. The difference between the models shows up only in the samples.
    unif_v = UniformEllipse.from_covariance(Sigma_v)
    uvo_unif = UncertaintyAwareVelocityObstacle(own, target, K_p=K_p, K_v=unif_v.uncertainty_set(1))
    # K_p widens the cone; K_v only offsets its edges and rounds the apex.
    print(f"cone half-angle: VO {np.degrees(vo.half_angle):.1f} deg, "
          f"UVO {np.degrees(uvo.half_angle):.1f} deg")
    for name, u in (("Gaussian", uvo), ("uniform ellipse", uvo_unif)):
        print(f"UVO with {name} K_v contains own velocity: {u.contains(own.vel)}, "
              f"contains (2.5, 0): {u.contains([2.5, 0.0])}")

    # 6. Sampling: noise models can differ per channel.
    s = sample_cpa(own, target, Gaussian(Sigma_p), unif_v, n=20000, rng=np.random.default_rng(0))
    q = lambda a: np.percentile(a, [5, 50, 95])
    print("Gaussian position + uniform-ellipse velocity noise, 5/50/95th percentiles:")
    print("  TCPA [s]:", np.round(q(s.tcpa), 2), "  DCPA [m]:", np.round(q(s.dcpa), 2))

    # 7. Figure. (a) Relative position space (target relative to own); the cone edges are
    # directions of -relvel = v_own - v_target. (b) Own-velocity space.
    fig, (ax_p, ax_v) = plt.subplots(1, 2, figsize=(13, 6.3))
    L = 1000.0                                  # edge length drawn; the axes clip it
    for S, fill, ls, label in ((MinkowskiSum(C, K_p), 0.0, ":", rf"$C \oplus K_p$ (${k:g}\sigma$)"),
                               (C, 0.28, "-", "config obstacle $C$")):
        ring = S.sample_boundary(300)
        ax_p.fill(*ring.T, fc="tab:red", alpha=fill, lw=0)
        ax_p.plot(*np.vstack([ring, ring[:1]]).T, color="tab:red", lw=1.5 if ls == "-" else 0.8, ls=ls, label=label)
    for u, fc, ls, label in ((uvo, LIGHT_RED, "--", r"cone over $C \oplus K_p$"),     # widest first
                             (vo, MID_RED, "-", "cone over $C$")):
        ax_p.fill(*np.array([(0, 0), L * u.edges[0], L * u.edges[1]]).T, fc=fc, ec="tab:red", lw=1.1, ls=ls,
                  zorder=0, label=label)
        ax_p.plot(*u.tangent_points.T, "o", mfc="white", color="tab:red", ms=6, ls="none", zorder=6,
                  label="tangent points" if u is vo else None)
    ax_p.plot(0, 0, "o", color="tab:blue", ms=9, label="own ship (origin)")
    ax_p.set(xlim=(-45, 45), ylim=(-5, 85), aspect="equal", xlabel="East [m]", ylabel="North [m]")
    ax_p.set_title("(a) relative space: obstacle + collision cones", fontsize=11.5)
    ax_p.legend(loc="lower left", fontsize=8.5)

    ax_v.fill(*uvo.outline(L).T, fc=LIGHT_RED, ec="tab:red", lw=0.9, label=rf"UVO, Gaussian $K_v$ (${k:g}\sigma$)")
    ax_v.plot(*uvo_unif.outline(L).T, color="tab:purple", lw=1.6, ls="--", label="UVO, uniform-ellipse $K_v$ (support)")
    ax_v.plot(*vo.outline(L).T, color="k", lw=1.2, label="deterministic VO")
    ax_v.plot(0, 0, "+", color="k", ms=10)
    ax_v.annotate("", xy=target.vel, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2))
    ax_v.plot(*target.vel, "s", color="tab:blue", ms=7, label=r"$v_{target}$ (apex)")
    hit = vo.contains(own.vel)
    c = "tab:red" if hit else "tab:green"
    ax_v.annotate("", xy=own.vel, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
    ax_v.plot(*own.vel, "o", color=c, ms=9, label=r"$v_{own}$ " + ("(collision)" if hit else "(safe)"))
    ax_v.set(xlim=(-9, 9), ylim=(-11.5, 6.5), aspect="equal",          # room below for the legend
             xlabel=r"$v_{East}$ [m/s]", ylabel=r"$v_{North}$ [m/s]")
    ax_v.set_title("(b) velocity space: VO and uncertainty-aware VOs", fontsize=11.5)
    ax_v.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print("saved", OUT)
