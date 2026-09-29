r"""Nested uncertainty-aware VOs: how position and velocity uncertainty each change the VO.

A velocity obstacle is the set of own-ship velocities that lead to a collision if both ships
hold their course, so a safe course is one outside it.

Position uncertainty $K_p$ widens the cone; velocity uncertainty $K_v$ offsets its edges
and rounds the apex without changing the opening angle. The nested $k\sigma$ sets are confidence
sets, not probability contours: in 2-D a $2\sigma$ ellipse holds 86% of the draws, not 95%,
and inflating the VO by it bounds the velocities that could collide rather than marking the
ones that collide with some probability.

Run from the repo root:  python examples/05_uncertainty_aware_vo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_metrics import (
    Ellipse, Gaussian, MinkowskiSum, Polygon, Ship, UncertaintyAwareVelocityObstacle,
    VelocityObstacle, config_space_obstacle,
)

OUT = Path(__file__).resolve().parent.parent / "output" / "examples" / "05_uncertainty_aware_vo.png"
LEVELS = (3.0, 2.0, 1.0)            # widest first, so each narrower set is drawn on top
# Opaque tints of tab:red, lightest for the widest set, so each legend swatch matches the plot.
SHADE = {3.0: "#fbe9e9", 2.0: "#f4c9c9", 1.0: "#ed9e9e"}
SHADE_BOTH = {(3.0, 3.0): "#fbe9e9", (3.0, 1.0): "#f5cfcf", (1.0, 3.0): "#f0b0b1", (1.0, 1.0): "#eb9394"}

if __name__ == "__main__":
    plt.style.use(Path(__file__).resolve().parent / "risk_metrics.mplstyle")

    # 1. Encounter: a polygon own ship meeting an ellipse-shaped target head on. A domain is
    # the area a ship occupies. Covariances are of the relative state.
    hull = Polygon([(0.0, 3.0), (1, 1), (1, -2), (-1, -2), (-1, 1)])
    own = Ship([0.0, 0.0], [0.0, 5.0], domain=hull)
    target = Ship([0.0, 70.0], [0.0, -6.0], domain=Ellipse(3.0, 2.0, theta=np.deg2rad(20)))
    pos_noise = Gaussian(np.array([[30.0, 12.0], [12.0, 20.0]]))    # m^2
    vel_noise = Gaussian(np.array([[0.5, 0.15], [0.15, 0.3]]))      # (m/s)^2
    # The config-space obstacle grows the target domain by the own domain, so the own ship
    # can be treated as a point and the pair collides when that point lands inside C.
    C = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
    vo = VelocityObstacle(own, target)

    # 2. Position uncertainty only: the cone over C + K_p widens with k.
    pos_only = {k: UncertaintyAwareVelocityObstacle(own, target, K_p=pos_noise.uncertainty_set(k)) for k in LEVELS}
    # 3. Velocity uncertainty only: same opening angle, offset edges and a rounded apex.
    vel_only = {k: UncertaintyAwareVelocityObstacle(own, target, K_v=vel_noise.uncertainty_set(k)) for k in LEVELS}
    # 4. Both channels, for every (k_p, k_v) pair.
    both = {(kp, kv): UncertaintyAwareVelocityObstacle(own, target, K_p=pos_noise.uncertainty_set(kp),
                                                       K_v=vel_noise.uncertainty_set(kv))
            for kp in (3.0, 1.0) for kv in (3.0, 1.0)}

    print(f"deterministic VO half-angle: {np.degrees(vo.half_angle):.1f} deg")
    for k in sorted(LEVELS):
        print(f"  {k:g} sigma: position-only half-angle {np.degrees(pos_only[k].half_angle):5.1f} deg, "
              f"velocity-only {np.degrees(vel_only[k].half_angle):5.1f} deg, "
              f"own velocity in VO: {vel_only[k].contains(own.vel)}")

    # 5. Figure. (a) relative space, own ship at the origin; (b), (c) own-velocity space.
    fig, (ax_p, ax_v, ax_b) = plt.subplots(1, 3, figsize=(19, 6.3))
    ring = C.sample_boundary(300)
    ax_p.fill(*ring.T, fc="tab:red", alpha=0.5, lw=0)
    ax_p.plot(*np.vstack([ring, ring[:1]]).T, color="tab:red", lw=1.5, label="obstacle $C$")
    L = 1000.0                                  # edge length drawn; the axes clip it
    for k, u in pos_only.items():
        ring = MinkowskiSum(C, pos_noise.uncertainty_set(k)).sample_boundary(300)
        ax_p.plot(*np.vstack([ring, ring[:1]]).T, color="tab:red", lw=0.8, ls=":",   # close the loop
                  label=r"$C \oplus K_p$" if k == max(pos_only) else None)
        cone = np.array([(0.0, 0.0), L * u.edges[0], L * u.edges[1]])
        ax_p.fill(*cone.T, fc=SHADE[k], lw=0.8, ec="tab:red", zorder=0, label=rf"${k:g}\sigma$")
    ax_p.plot(0, 0, "o", color="tab:blue", ms=9, label="own ship")
    ax_p.set(xlim=(-50, 50), ylim=(-5, 95), aspect="equal", xlabel="East [m]", ylabel="North [m]")
    ax_p.set_title(r"(a) relative space: nested $k\sigma_p$ obstacles + cones", fontsize=11.5)
    ax_p.legend(loc="lower left", fontsize=8.5)

    # (b) and (c) share square limits framing the origin, both velocities and every cap.
    pts = np.vstack([[0.0, 0.0], own.vel, target.vel] + [u.cap for u in [*vel_only.values(), *both.values()]])
    c, half = (pts.min(0) + pts.max(0)) / 2, 1.18 * (pts.max(0) - pts.min(0)).max() / 2
    for ax, sets, title in ((ax_v, vel_only, r"(b) velocity space: nested $k\sigma_v$ confidence sets"),
                            (ax_b, both, r"(c) velocity space: nested $k\sigma_p \times k\sigma_v$ confidence sets")):
        for key, u in sets.items():
            label = rf"VO $\oplus$ ${key:g}\sigma$" if ax is ax_v else rf"$k_p={key[0]:g}$, $k_v={key[1]:g}$"
            fc = SHADE[key] if ax is ax_v else SHADE_BOTH[key]
            ax.fill(*u.outline(L).T, fc=fc, lw=0.8, ec="tab:red", zorder=0, label=label)
        for u in sets.values():             # the (k_p, k_v) sets cross, so redraw every edge on top
            ax.plot(*u.outline(L).T, color="tab:red", lw=0.8, zorder=1)
        ax.plot(*vo.outline(L).T, color="k", lw=1.2, label="deterministic VO")
        ax.plot(0, 0, "+", color="k", ms=10)
        ax.annotate("", xy=target.vel, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2))
        ax.plot(*target.vel, "s", color="tab:blue", ms=7, label=r"$v_{target}$")
        ax.annotate("", xy=own.vel, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=2))
        ax.plot(*own.vel, "o", color="k", ms=8, label=r"$v_{own}$")
        ax.set(xlim=(c[0] - half, c[0] + half), ylim=(c[1] - half, c[1] + half), aspect="equal",
               xlabel=r"$v_{East}$ [m/s]", ylabel=r"$v_{North}$ [m/s]")
        ax.set_title(title, fontsize=11.5)
        ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print("saved", OUT)
