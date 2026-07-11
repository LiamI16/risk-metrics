"""Visualization for velocity obstacles (matplotlib isolated to this module).

Draws collision cones / velocity obstacles.  Reuses minkowski_utils.plotting for
the config-space obstacle boundary.
"""

import numpy as np
import matplotlib.pyplot as plt

from .obstacle import VelocityObstacle


def plot_collision_cone(cone, ax, apex=(0.0, 0.0), length=10.0,
                        color="#d62728", alpha=0.15, lw=1.5, label=None):
    """Draw a collision cone from ``apex`` along its two edge rays, shaded.

    Works for a cone at the origin (relative-velocity space) or translated to a
    target velocity (the velocity obstacle).
    """
    apex = np.asarray(apex, dtype=float)
    if cone.contains_origin:
        ax.annotate("collision cone = every direction", apex, color=color,
                    fontsize=9, ha="center")
        ax.plot(*apex, "o", color=color, ms=6)
        return ax

    e1, e2 = cone.edges
    tri = np.array([apex, apex + e1 * length, apex + e2 * length])
    ax.fill(tri[:, 0], tri[:, 1], color=color, alpha=alpha, lw=0, label=label)
    for e in (e1, e2):
        ax.plot([apex[0], apex[0] + e[0] * length],
                [apex[1], apex[1] + e[1] * length], "-", color=color, lw=lw)
    return ax


def plot_velocity_obstacle(vo, ax, length=10.0, v_own=None):
    """Draw the velocity obstacle: cone at apex = v_target, plus velocities."""
    c_safe, c_hit, c_tgt = "#2ca02c", "#d62728", "#1f77b4"
    plot_collision_cone(vo.cone, ax, apex=vo.apex, length=length,
                        label="velocity obstacle")
    ax.plot(0, 0, "+", color="k", ms=10)
    ax.annotate("", xy=vo.apex, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=c_tgt, lw=2))
    ax.plot(*vo.apex, "s", color=c_tgt, ms=7, label=r"$v_{target}$ (apex)")

    if v_own is not None:
        v_own = np.asarray(v_own, dtype=float)
        hit = vo.contains(v_own)
        c = c_hit if hit else c_safe
        ax.annotate("", xy=v_own, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
        ax.plot(*v_own, "o", color=c, ms=9,
                label=r"$v_{own}$ " + ("(collision)" if hit else "(safe)"))
    return ax


if __name__ == "__main__":
    from minkowski_utils import Circle, Ellipse, MinkowskiSum
    from minkowski_utils.plotting import plot_shape
    from .cone import collision_cone

    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "grid.linestyle": "--",
                         "axes.axisbelow": True, "figure.dpi": 150})

    # --- scenario: own = disc, target = oriented ELLIPSE domain -------------
    p_own, v_own = np.array([0.0, 0.0]), np.array([0.0, 5.0])
    p_tgt, v_tgt = np.array([0.0, 120.0]), np.array([0.0, -6.0])
    D_own = Circle(8.0)                                   # own footprint (origin)
    D_tgt = Ellipse(25.0, 10.0, theta=np.deg2rad(60))    # target ellipse domain
    relpos = p_tgt - p_own

    # Config-space obstacle O = D_own (+) (-D_tgt), translated to relpos by
    # Minkowski-summing a point (a zero-radius disc at relpos).
    O = MinkowskiSum(D_own, D_tgt.reflect(), Circle(0.0, center=relpos))
    cone = collision_cone(O)
    vo = VelocityObstacle(O, v_tgt)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6.3))

    # Panel A: relative-position space -- obstacle + tangent (collision) cone
    plot_shape(O, ax=axA, n=300, alpha=0.15, color="#d62728", label="config obstacle $O$")
    plot_collision_cone(cone, axA, apex=(0, 0), length=175, alpha=0.08)
    axA.plot(cone.contacts[:, 0], cone.contacts[:, 1], "o", mfc="white",
             color="#d62728", ms=6, zorder=6)
    axA.plot(0, 0, "o", color="#1f77b4", ms=9, label="own ship (origin)")
    axA.set_title("(a) relative space: obstacle + collision cone", fontsize=11.5)
    axA.set_xlabel("East [m]"); axA.set_ylabel("North [m]")
    axA.legend(loc="lower left", fontsize=8.5)
    axA.set_aspect("equal", adjustable="datalim")

    # Panel B: velocity space -- the velocity obstacle
    plot_velocity_obstacle(vo, axB, length=16, v_own=v_own)
    axB.set_title("(b) velocity space: velocity obstacle", fontsize=11.5)
    axB.set_xlabel(r"$v_{East}$ [m/s]"); axB.set_ylabel(r"$v_{North}$ [m/s]")
    axB.legend(loc="upper right", fontsize=8.5)
    axB.set_aspect("equal", adjustable="datalim")

    fig.suptitle("Velocity obstacle for an elliptical target domain", fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    from pathlib import Path
    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "vo_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "vo_ellipse_demo.png"
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)
