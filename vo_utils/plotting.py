"""Visualization for velocity obstacles (matplotlib isolated to this module).

Draws collision cones / velocity obstacles.  Reuses minkowski_utils.plotting for
the config-space obstacle boundary.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as _Polygon

from minkowski_utils import MinkowskiSum
from minkowski_utils.plotting import plot_shape
from .cone import collision_cone
from .obstacle import VelocityObstacle, config_space_obstacle
from .uncertainty import probabilistic_collision_cone, covariance_ellipse

_BIG = 1e6   # wedge/edge extent; clipped to the axes at render time


def _apply_style():
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "grid.linestyle": "--",
                         "axes.axisbelow": True, "figure.dpi": 150})


def plot_collision_cone(cone, ax, apex=(0.0, 0.0),
                        color="#d62728", alpha=0.15, lw=1.5, label=None):
    """Draw a collision cone from ``apex``: shaded half-infinite wedge + edges.

    The cone is unbounded, so both the shaded wedge and its supporting lines are
    drawn oversized and clipped to the axes at render time (via add_artist /
    axline), which spans the plot without disturbing autoscaling.  Works at the
    origin (relative-velocity space) or translated to a target velocity (the VO).
    """
    apex = np.asarray(apex, dtype=float)
    if cone.contains_origin:
        ax.annotate("collision cone = every direction", apex, color=color,
                    fontsize=9, ha="center")
        ax.plot(*apex, "o", color=color, ms=6)
        return ax

    e1, e2 = cone.edges
    wedge = _Polygon([apex, apex + e1 * _BIG, apex + e2 * _BIG], closed=True,
                     facecolor=color, edgecolor="none", alpha=alpha, label=label)
    ax.add_artist(wedge)                     # clips to axes, no autoscale effect
    for e in (e1, e2):
        ax.axline(tuple(apex), tuple(apex + e), color=color, lw=lw)
    return ax


def plot_velocity_obstacle(vo, ax, v_own=None):
    """Draw the velocity obstacle: cone at apex = v_target, plus velocities."""
    c_safe, c_hit, c_tgt = "#2ca02c", "#d62728", "#1f77b4"
    plot_collision_cone(vo.cone, ax, apex=vo.apex, label="velocity obstacle")
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


def plot_probabilistic_cone(sigma_cones, ax, apex=(0.0, 0.0),
                            color="#d62728", alpha=0.13):
    """Draw nested k-sigma collision cones as a probability heatmap.

    Cones are drawn widest-first so the overlapping fills build up alpha toward
    the narrow, high-probability core; each level's edges are labeled ``k s``.
    """
    apex = np.asarray(apex, dtype=float)
    for sc in sorted(sigma_cones, key=lambda s: -s.k):
        plot_collision_cone(sc.cone, ax, apex=apex, color=color,
                            alpha=alpha, lw=1.1, label=fr"${sc.k:g}\sigma$")
    ax.plot(*apex, "o", color=color, ms=5, zorder=6)
    return ax


def deterministic_vo_figure(own, target):
    """Two-panel VO figure for an encounter: relative space + velocity space."""
    _apply_style()
    O = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
    cone = collision_cone(O)
    vo = VelocityObstacle(O, target.vel)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6.3))

    # Panel A: relative-position space -- obstacle + tangent (collision) cone
    plot_shape(O, ax=axA, n=300, alpha=0.15, color="#d62728", label="config obstacle $O$")
    plot_collision_cone(cone, axA, apex=(0, 0), alpha=0.08)
    axA.plot(cone.contacts[:, 0], cone.contacts[:, 1], "o", mfc="white",
             color="#d62728", ms=6, zorder=6)
    axA.plot(0, 0, "o", color="#1f77b4", ms=9, label="own ship (origin)")
    axA.set_title("(a) relative space: obstacle + collision cone", fontsize=11.5)
    axA.set_xlabel("East [m]"); axA.set_ylabel("North [m]")
    axA.legend(loc="lower left", fontsize=8.5)
    axA.set_aspect("equal", adjustable="datalim")

    # Panel B: velocity space -- the velocity obstacle
    plot_velocity_obstacle(vo, axB, v_own=own.vel)
    axB.set_title("(b) velocity space: velocity obstacle", fontsize=11.5)
    axB.set_xlabel(r"$v_{East}$ [m/s]"); axB.set_ylabel(r"$v_{North}$ [m/s]")
    axB.legend(loc="upper right", fontsize=8.5)
    axB.set_aspect("equal", adjustable="datalim")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def probabilistic_vo_figure(own, target, Sigma, k_levels=(1.0, 2.0, 3.0)):
    """Two-panel probabilistic VO figure: nested k-sigma cones (level sets)."""
    _apply_style()
    O = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
    scones = probabilistic_collision_cone(O, Sigma, k_levels)

    fig, (axC, axD) = plt.subplots(1, 2, figsize=(13, 6.3))

    # Panel C: relative space -- inflated obstacles + nested cones
    plot_shape(O, ax=axC, n=300, alpha=0.28, color="#d62728", label="obstacle $O$")
    for k in k_levels:
        Ok = MinkowskiSum(O, covariance_ellipse(Sigma, k))
        plot_shape(Ok, ax=axC, n=300, fill=False, lw=0.8, color="#d62728", ls=":")
    plot_probabilistic_cone(scones, axC, apex=(0, 0))
    axC.plot(0, 0, "o", color="#1f77b4", ms=9, label="own ship")
    axC.set_title("(a) relative space: k-sigma-inflated obstacles + cones", fontsize=11.5)
    axC.set_xlabel("East [m]"); axC.set_ylabel("North [m]")
    axC.legend(loc="lower left", fontsize=8.5)
    axC.set_aspect("equal", adjustable="datalim")

    # Panel D: velocity space -- nested VO cones (probability level sets)
    plot_probabilistic_cone(scones, axD, apex=target.vel)
    axD.plot(0, 0, "+", color="k", ms=10)
    axD.annotate("", xy=target.vel, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2))
    axD.plot(*target.vel, "s", color="#1f77b4", ms=7, label=r"$v_{target}$")
    axD.annotate("", xy=own.vel, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=2))
    axD.plot(*own.vel, "o", color="k", ms=8, label=r"$v_{own}$")
    axD.set_title("(b) velocity space: probability level sets", fontsize=11.5)
    axD.set_xlabel(r"$v_{East}$ [m/s]"); axD.set_ylabel(r"$v_{North}$ [m/s]")
    axD.legend(loc="upper right", fontsize=8.5)
    axD.set_aspect("equal", adjustable="datalim")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
