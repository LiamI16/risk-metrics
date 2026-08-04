"""Visualization for velocity obstacles (matplotlib isolated to this module).

Draws collision cones / velocity obstacles.  Reuses minkowski_utils.plotting for
the config-space obstacle boundary.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as _Polygon, Rectangle as _Rectangle

from minkowski_utils import MinkowskiSum, Circle
from minkowski_utils.plotting import plot_shape
from .cone import collision_cone
from .metrics import Metrics
from .obstacle import VelocityObstacle, config_space_obstacle
from .uncertainty import probabilistic_collision_cone, covariance_ellipse
from .velocity_uncertainty import robust_velocity_obstacle
from .sampling import sample_cpa

_BIG = 1e6   # wedge/edge extent; clipped to the axes at render time


def _apply_style():
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "grid.linestyle": "--",
                         "axes.axisbelow": True, "figure.dpi": 150})


def plot_collision_cone(cone, ax, apex=(0.0, 0.0),
                        color="#d62728", alpha=0.15, lw=1.5, label=None):
    """Draw a collision cone from ``apex``: shaded half-infinite wedge + edges.

    The cone is unbounded, so the wedge and its edges are drawn oversized and
    clipped to the axes at render time, without disturbing autoscaling.
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
    ax.add_artist(wedge)                     # clips to axes without autoscaling
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


def plot_robust_velocity_obstacle(rvo, ax, color="#d62728", alpha=0.15, lw=1.5,
                                  label=None):
    """Draw the velocity-uncertainty-inflated VO: offset parallel edges + cap.

    The fill is drawn oversized and clipped to the axes; the boundary is the two
    offset edges plus the rounded apex arc. Axis limits are set by the caller.
    """
    if rvo.contains_origin:
        ax.annotate("inflated cone = every direction", rvo.apex, color=color,
                    fontsize=9, ha="center")
        ax.plot(*rvo.apex, "o", color=color, ms=6)
        return ax

    (p1, p2), (e1, e2) = rvo.contacts, rvo.edges
    verts = np.vstack([p1 + e1 * _BIG, rvo.cap, p2 + e2 * _BIG])
    ax.add_artist(_Polygon(verts, closed=True, facecolor=color, edgecolor="none",
                           alpha=alpha, label=label))
    for p, e in ((p1, e1), (p2, e2)):
        ax.plot([p[0], p[0] + e[0] * _BIG], [p[1], p[1] + e[1] * _BIG],
                color=color, lw=lw)
    ax.plot(rvo.cap[:, 0], rvo.cap[:, 1], color=color, lw=lw)
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


def _plot_position_space(ax, own, target, Sigma_pos=None, k_levels=(1.0, 2.0, 3.0)):
    """Relative/position space: config obstacle O and its collision cone.

    With Sigma_pos, the nested k-sigma-inflated obstacles and their widened cones
    are drawn (position uncertainty lives here); otherwise the deterministic cone.
    """
    O = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
    plot_shape(O, ax=ax, n=300, alpha=0.28, color="#d62728", label="obstacle $O$")
    if Sigma_pos is not None:
        for k in k_levels:
            Ok = MinkowskiSum(O, covariance_ellipse(Sigma_pos, k))
            plot_shape(Ok, ax=ax, n=300, fill=False, lw=0.8, color="#d62728", ls=":")
        plot_probabilistic_cone(probabilistic_collision_cone(O, Sigma_pos, k_levels),
                                ax, apex=(0, 0))
    else:
        plot_collision_cone(collision_cone(O), ax, apex=(0, 0), alpha=0.08)
    ax.plot(0, 0, "o", color="#1f77b4", ms=9, label="own ship")

    # Relative-velocity arrow from the origin: collision iff it points into the cone.
    w = np.asarray(own.vel, dtype=float) - np.asarray(target.vel, dtype=float)
    wn = np.linalg.norm(w)
    if wn > 1e-9:
        reach = 0.35 * np.linalg.norm(np.asarray(target.pos, dtype=float)
                                      - np.asarray(own.pos, dtype=float))
        tip = w / wn * max(reach, 1.0)
        ax.annotate("", xy=tip, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2))
        ax.update_datalim([(0, 0), tip])         # keep arrow in frame
        ax.plot([], [], color="#1f77b4", lw=2, label=r"rel. velocity $v_O - v_T$")

    # Deterministic CPA readout
    m = Metrics(own, target)
    ax.text(0.98, 0.02, f"TCPA = {m.TCPA():.1f} s\nDCPA = {m.DCPA():.1f} m\n"
            fr"$R_{{safe}}$ = {m.safety_radius:.0f} m", transform=ax.transAxes,
            fontsize=9, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.9))

    ax.set_title("position space", fontsize=11.5)
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_aspect("equal", adjustable="datalim")


def _plot_velocity_space(ax, own, target, Sigma_vel=None, k_levels=(1.0, 2.0, 3.0)):
    """Velocity space: the VO, inflated by nested k-sigma velocity uncertainty."""
    O = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
    vo = VelocityObstacle(O, target.vel)

    frame = [(0.0, 0.0), tuple(vo.apex), tuple(own.vel), tuple(target.vel)]
    # A degenerate (near-zero) velocity covariance has no inflation to draw and its
    # support point is undefined, so fall back to the deterministic cone.
    has_vel = Sigma_vel is not None and np.trace(np.asarray(Sigma_vel, dtype=float)) > 1e-12
    if has_vel:
        rvos = [(k, robust_velocity_obstacle(vo, covariance_ellipse(Sigma_vel, k)))
                for k in k_levels]
        for k, rvo in sorted(rvos, key=lambda t: -t[0]):   # widest first
            plot_robust_velocity_obstacle(rvo, ax, alpha=0.13, lw=0.9,
                                          label=fr"VO $\oplus\ {k:g}\sigma$")
            if not rvo.contains_origin:
                frame += rvo.cap.tolist()
    else:
        plot_collision_cone(vo.cone, ax, apex=vo.apex, label="velocity obstacle")

    _annotate_velocity_space(ax, own, target)
    _square_limits((ax,), frame)
    ax.set_title("velocity space", fontsize=11.5)
    ax.set_xlabel(r"$v_{East}$ [m/s]"); ax.set_ylabel(r"$v_{North}$ [m/s]")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_aspect("equal", adjustable="box")


def _plot_cpa_joint(subfig, own, target, Sigma_pos=None, Sigma_vel=None,
                    n=20000, R=None, horizon=None, rng=None):
    """The (TCPA, DCPA) sample distribution jointplot, drawn into a SubFigure.

    n Monte-Carlo CPA samples as a joint scatter with marginal histograms, colored
    by membership of the collision corner {DCPA <= R, TCPA >= 0} (optionally TCPA <=
    horizon) so P[collision] is the red fraction. The deterministic CPA is marked
    for reference.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    m = Metrics(own, target)
    R = m.safety_radius if R is None else R

    s = sample_cpa(own, target, Sigma_pos, Sigma_vel, n=n, rng=rng)
    tau, d = s.tcpa, s.dcpa
    hit = (d <= R) & (tau >= 0) & (True if horizon is None else tau <= horizon)
    p_hit = float(hit.mean())

    # Robust limits -- TCPA tails can be enormous near low closing speed.
    xlo, xhi = np.percentile(tau, [0.5, 99.5])
    yhi = float(np.percentile(d, 99.5))
    xpad = 0.05 * (xhi - xlo + 1e-9)
    xlo, xhi, ylo, yhi = xlo - xpad, xhi + xpad, 0.0, yhi * 1.05

    gs = subfig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                             wspace=0.04, hspace=0.04)
    ax = subfig.add_subplot(gs[1, 0])
    ax_top = subfig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = subfig.add_subplot(gs[1, 1], sharey=ax)

    # Collision corner {d <= R, tau >= 0}
    ax.add_patch(_Rectangle((0.0, 0.0), xhi, R, facecolor="#d62728", alpha=0.07,
                            edgecolor="none", zorder=0))
    ax.axvline(0.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(R, color="#d62728", lw=1.0, ls="--", alpha=0.7, label=f"$R = {R:g}$")

    # Joint scatter, colored by collision membership
    ax.scatter(tau[~hit], d[~hit], s=3, alpha=0.15, color="#2ca02c", lw=0,
               label="safe sample")
    ax.scatter(tau[hit], d[hit], s=3, alpha=0.25, color="#d62728", lw=0,
               label="collision sample")
    ax.plot(m.TCPA(), m.DCPA(), "o", color="k", ms=5, zorder=6,
            label="deterministic CPA")

    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel("TCPA [s]"); ax.set_ylabel("DCPA [m]")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Marginals: density histograms + deterministic reference line
    for data, axm, orient, det in ((tau, ax_top, "v", m.TCPA()),
                                   (d, ax_right, "h", m.DCPA())):
        lo, hi = (xlo, xhi) if orient == "v" else (ylo, yhi)
        if orient == "v":
            axm.hist(data, bins=80, range=(lo, hi), density=True,
                     color="#7f7f7f", alpha=0.65)
            axm.axvline(det, color="k", ls=":", lw=1.0)
            axm.tick_params(labelbottom=False); axm.set_yticks([])
        else:
            axm.hist(data, bins=80, range=(lo, hi), density=True,
                     orientation="horizontal", color="#7f7f7f", alpha=0.65)
            axm.axhline(det, color="k", ls=":", lw=1.0)
            axm.tick_params(labelleft=False); axm.set_xticks([])

    ax_top.set_title(
        rf"$(\mathrm{{TCPA}}, \mathrm{{DCPA}})$ distribution   |   "
        rf"$P[\mathrm{{collision}}]={p_hit:.3f}$", fontsize=11.5)


def three_panel(own, target, Sigma_pos=None, Sigma_vel=None, n=20000, R=None,
                horizon=None, k_levels=(1.0, 2.0, 3.0), rng=None, title=None):
    """Three panels for one uncertain encounter: position, velocity, (TCPA, DCPA).

    own, target:          the two Ships.
    Sigma_pos, Sigma_vel: relative 2x2 position/velocity covariances; None = exact.
    n:                    number of Monte-Carlo samples for the distribution panel.
    R:                    collision-corner radius; defaults to the disc safety radius.
    horizon:              optional upper TCPA bound for the collision corner.
    k_levels:             confidence levels for the nested inflated obstacles/VOs.
    rng:                  numpy Generator for reproducibility.
    title:                optional figure suptitle (e.g. the scenario name).

    Left: position space (obstacle + collision cone, k-sigma position inflation).
    Middle: velocity space (VO inflated by k-sigma velocity uncertainty).
    Right: the (TCPA, DCPA) sample distribution.
    """
    _apply_style()
    fig = plt.figure(figsize=(19, 6.3))
    sf_pos, sf_vel, sf_joint = fig.subfigures(
        1, 3, width_ratios=(1.0, 1.0, 1.25), wspace=0.03)

    _plot_position_space(sf_pos.subplots(), own, target, Sigma_pos, k_levels)
    _plot_velocity_space(sf_vel.subplots(), own, target, Sigma_vel, k_levels)
    _plot_cpa_joint(sf_joint, own, target, Sigma_pos, Sigma_vel, n, R, horizon, rng)
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    return fig


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


def _annotate_velocity_space(ax, own, target):
    """Origin cross, target-velocity apex and own-velocity markers."""
    ax.plot(0, 0, "+", color="k", ms=10)
    ax.annotate("", xy=target.vel, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2))
    ax.plot(*target.vel, "s", color="#1f77b4", ms=7, label=r"$v_{target}$")
    ax.annotate("", xy=own.vel, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=2))
    ax.plot(*own.vel, "o", color="k", ms=8, label=r"$v_{own}$")


def _square_limits(axes, points, pad_frac=0.18):
    """Give every ax the same square limits framing ``points`` (equal aspect)."""
    P = np.asarray(points, dtype=float)
    lo, hi = P.min(axis=0), P.max(axis=0)
    c = 0.5 * (lo + hi)
    half = max(0.5 * float((hi - lo).max()) * (1 + pad_frac), 1e-6)
    for ax in axes:
        ax.set_xlim(c[0] - half, c[0] + half)
        ax.set_ylim(c[1] - half, c[1] + half)


def velocity_uncertainty_vo_figure(own, target, Sigma_v, k_levels=(1.0, 2.0, 3.0)):
    """Two-panel velocity-space figure: deterministic VO vs. nested velocity levels.

    Each level's inflated VO has edges parallel to the deterministic cone, offset
    outward with a rounded apex; the half-angle is unchanged (unlike position
    uncertainty, which widens the angle). Nested levels overlap toward the core.
    """
    _apply_style()
    O = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
    vo = VelocityObstacle(O, target.vel)
    rvos = [(k, robust_velocity_obstacle(vo, covariance_ellipse(Sigma_v, k))) for k in k_levels]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6.3))

    # Panel A: deterministic velocity obstacle
    plot_collision_cone(vo.cone, axA, apex=vo.apex, label="velocity obstacle")
    _annotate_velocity_space(axA, own, target)
    axA.set_title("(a) velocity space: deterministic VO", fontsize=11.5)

    # Panel B: nested velocity-uncertainty level sets (VO (+) k-sigma ellipse).
    # Largest k first so overlapping fills build toward the deterministic core.
    for k, rvo in sorted(rvos, key=lambda t: -t[0]):
        plot_robust_velocity_obstacle(rvo, axB, alpha=0.13, lw=0.9,
                                      label=fr"VO $\oplus\ {k:g}\sigma$")
    _annotate_velocity_space(axB, own, target)
    axB.set_title(r"(b) velocity space: nested $k\sigma_v$ level sets", fontsize=11.5)

    frame = [(0.0, 0.0), tuple(vo.apex), tuple(own.vel), tuple(target.vel)]
    for _, rvo in rvos:
        if not rvo.contains_origin:
            frame += rvo.cap.tolist()
    _square_limits((axA, axB), frame)

    for ax in (axA, axB):
        ax.set_xlabel(r"$v_{East}$ [m/s]"); ax.set_ylabel(r"$v_{North}$ [m/s]")
        ax.legend(loc="upper right", fontsize=8.5)
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def combined_uncertainty_vo_figure(own, target, Sigma_pos, Sigma_vel,
                                   k_pos_levels=(1.0, 2.0, 3.0),
                                   k_vel_levels=(1.0, 2.0, 3.0)):
    """Two-panel figure with nested position AND velocity uncertainty.

    (a) relative space: nested k_pos-sigma position-inflated obstacles and their
        widened collision cones (position uncertainty lives here).
    (b) velocity space: every (k_pos, k_vel) pair as an overlapping level set --
        the k_pos-widened cone further offset by the k_vel-sigma velocity ellipse
        (parallel edges + rounded apex). Overlapping fills build alpha toward the
        high-probability (small-k) core.
    """
    _apply_style()
    O = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
    scones = probabilistic_collision_cone(O, Sigma_pos, k_pos_levels)

    # The cone depends only on k_pos, so build it once per position level and
    # reuse across velocity levels.
    combined = []                                   # (k_pos, k_vel, rvo)
    for kp in k_pos_levels:
        vo_k = VelocityObstacle(MinkowskiSum(O, covariance_ellipse(Sigma_pos, kp)),
                                target.vel)
        for kv in k_vel_levels:
            combined.append((kp, kv, robust_velocity_obstacle(vo_k, covariance_ellipse(Sigma_vel, kv))))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6.3))

    # Panel A: relative space -- nested inflated obstacles + nested cones (position)
    plot_shape(O, ax=axA, n=300, alpha=0.28, color="#d62728", label="obstacle $O$")
    for kp in k_pos_levels:
        Ok = MinkowskiSum(O, covariance_ellipse(Sigma_pos, kp))
        plot_shape(Ok, ax=axA, n=300, fill=False, lw=0.8, color="#d62728", ls=":")
    plot_probabilistic_cone(scones, axA, apex=(0, 0))
    axA.plot(0, 0, "o", color="#1f77b4", ms=9, label="own ship")
    axA.set_title(r"(a) relative space: nested $k\sigma_p$ obstacles + cones",
                  fontsize=11.5)
    axA.set_xlabel("East [m]"); axA.set_ylabel("North [m]")
    axA.legend(loc="lower left", fontsize=8.5)
    axA.set_aspect("equal", adjustable="datalim")

    # Panel B: velocity space -- overlapping (k_pos, k_vel) level sets.
    # Draw largest first so overlapping alpha builds toward the small-k core.
    for kp, kv, rvo in sorted(combined, key=lambda t: -(t[0] + t[1])):
        plot_robust_velocity_obstacle(rvo, axB, alpha=0.11, lw=0.7)
    _annotate_velocity_space(axB, own, target)
    axB.set_title(r"(b) velocity space: nested $k\sigma_p \times k\sigma_v$ level sets",
                  fontsize=11.5)
    axB.set_xlabel(r"$v_{East}$ [m/s]"); axB.set_ylabel(r"$v_{North}$ [m/s]")
    axB.legend(loc="upper right", fontsize=8.5)

    frame = [(0.0, 0.0), tuple(target.vel), tuple(own.vel)]
    for _, _, rvo in combined:
        if not rvo.contains_origin:
            frame += rvo.cap.tolist()
    _square_limits((axB,), frame)
    axB.set_aspect("equal", adjustable="box")

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
