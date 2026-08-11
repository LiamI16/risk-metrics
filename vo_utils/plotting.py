"""Figures for collision cones and velocity obstacles (matplotlib lives only here)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as _Polygon, Rectangle as _Rectangle
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D

from minkowski_utils import MinkowskiSum, Circle, Shape
from minkowski_utils.plotting import plot_shape
from .cone import collision_cone
from .metrics import Metrics
from .obstacle import VelocityObstacle, config_space_obstacle
from .uncertainty import probabilistic_collision_cone, covariance_ellipse, SigmaCone
from .uncertainty_models import (
    robust_velocity_obstacle, UncertaintyModel, Gaussian,
)
from .sampling import sample_cpa

_BIG = 1e6   # wedge/edge extent; clipped to the axes at render time

# Metric-field colormaps: orange = positive, blue = negative, pale = near zero, in
# every panel. A single-signed field takes the matching arm of the diverging ramp.
# Midpoint is gray, not white, and the arms start pale-but-not-white, so neither is
# lost under the density alpha in _plot_vo_metric.
_DIVERGING = LinearSegmentedColormap.from_list(
    "vo_diverging", ["#0d366b", "#2a78d6", "#f0efec", "#e08a63", "#eb6834", "#8c2f10"])
_SEQ_POS = LinearSegmentedColormap.from_list(
    "vo_seq_pos", ["#f6d9cb", "#e08a63", "#eb6834", "#8c2f10"])
_SEQ_NEG = LinearSegmentedColormap.from_list(
    "vo_seq_neg", ["#cde2fb", "#8fbdf0", "#2a78d6", "#0d366b"])


def _apply_style():
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "grid.linestyle": "--",
                         "axes.axisbelow": True, "figure.dpi": 150})


# A channel's noise `spec` is None (exact), a 2x2 covariance, an UncertaintyModel,
# or a bare Shape. These five helpers resolve it uniformly.

def _is_active(spec):
    """True if the channel carries (non-degenerate) noise worth drawing."""
    if spec is None:
        return False
    if isinstance(spec, (UncertaintyModel, Shape)):
        return True
    return float(np.trace(np.asarray(spec, dtype=float))) > 1e-12


def _body(spec, k):
    """Confidence-body Shape at level k for a channel noise spec (None -> None)."""
    if spec is None:
        return None
    if isinstance(spec, UncertaintyModel):
        return spec.body(k)
    if isinstance(spec, Shape):
        return spec
    return covariance_ellipse(spec, k)   # 2x2 covariance


def _is_bounded(spec):
    """True if the spec has a hard support (one set) rather than k-sigma levels."""
    if isinstance(spec, Shape):
        return True
    if isinstance(spec, UncertaintyModel):
        return not isinstance(spec, Gaussian)
    return False


def _levels_for(spec, k_levels):
    """Levels worth drawing: all of them, or just (1,) for a bounded spec."""
    return (1.0,) if _is_bounded(spec) else k_levels


def _level_label(spec, k):
    """Legend text for a noise body at level k."""
    return "support" if _is_bounded(spec) else fr"${k:g}\sigma$"


def plot_collision_cone(cone, ax, apex=(0.0, 0.0),
                        color="#d62728", alpha=0.15, lw=1.5, ls="-", label=None):
    """Draw a collision cone from apex: shaded wedge + edges.

    Unbounded, so the wedge is drawn oversized and clipped to the axes at render
    time rather than autoscaled to.
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
        ax.axline(tuple(apex), tuple(apex + e), color=color, lw=lw, ls=ls)
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
                                  ls="-", label=None):
    """Draw an inflated VO: offset parallel edges + rounded apex cap.

    Fill is oversized and clipped to the axes; the caller sets the limits.
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
                color=color, lw=lw, ls=ls)
    ax.plot(rvo.cap[:, 0], rvo.cap[:, 1], color=color, lw=lw, ls=ls)
    return ax


def plot_probabilistic_cone(sigma_cones, ax, apex=(0.0, 0.0),
                            color="#d62728", alpha=0.13, spec=None):
    """Nested k-sigma cones as a probability heatmap.

    Widest first, so overlapping fills build alpha toward the narrow core.
    """
    apex = np.asarray(apex, dtype=float)
    for sc in sorted(sigma_cones, key=lambda s: -s.k):
        plot_collision_cone(sc.cone, ax, apex=apex, color=color,
                            alpha=alpha, lw=1.1, label=_level_label(spec, sc.k))
    ax.plot(*apex, "o", color=color, ms=5, zorder=6)
    return ax


def _plot_position_space(ax, own, target, Sigma_pos=None, k_levels=(1.0, 2.0, 3.0)):
    """Position space: config obstacle O and its collision cone.

    Sigma_pos adds the nested inflated obstacles and their widened cones.
    """
    O = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
    plot_shape(O, ax=ax, n=300, alpha=0.28, color="#d62728", label="obstacle $O$")
    if _is_active(Sigma_pos):
        scones = []
        for k in _levels_for(Sigma_pos, k_levels):
            Ok = MinkowskiSum(O, _body(Sigma_pos, k))
            plot_shape(Ok, ax=ax, n=300, fill=False, lw=0.8, color="#d62728", ls=":")
            scones.append(SigmaCone(k, collision_cone(Ok)))
        plot_probabilistic_cone(scones, ax, apex=(0, 0), spec=Sigma_pos)
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


def _draw_vo_levelsets(ax, own, target, Sigma_vel, k_levels,
                       fill_alpha=0.13, color="#d62728", lw=0.9, ls="-",
                       label=True, deterministic=False):
    """Draw the deterministic cone or nested robust-VO level sets -> (vo, frame_points).

    fill_alpha=0 outlines only, to overlay a hexbin without hiding it;
    deterministic=True adds the un-inflated cone as a bold baseline.
    """
    O = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
    vo = VelocityObstacle(O, target.vel)
    frame = [tuple(vo.apex), tuple(own.vel), tuple(target.vel)]
    if _is_active(Sigma_vel):
        rvos = [(k, robust_velocity_obstacle(vo, _body(Sigma_vel, k)))
                for k in _levels_for(Sigma_vel, k_levels)]
        for k, rvo in sorted(rvos, key=lambda t: -t[0]):           # widest first
            lbl = fr"VO $\oplus$ {_level_label(Sigma_vel, k)}" if label else None
            plot_robust_velocity_obstacle(rvo, ax, color=color, alpha=fill_alpha, lw=lw, ls=ls, label=lbl)
            if not rvo.contains_origin:
                frame += rvo.cap.tolist()
        if deterministic:
            plot_collision_cone(vo.cone, ax, apex=vo.apex, color=color, alpha=0.0,
                                lw=1.8, ls="-", label="deterministic VO" if label else None)
    else:
        plot_collision_cone(vo.cone, ax, apex=vo.apex, color=color,
                            alpha=0.15 if fill_alpha else 0.0,
                            label="velocity obstacle" if label else None)
    return vo, frame


def _plot_velocity_space(ax, own, target, Sigma_vel=None, k_levels=(1.0, 2.0, 3.0)):
    """Velocity space: the VO, inflated by nested k-sigma velocity uncertainty."""
    _, frame = _draw_vo_levelsets(ax, own, target, Sigma_vel, k_levels)
    _annotate_velocity_space(ax, own, target)
    _square_limits((ax,), frame + [(0.0, 0.0)])
    ax.set_title("velocity space", fontsize=11.5)
    ax.set_xlabel(r"$v_{East}$ [m/s]"); ax.set_ylabel(r"$v_{North}$ [m/s]")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_aspect("equal", adjustable="box")


def _plot_vo_metric(ax, own, target, samples, values, cbar_label, Sigma_vel=None,
                    k_levels=(1.0, 2.0, 3.0), reduce=np.median, cmap=None,
                    diverging=False, symmetric=False, div_cmap=_DIVERGING, gridsize=40,
                    alpha_floor=0.35):
    """VO level sets over the sampled velocities, hexbinned and colored by a metric.

    Each sample sits at the own-velocity it implies (v_target - v); each bin's color
    is ``reduce`` over its metric values -- median for the field, np.ptp / IQR for
    spread. Color limits are robust percentiles, so the Cauchy-tail bins near the
    apex don't wash out the map. Pass ``cmap`` to override the automatic choice.
    """
    vown = np.asarray(target.vel, dtype=float) - samples.relvel       # own-velocity frame
    hb = ax.hexbin(vown[:, 0], vown[:, 1], C=values, reduce_C_function=reduce,
                   gridsize=gridsize, mincnt=1, cmap=cmap, linewidths=0.0, zorder=1)

    # Sparse bins fade rather than being dropped: conditioned on v the metrics are
    # near-deterministic, so a 1-sample bin's color is sound and only its weight is
    # suspect. sqrt keeps the fade gentle near the median, or the Poisson scatter of
    # a flat density (the uniform models) speckles the field. Identical hexbin args
    # so the count bins line up one-for-one.
    counts = ax.hexbin(vown[:, 0], vown[:, 1], gridsize=gridsize, mincnt=1)
    cnt = np.asarray(counts.get_array(), dtype=float)
    counts.remove()
    if cnt.size == hb.get_array().size:
        denom = float(np.percentile(cnt, 50)) or 1.0
        hb.set_alpha(np.clip(np.sqrt(cnt / denom), alpha_floor, 1.0))

    binned = np.asarray(hb.get_array(), dtype=float)
    lo, hi = np.nanpercentile(binned, [2, 98])
    if diverging and lo < 0 < hi:
        hb.set_cmap(div_cmap)
        if symmetric:                                    # 0 at center, symmetric range
            m = float(np.nanpercentile(np.abs(binned), 98)) or 1.0
            hb.set_norm(Normalize(vmin=-m, vmax=m))
        else:                                            # 0 at center, per-side contrast
            hb.set_norm(TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi))
    elif cmap is None:
        # Single-signed: the arm on that side, reversed for negatives so pale stays
        # at the zero end. Unsigned metrics (spread) get a plain blue ramp.
        if not diverging:
            hb.set_cmap(_SEQ_NEG)
        else:
            hb.set_cmap(_SEQ_POS if hi > 0 else _SEQ_NEG.reversed())
        hb.set_norm(Normalize(vmin=lo, vmax=hi))
    else:
        hb.set_norm(Normalize(vmin=lo, vmax=hi))
    cb = ax.figure.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cbar_label, fontsize=9)

    # Level sets on top, dark so they read over any colormap.
    vo, frame = _draw_vo_levelsets(ax, own, target, Sigma_vel, k_levels, fill_alpha=0.0,
                                   color="k", lw=0.9, ls="--", label=False, deterministic=True)
    # Markers only: this panel is zoomed to the cloud, so an arrow from the (usually
    # off-frame) origin would shoot off the plot.
    ax.plot(*target.vel, "s", color="#1f77b4", ms=7)     # apex = v_target
    ax.plot(*own.vel, "o", color="k", ms=7)              # v_own
    lvl_label = "bounded VO" if _is_bounded(Sigma_vel) else r"$k\sigma$ VO"
    ax.legend(handles=[Line2D([], [], color="k", lw=1.8, ls="-", label="deterministic VO"),
                       Line2D([], [], color="k", lw=0.9, ls="--", label=lvl_label)],
              loc="lower left", fontsize=7.5, framealpha=0.85)

    # Frame the level-set caps and the cloud together.
    lov, hiv = np.nanpercentile(vown, [1, 99], axis=0)
    _square_limits((ax,), frame + [tuple(lov), tuple(hiv)])
    ax.set_xlabel(r"$v_{East}$ [m/s]"); ax.set_ylabel(r"$v_{North}$ [m/s]")
    ax.set_aspect("equal", adjustable="box")
    return ax


def _noise_outlines(spec, k_levels):
    """Confidence-body outlines for a spec: k-sigma rings, or one support boundary."""
    if spec is None:
        return []
    return [_body(spec, k) for k in _levels_for(spec, k_levels)]


def _plot_noise(ax, samples2d, center, spec, title, xlabel, ylabel, k_levels=(1.0, 2.0, 3.0)):
    """Scatter the draws (relpos or relvel) about their mean, with the body outline.

    Shows the sampling distribution in place: peaked Gaussian vs flat-edged uniform.
    """
    c = np.asarray(center, dtype=float)
    ax.scatter(samples2d[:, 0], samples2d[:, 1], s=3, alpha=0.12, color="#4c72b0", lw=0)
    shapes = _noise_outlines(spec, k_levels)
    for sh in shapes:
        b = sh.sample_boundary(200) + c                  # bodies are origin-centered
        b = np.vstack([b, b[:1]])                         # close the loop
        ax.plot(b[:, 0], b[:, 1], color="#d62728", lw=1.1)
    ax.plot(c[0], c[1], "+", color="k", ms=8)

    dirs = (np.array([1., 0]), np.array([-1., 0]), np.array([0, 1.]), np.array([0, -1.]))
    off = samples2d - c
    reach = max([sh.support(d) for sh in shapes for d in dirs] +
                ([float(np.nanpercentile(np.abs(off), 99))] if off.size else [0.0]))
    reach = max(reach, 1e-6) * 1.12
    ax.set_xlim(c[0] - reach, c[0] + reach); ax.set_ylim(c[1] - reach, c[1] + reach)
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    return ax


def _plot_cpa_joint(subfig, own, target, Sigma_pos=None, Sigma_vel=None,
                    n=20000, R=None, horizon=None, rng=None, samples=None):
    """(TCPA, DCPA) jointplot -- scatter + marginals -- drawn into a SubFigure.

    Points are colored by membership of the collision band {|DCPA| <= R, TCPA >= 0,
    optionally TCPA <= horizon}, so P[collision] is the red fraction. DCPA is signed,
    so the band straddles 0. Pass ``samples`` to reuse an existing CpaSamples.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    m = Metrics(own, target)
    R = m.safety_radius if R is None else R

    s = samples if samples is not None else sample_cpa(own, target, Sigma_pos, Sigma_vel, n=n, rng=rng)
    tau, d = s.tcpa, s.dcpa
    hit = (np.abs(d) <= R) & (tau >= 0) & (True if horizon is None else tau <= horizon)
    p_hit = float(hit.mean())

    # Robust limits: TCPA tails run huge at low closing speed. DCPA is signed, so
    # frame it symmetrically about 0.
    xlo, xhi = np.percentile(tau, [0.5, 99.5])
    ymax = float(np.percentile(np.abs(d), 99.5))
    xpad = 0.05 * (xhi - xlo + 1e-9)
    xlo, xhi, ylo, yhi = xlo - xpad, xhi + xpad, -ymax * 1.05, ymax * 1.05

    gs = subfig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                             wspace=0.04, hspace=0.04)
    ax = subfig.add_subplot(gs[1, 0])
    ax_top = subfig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = subfig.add_subplot(gs[1, 1], sharey=ax)

    # Collision band {|d| <= R, tau >= 0}
    ax.add_patch(_Rectangle((0.0, -R), xhi, 2 * R, facecolor="#d62728", alpha=0.07,
                            edgecolor="none", zorder=0))
    ax.axvline(0.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0.0, color="k", lw=0.6, ls="-", alpha=0.3)
    for y in (R, -R):
        ax.axhline(y, color="#d62728", lw=1.0, ls="--", alpha=0.7,
                   label=(f"$|d| = R = {R:g}$" if y == R else None))

    # Joint scatter, colored by collision membership. The collision label is dropped
    # when the band is empty, so P[collision] = 0 leaves no empty legend handle.
    ax.scatter(tau[~hit], d[~hit], s=3, alpha=0.15, color="#2ca02c", lw=0,
               label="safe sample")
    ax.scatter(tau[hit], d[hit], s=3, alpha=0.25, color="#d62728", lw=0,
               label="collision sample" if hit.any() else None)
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
    """Three panels: position space | velocity space | (TCPA, DCPA) distribution.

    Sigma_pos / Sigma_vel are the relative noise per channel (None = exact), R the
    collision radius (default: the disc safety radius), horizon an optional upper
    TCPA bound on the collision band, k_levels the confidence levels to nest.
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


def four_panel(own, target, Sigma_pos=None, Sigma_vel=None, n=20000, R=None,
               horizon=None, k_levels=(1.0, 2.0, 3.0), rng=None, title=None,
               reduce=np.median):
    """Four panels: position space | VO+DCPA | VO+TCPA | (TCPA, DCPA) distribution.

    Samples are drawn once and shared. The two middle panels map each sample back to
    its own-velocity over the VO cone, colored by ``reduce`` of the metric per bin
    (np.median for the field, np.ptp / IQR for spread). Other args as in three_panel.
    """
    _apply_style()
    rng = np.random.default_rng(0) if rng is None else rng
    s = sample_cpa(own, target, Sigma_pos, Sigma_vel, n=n, rng=rng)

    fig = plt.figure(figsize=(15, 12.5))
    (sf_pos, sf_dcpa), (sf_tcpa, sf_joint) = fig.subfigures(2, 2, wspace=0.04, hspace=0.08)

    _plot_position_space(sf_pos.subplots(), own, target, Sigma_pos, k_levels)

    ax_d = sf_dcpa.subplots()
    _plot_vo_metric(ax_d, own, target, s, s.dcpa, "DCPA [m]", Sigma_vel=Sigma_vel,
                    k_levels=k_levels, reduce=reduce, diverging=True,
                    symmetric=True)
    ax_d.set_title("velocity space + DCPA", fontsize=11.5)

    ax_t = sf_tcpa.subplots()
    _plot_vo_metric(ax_t, own, target, s, s.tcpa, "TCPA [s]", Sigma_vel=Sigma_vel,
                    k_levels=k_levels, reduce=reduce, diverging=True)
    ax_t.set_title("velocity space + TCPA", fontsize=11.5)

    _plot_cpa_joint(sf_joint, own, target, Sigma_pos, Sigma_vel, n, R, horizon, rng, samples=s)
    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    return fig


def six_panel(own, target, Sigma_pos=None, Sigma_vel=None, n=20000, R=None,
              horizon=None, k_levels=(1.0, 2.0, 3.0), rng=None, title=None,
              reduce=np.median):
    """Six panels, 2x3: a geometry row over a distribution row.

    Top:    position space | VO+DCPA | VO+TCPA.
    Bottom: position noise | velocity noise | (TCPA, DCPA) distribution.

    The noise panels scatter each channel's draws against its confidence-body
    outline, making the model (Gaussian vs uniform) visible. Args as in four_panel.
    """
    _apply_style()
    rng = np.random.default_rng(0) if rng is None else rng
    s = sample_cpa(own, target, Sigma_pos, Sigma_vel, n=n, rng=rng)
    r_hat = np.asarray(target.pos, dtype=float) - np.asarray(own.pos, dtype=float)
    v_hat = np.asarray(target.vel, dtype=float) - np.asarray(own.vel, dtype=float)

    fig = plt.figure(figsize=(19, 12.5))
    (sf_pos, sf_dcpa, sf_tcpa), (sf_pn, sf_vn, sf_joint) = fig.subfigures(
        2, 3, width_ratios=(1.0, 1.0, 1.15), wspace=0.03, hspace=0.08)

    _plot_position_space(sf_pos.subplots(), own, target, Sigma_pos, k_levels)

    ax_d = sf_dcpa.subplots()
    _plot_vo_metric(ax_d, own, target, s, s.dcpa, "DCPA [m]", Sigma_vel=Sigma_vel,
                    k_levels=k_levels, reduce=reduce, diverging=True,
                    symmetric=True)
    ax_d.set_title("velocity space + DCPA", fontsize=11.5)

    ax_t = sf_tcpa.subplots()
    _plot_vo_metric(ax_t, own, target, s, s.tcpa, "TCPA [s]", Sigma_vel=Sigma_vel,
                    k_levels=k_levels, reduce=reduce, diverging=True)
    ax_t.set_title("velocity space + TCPA", fontsize=11.5)

    _plot_noise(sf_pn.subplots(), s.relpos, r_hat, Sigma_pos, "relative position (sampled)",
                "East [m]", "North [m]", k_levels)
    _plot_noise(sf_vn.subplots(), s.relvel, v_hat, Sigma_vel, "relative velocity (sampled)",
                r"$v_{East}$ [m/s]", r"$v_{North}$ [m/s]", k_levels)

    _plot_cpa_joint(sf_joint, own, target, Sigma_pos, Sigma_vel, n, R, horizon, rng, samples=s)
    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    return fig


def deterministic_vo_figure(own, target):
    """Two-panel VO figure for an encounter: relative space + velocity space."""
    _apply_style()
    O = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
    cone = collision_cone(O)
    vo = VelocityObstacle(O, target.vel)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6.3))

    # Panel A
    plot_shape(O, ax=axA, n=300, alpha=0.15, color="#d62728", label="config obstacle $O$")
    plot_collision_cone(cone, axA, apex=(0, 0), alpha=0.08)
    axA.plot(cone.contacts[:, 0], cone.contacts[:, 1], "o", mfc="white",
             color="#d62728", ms=6, zorder=6)
    axA.plot(0, 0, "o", color="#1f77b4", ms=9, label="own ship (origin)")
    axA.set_title("(a) relative space: obstacle + collision cone", fontsize=11.5)
    axA.set_xlabel("East [m]"); axA.set_ylabel("North [m]")
    axA.legend(loc="lower left", fontsize=8.5)
    axA.set_aspect("equal", adjustable="datalim")

    # Panel B
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
    """Two panels: deterministic VO vs. nested velocity-uncertainty levels.

    Velocity uncertainty offsets the cone edges outward and rounds the apex but
    leaves the half-angle alone -- unlike position uncertainty, which widens it.
    """
    _apply_style()
    O = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
    vo = VelocityObstacle(O, target.vel)
    rvos = [(k, robust_velocity_obstacle(vo, covariance_ellipse(Sigma_v, k))) for k in k_levels]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6.3))

    # Panel A
    plot_collision_cone(vo.cone, axA, apex=vo.apex, label="velocity obstacle")
    _annotate_velocity_space(axA, own, target)
    axA.set_title("(a) velocity space: deterministic VO", fontsize=11.5)

    # Panel B. Largest k first, so overlapping fills build toward the core.
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
    """Two panels with nested position AND velocity uncertainty.

    (a) relative space: position-inflated obstacles + their widened cones.
    (b) velocity space: every (k_pos, k_vel) pair as an overlapping level set.
    """
    _apply_style()
    O = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
    scones = probabilistic_collision_cone(O, Sigma_pos, k_pos_levels)

    # The cone depends only on k_pos, so build it once per position level.
    combined = []                                   # (k_pos, k_vel, rvo)
    for kp in k_pos_levels:
        vo_k = VelocityObstacle(MinkowskiSum(O, covariance_ellipse(Sigma_pos, kp)),
                                target.vel)
        for kv in k_vel_levels:
            combined.append((kp, kv, robust_velocity_obstacle(vo_k, covariance_ellipse(Sigma_vel, kv))))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6.3))

    # Panel A
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

    # Panel B. Largest first, so overlapping alpha builds toward the small-k core.
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

    # Panel C
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

    # Panel D
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
