"""CPA metrics, the collision test and the velocity obstacle, for any of the preset scenarios.

TCPA is the time until two ships are nearest, negative if that moment is already past, and
DCPA how far apart they are then. DCPA is signed by which side the target passes on: positive
when it passes to the left of the relative track, the direction the target moves as seen from
the own ship. The sign only records the side; the separation itself is |DCPA|. R is the two
ships' safety radii added: they are in collision whenever they are closer than R.

Two ships holding the same velocity have no relative motion. The
sign is then undefined and DCPA is the current separation, as in the collinear_equal preset.

Run from the repo root:  python examples/01_cpa_and_vo.py [--scenario NAME] [--all]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_metrics import (
    CANONICAL_SCENARIOS, VelocityObstacle, collides, config_space_obstacle, cpa,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "examples"
BY_NAME = {sc.name: sc for sc in CANONICAL_SCENARIOS}


def draw_encounter(ax, sc, legend=True):
    """The encounter as it happens: both ships, their tracks, and the closest approach.

    Returns (TCPA, DCPA)
    """
    own, target = sc.ships()
    tcpa, dcpa = cpa(target.pos - own.pos, target.vel - own.vel)
    closing = tcpa >= 0                              # at TCPA = 0 the closest approach is now
    marks = [own.pos, target.pos]
    if closing:
        marks += [own.pos + own.vel * tcpa, target.pos + target.vel * tcpa]
    marks = np.array(marks)
    center = (marks.max(axis=0) + marks.min(axis=0)) / 2
    half = max(0.65 * (marks.max(axis=0) - marks.min(axis=0)).max(), 3.0 * sc.R)

    # Own gets a thick soft track, target a thin dashed one, so neither hides the other
    # where the two tracks lie on one line, as in head_on and the collinear scenarios.
    for ship, color, style, label in ((own, "tab:green", dict(lw=3.0, alpha=0.45), "own"),
                                      (target, "tab:blue", dict(lw=1.2, ls="--"), "target")):
        ring = ship.domain.sample_boundary(120) + ship.pos
        ax.fill(*ring.T, fc=color, alpha=0.35, lw=0)
        speed = np.linalg.norm(ship.vel)
        if speed > 0:                               # run the track to the edge of the window
            ax.plot(*np.array([ship.pos, ship.pos + ship.vel * (2.5 * half / speed)]).T,
                    color=color, **style)
        ax.plot(*ship.pos, "o", color=color, ms=5, label=label)
        if closing:                                 # where the ship is at the closest approach
            ax.plot(*(ship.pos + ship.vel * tcpa), "o", mfc="white", color=color, ms=6)
    if closing:
        pair = np.array([own.pos + own.vel * tcpa, target.pos + target.vel * tcpa])
        ax.plot(*pair.T, color="k", lw=1.0,
                label=f"|DCPA| = {abs(dcpa):.0f} m"
                      + (" (within R, a collision)" if abs(dcpa) <= sc.R else ""))
    ax.set(xlim=(center[0] - half, center[0] + half), ylim=(center[1] - half, center[1] + half),
           aspect="equal")
    if legend:
        ax.legend(loc="best", fontsize=8.5)
    return tcpa, dcpa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="crossing", choices=sorted(BY_NAME),
                        help="which preset scenario to work through (default: crossing)")
    parser.add_argument("--all", action="store_true",
                        help="draw every preset scenario instead of working through one")
    args = parser.parse_args()
    plt.style.use(Path(__file__).resolve().parent / "risk_metrics.mplstyle")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        # --all draws the whole preset catalog and stops here, every scenario the same way;
        # the walkthrough for a single scenario picks up at step 1 below.
        ncols = 5
        nrows = -(-len(CANONICAL_SCENARIOS) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
        for ax, sc in zip(axes.ravel(), CANONICAL_SCENARIOS):
            tcpa, dcpa = draw_encounter(ax, sc, legend=False)
            ax.set_title(f"{sc.name}\nTCPA {np.round(tcpa) + 0.0:.0f} s, "
                         f"DCPA {np.round(dcpa) + 0.0:.0f} m", fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in axes.ravel()[len(CANONICAL_SCENARIOS):]:
            ax.axis("off")
        fig.tight_layout()
        out = OUT_DIR / "01_scenarios.png"
        fig.savefig(out)
        print("saved", out)
        raise SystemExit                        # done: --all produces only the catalog

    # 1. Encounter. Positions in m, velocities in m/s.
    sc = BY_NAME[args.scenario]
    own, target = sc.ships()
    R = sc.R
    relpos = target.pos - own.pos
    relvel = target.vel - own.vel

    # 2. Closest point of approach.
    tcpa, dcpa = cpa(relpos, relvel)
    print(f"{sc.name}: TCPA = {tcpa:.1f} s, DCPA = {dcpa:.1f} m, R = {R:.0f} m")

    # 3. Collision test, over an unbounded and a finite horizon.
    for horizon in (None, 10.0, 30.0):
        label = "infinite" if horizon is None else f"{horizon:g} s"
        print(f"  collides, horizon {label:>8}: {bool(collides(relpos, relvel, R, horizon=horizon))}")

    # 4. Velocity obstacle: the own velocities that lead to a collision (infinite horizon).
    vo = VelocityObstacle(own, target)
    hit = vo.contains(own.vel)
    print(f"  own velocity {own.vel} in VO: {hit}")

    # 5. Check contains() against collides() on a grid of candidate own velocities. contains()
    # works from the ship domains and collides() from the scalar R, so they agree only because
    # every preset uses circular domains whose radii add up to R.
    speed = max(np.linalg.norm(own.vel), np.linalg.norm(target.vel), 1.0)
    lim = 2.0 * speed
    vx, vy = np.meshgrid(np.linspace(-lim, lim, 41), np.linspace(-lim, lim, 41))
    candidates = np.stack([vx, vy], axis=-1).reshape(-1, 2)
    in_vo = vo.contains(candidates)
    # Changing the own velocity changes relvel = target.vel - v_own.
    agree = collides(relpos, target.vel - candidates, R)
    print(f"  {in_vo.sum()} of {len(candidates)} candidates in the VO; "
          f"agreement with collides(): {(in_vo == agree).mean():.2%} "
          f"({(in_vo != agree).sum()} disagree)")
    if (in_vo != agree).any():
        _, d_off = cpa(relpos, target.vel - candidates[in_vo != agree])
        print(f"  the disagreements are grazing courses, |DCPA| = R to "
              f"{np.abs(np.abs(d_off) - R).max():.1e} m, where the two tests round differently")

    # 6. Figure: the same encounter in the three frames the package works in.
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16.5, 5.8))

    # (a) Absolute space: what an observer sees.
    draw_encounter(ax_a, sc)
    ax_a.set(xlabel="East [m]", ylabel="North [m]")
    ax_a.set_title(f"(a) {sc.name}: the encounter", fontsize=11.5)

    # (b) Relative space: the own ship at the origin, the obstacle and the cone of relative
    # headings that hit it.
    C = config_space_obstacle(own.domain, target.domain, relpos)
    ring = C.sample_boundary(300)
    ax_b.fill(*ring.T, fc="tab:red", alpha=0.15, lw=0, label="config obstacle $C$")
    ax_b.plot(*np.vstack([ring, ring[:1]]).T, color="tab:red", lw=1.5)
    L = 1000.0                                  # wedge length; the axes clip it
    ax_b.fill(*np.array([(0, 0), L * vo.edges[0], L * vo.edges[1]]).T, fc="none", hatch="///",
              ec="tab:red", lw=0, alpha=0.5, label="collision cone")
    for e in vo.edges:
        ax_b.plot([0, L * e[0]], [0, L * e[1]], color="tab:red", lw=1.5)
    ax_b.plot(*vo.tangent_points.T, "o", mfc="white", color="tab:red", ms=6, zorder=6,
              ls="none", label="tangent points (cone edges touch $C$)")
    ax_b.plot(0, 0, "o", color="tab:green", ms=9, label="own ship (origin)")
    reach = 1.4 * np.linalg.norm(relpos) + R
    ax_b.set(xlim=(-reach, reach), ylim=(-reach, reach), aspect="equal",
             xlabel="East [m]", ylabel="North [m]")
    ax_b.set_title("(b) relative space: obstacle + collision cone", fontsize=11.5)
    ax_b.legend(loc="lower left", fontsize=8.5)

    # (c) Velocity space: the VO is the collision cone moved to the target velocity.
    ax_c.fill(*vo.outline(L).T, fc="tab:red", alpha=0.15, lw=0, label="velocity obstacle")
    ax_c.plot(*vo.outline(L).T, color="tab:red", lw=1.5)
    ax_c.plot(0, 0, "+", color="k", ms=10)
    ax_c.annotate("", xy=target.vel, xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2))
    ax_c.plot(*target.vel, "s", color="tab:blue", ms=7, label=r"$v_{target}$ (apex)")
    c = "tab:red" if hit else "tab:green"
    ax_c.annotate("", xy=own.vel, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
    ax_c.plot(*own.vel, "o", color=c, ms=9,
              label=r"$v_{own}$ " + ("(collision)" if hit else "(safe)"))
    ax_c.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal",
             xlabel=r"$v_{East}$ [m/s]", ylabel=r"$v_{North}$ [m/s]")
    ax_c.set_title("(c) velocity space: velocity obstacle", fontsize=11.5)
    ax_c.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()

    out = OUT_DIR / f"01_cpa_and_vo_{sc.name}.png"
    fig.savefig(out)
    print("saved", out)
