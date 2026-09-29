"""Convex shapes, reflections and Minkowski sums: the geometry the velocity obstacle is built on.

Run from the repo root:  python examples/00_shapes.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_metrics import Circle, Ellipse, MinkowskiSum, Polygon, config_space_obstacle

OUT = Path(__file__).resolve().parent.parent / "output" / "examples" / "00_shapes.png"


def draw(ax, shape, color, label, fill=0.15, **kw):
    """Fill and outline a shape, using the 300 boundary points sample_boundary() returns."""
    ring = shape.sample_boundary(300)
    ax.fill(*ring.T, fc=color, alpha=fill, lw=0)
    ax.plot(*np.vstack([ring, ring[:1]]).T, color=color, label=label, **kw)


if __name__ == "__main__":
    plt.style.use(Path(__file__).resolve().parent / "risk_metrics.mplstyle")

    # 1. Three convex shapes. A ship's "domain" is the area it occupies (or a safety region around it) 
    # given in the ship's own frame, in m.
    hull = Polygon([(0.0, 3.0), (1, 1), (1, -2), (-1, -2), (-1, 1)])     # a ship outline
    ellipse = Ellipse(2.0, 0.9, theta=np.deg2rad(25))
    circle = Circle(0.8)

    # 2. A Minkowski sum sweeps one shape around the boundary of another. The sum is itself a
    # convex Shape, and its support(d) is how far it reaches in direction d. That reach is the
    # sum of the parts' reaches. The outline is never enumerated as a polygon, so sums of
    # arbitrary convex shapes stay cheap.
    total = MinkowskiSum(hull, ellipse, circle)
    for d, name in (([1.0, 0.0], "east"), ([0.0, 1.0], "north")):
        parts = [hull.support(d), ellipse.support(d), circle.support(d)]
        print(f"reach {name:>5}: {' + '.join(f'{r:.2f}' for r in parts)} "
              f"= {sum(parts):.2f} m, and the sum reaches {total.support(d):.2f} m")

    # 3. The velocity obstacle needs the config-space obstacle C. The ships overlap when a point
    # of one domain coincides with a point of the other: relpos + b - a = 0 for some a in D_own
    # and b in D_target. Every b - a fills out D_target + (-D_own), so the ships overlap exactly
    # when that set, shifted to relpos, covers the origin. C is that shifted set. The own domain
    # enters reflected because it is the one being subtracted, and shifting it to relpos is what
    # lets the own ship be a single point at the origin.
    relpos = np.array([6.0, 4.0])                # target position minus own position
    C = config_space_obstacle(own_domain=hull, target_domain=ellipse, relpos=relpos)
    # support(d) is measured from the origin, so it turns negative when C lies entirely on the
    # far side. The origin is inside C, and the ships overlap, when no direction is negative.
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    h_min = min(C.support([np.cos(a), np.sin(a)]) for a in angles)
    print(f"\nsmallest support of C over 360 directions: {h_min:.2f} m -> the origin is "
          f"{'inside' if h_min >= 0 else 'outside'} C, so the ships "
          f"{'overlap' if h_min >= 0 else 'are clear'} at t = 0")

    fig, (ax_s, ax_o) = plt.subplots(1, 2, figsize=(13, 6.3))

    # (a) The three shapes and their sum.
    draw(ax_s, hull, "tab:orange", "polygon (own hull)", lw=1.5)
    draw(ax_s, ellipse, "tab:blue", "ellipse", lw=1.5)
    draw(ax_s, circle, "tab:purple", "circle", lw=1.5)
    draw(ax_s, total, "tab:green", "Minkowski sum", fill=0.10, lw=1.8)
    ax_s.set(aspect="equal", xlabel="x [m]", ylabel="y [m]")
    ax_s.set_title("(a) convex shapes add by Minkowski sum", fontsize=11.5)
    ax_s.legend(loc="upper left", fontsize=8.5)

    # (b) The same machinery applied to a two-ship scenario, in relative space.
    draw(ax_o, hull, "tab:orange", r"own domain $D_{own}$", lw=1.5)
    draw(ax_o, hull.reflect(), "tab:purple", r"reflected $-D_{own}$", fill=0.0, lw=1.5, ls="--")
    draw(ax_o, Ellipse(2.0, 0.9, center=relpos, theta=np.deg2rad(25)), "tab:blue",
         r"target domain at relpos", lw=1.5)
    draw(ax_o, C, "tab:red", r"$C$: $D_{target} \oplus (-D_{own})$ at relpos", fill=0.10, lw=1.8)
    ax_o.plot(0, 0, "o", color="k", ms=7, label="own ship (origin)")
    ax_o.plot(*relpos, "s", color="tab:blue", ms=7)
    ax_o.set(aspect="equal", xlabel="East [m]", ylabel="North [m]")
    ax_o.set_title("(b) relative space: the collision set is one Minkowski sum", fontsize=11.5)
    ax_o.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print("saved", OUT)
