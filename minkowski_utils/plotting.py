"""
Visualization for convex shapes
"""

import numpy as np
import matplotlib.pyplot as plt

from .shapes import Shape


def plot_shape(shape, ax=None, n=200, fill=True, alpha=0.15, lw=1.6,
               color=None, label=None, **kw):
    """Draw a convex Shape's boundary on ``ax`` (created if None).

    Parameters
    ----------
    shape : Shape
        Any shape (Circle, Ellipse, Polygon, MinkowskiSum, ...).
    n : int
        Boundary samples.  Ignored by Polygon (returns its exact vertices).
    fill : bool
        Shade the interior with ``alpha`` in the outline color.
    color, lw, label, **kw
        Forwarded to the outline ``plot`` (kw also to the line).

    Returns
    -------
    matplotlib.axes.Axes
    """
    if not isinstance(shape, Shape):
        raise TypeError(f"plot_shape expects a Shape, got {type(shape)}")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal", adjustable="datalim")

    pts = shape.sample_boundary(n)
    ring = np.vstack([pts, pts[0]])                 # close the loop for drawing

    line, = ax.plot(ring[:, 0], ring[:, 1], lw=lw, color=color, label=label, **kw)
    if fill:
        ax.fill(ring[:, 0], ring[:, 1], color=line.get_color(), alpha=alpha, lw=0)
    return ax


if __name__ == "__main__":
    from .shapes import Circle, Ellipse, Polygon
    from .minkowski import MinkowskiSum

    A = Ellipse(3.0, 1.0, center=(0, 0), theta=np.deg2rad(20))
    B = Polygon([(0.0, 3.0), (1, 1), (1, -2), (-1, -2), (-1, 1)])

    ax = plot_shape(A, label=str(A))
    plot_shape(B,ax=ax, label=str(B))

    plot_shape(MinkowskiSum(A, B), ax=ax, label = "Minkowski Sum")

    ax.legend(loc="upper left", fontsize=9)
    from pathlib import Path
    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "shape_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "plotting_demo.png"
    ax.figure.savefig(out, bbox_inches="tight", dpi=150)
    print("saved", out)