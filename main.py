"""Velocity-obstacle scenario runner.

Builds two ships, reports the closest-point-of-approach risk metrics, and
evaluates the velocity obstacle for the encounter -- deterministic and
probabilistic (positional Gaussian uncertainty).

Run:  python3 main.py
"""

from pathlib import Path

import numpy as np

from minkowski_utils import Circle, Ellipse, Polygon
from vo_utils import (
    Ship, Metrics, config_space_obstacle, probabilistic_collision_cone,
)
from vo_utils.plotting import deterministic_vo_figure, probabilistic_vo_figure

ARTIFACTS = Path(__file__).resolve().parent / "artifacts" / "vo_plots"


def build_encounter():
    """Own ship (disc domain) meeting a target with an oriented ellipse domain."""
    ship_shape = Polygon([(0.0, 3.0), (1, 1), (1, -2), (-1, -2), (-1, 1)])
    own = Ship([0.0, 0.0], [0.0, 5.0], domain=ship_shape)
    target = Ship([0.0, 70.0], [0.0, -6.0],
                  domain=Ellipse(3, 2.0, theta=np.deg2rad(20)))
    return own, target


def main():
    own, target = build_encounter()

    # -- CPA risk metrics ----------------------------------------------------
    # DCPA/TCPA are disc-domain metrics; the target here carries an ellipse
    # domain, so they raise -- the VO below handles the arbitrary shape.
    m = Metrics(own, target)
    try:
        print(f"DCPA = {m.DCPA():7.1f} m")
        print(f"TCPA = {m.TCPA():7.1f} s")
    except TypeError as e:
        print(f"DCPA/TCPA skipped: {e}")

    # -- Deterministic velocity obstacle -------------------------------------
    vo = m.vo()
    print(f"own velocity on collision course: {vo.contains(own.vel)}")

    # -- Probabilistic (positional uncertainty) nested cones -----------------
    O = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
    Sigma = np.array([[30.0, 12.0], [12.0, 20.0]])   # positional covariance [m^2]
    scones = probabilistic_collision_cone(O, Sigma, k_levels=(1.0, 2.0, 3.0))
    for sc in scones:
        e1, e2 = sc.cone.edges
        spread = np.degrees(np.arccos(np.clip(e1 @ e2, -1, 1)))
        print(f"{sc.k:.0f}-sigma cone angular width: {spread:6.1f} deg")

    # -- Figures -------------------------------------------------------------
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    deterministic_vo_figure(own, target).savefig(
        ARTIFACTS / "vo_ellipse_demo.png", bbox_inches="tight")
    probabilistic_vo_figure(own, target, Sigma).savefig(
        ARTIFACTS / "pvo_ellipse_demo.png", bbox_inches="tight")
    print(f"saved figures to {ARTIFACTS}")


if __name__ == "__main__":
    main()
