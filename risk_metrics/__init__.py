"""Collision-risk metrics (CPA, velocity obstacles) for ships with convex domains and uncertainty."""

from .geometry import Shape, Circle, Ellipse, Polygon, MinkowskiSum, covariance_ellipse
from .distributions import Distribution, IndependentSum, Reflected, Gaussian, UniformEllipse, UniformBox
from .cpa import cpa, collides
from .vo import config_space_obstacle, VelocityObstacle, UncertaintyAwareVelocityObstacle
from .sampling import CPA_Samples, sample_cpa
from .scenarios import (
    Ship, Scenario, CANONICAL_SCENARIOS,
    HEAD_ON, HEAD_ON_FASTER, CROSSING, CROSSING_WIDE, NORTH_EAST,
    OVERTAKING, COLLINEAR_SLOWER, COLLINEAR_EQUAL, COLLINEAR_FALLING_BEHIND,
    DIVERGING, STATIONARY, LOW_CLOSING, BEAM,
)

__all__ = [
    "Shape", "Circle", "Ellipse", "Polygon", "MinkowskiSum", "covariance_ellipse",
    "Distribution", "IndependentSum", "Reflected", "Gaussian", "UniformEllipse", "UniformBox",
    "cpa", "collides",
    "config_space_obstacle", "VelocityObstacle", "UncertaintyAwareVelocityObstacle",
    "CPA_Samples", "sample_cpa",
    "Ship", "Scenario", "CANONICAL_SCENARIOS",
    "HEAD_ON", "HEAD_ON_FASTER", "CROSSING", "CROSSING_WIDE", "NORTH_EAST",
    "OVERTAKING", "COLLINEAR_SLOWER", "COLLINEAR_EQUAL", "COLLINEAR_FALLING_BEHIND",
    "DIVERGING", "STATIONARY", "LOW_CLOSING", "BEAM",
]
