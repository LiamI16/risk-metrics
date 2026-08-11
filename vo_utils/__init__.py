"""vo_utils: velocity obstacles built on the minkowski_utils module.

"""

from .ship import Ship
from .metrics import Metrics
from .cone import collision_cone, CollisionCone
from .obstacle import VelocityObstacle, config_space_obstacle
from .uncertainty import covariance_ellipse, probabilistic_collision_cone, SigmaCone
from .uncertainty_models import (
    robust_velocity_obstacle, RobustVelocityObstacle, uncertain_velocity_obstacle,
    UncertaintyModel, Gaussian, UniformBox, UniformEllipse, IndependentSum,
)
from .sampling import (
    sample_cpa, CpaSamples, anchored_covariances, shaped_covariance, uniform_like,
)
from .scenario import (
    Scenario, CANONICAL_SCENARIOS,
    HEAD_ON, HEAD_ON_FASTER, CROSSING, CROSSING_WIDE, NORTH_EAST,
    OVERTAKING, COLLINEAR_SLOWER, COLLINEAR_EQUAL, COLLINEAR_FALLING_BEHIND,
    DIVERGING, STATIONARY, LOW_CLOSING, BEAM,
)

__all__ = [
    "Ship", "Metrics",
    "collision_cone", "CollisionCone",
    "VelocityObstacle", "config_space_obstacle",
    "covariance_ellipse", "probabilistic_collision_cone", "SigmaCone",
    "robust_velocity_obstacle", "RobustVelocityObstacle", "uncertain_velocity_obstacle",
    "UncertaintyModel", "Gaussian", "UniformBox", "UniformEllipse", "IndependentSum",
    "sample_cpa", "CpaSamples", "anchored_covariances", "shaped_covariance", "uniform_like",
    "Scenario", "CANONICAL_SCENARIOS",
    "HEAD_ON", "HEAD_ON_FASTER", "CROSSING", "CROSSING_WIDE", "NORTH_EAST",
    "OVERTAKING", "COLLINEAR_SLOWER", "COLLINEAR_EQUAL", "COLLINEAR_FALLING_BEHIND",
    "DIVERGING", "STATIONARY", "LOW_CLOSING", "BEAM",
]
