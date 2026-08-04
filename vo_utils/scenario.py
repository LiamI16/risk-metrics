from dataclasses import dataclass

from minkowski_utils import Circle
from .ship import Ship
from .metrics import Metrics
from .sampling import anchored_covariances


@dataclass
class Scenario:
    """A named two-ship encounter with a scale-anchored Gaussian noise model.

    Noise is set by ``alpha`` (position) and ``beta`` (velocity) rather than raw
    covariances. Consumers pull ``ships()`` / ``covariances()`` / ``metrics()``.

    Attributes
    ----------
    name : str
        Encounter name.
    own_pos, own_vel, target_pos, target_vel : tuple
        Ship positions and velocities.
    own_r, target_r : float
        Ship disc radii.
    alpha, beta : float
        Position and velocity noise scales.
    vel_scale : float or None
        Velocity anchor override (e.g. for low closing speed).
    n : int
        Monte-Carlo sample count.
    horizon : float or None
        Optional TCPA upper bound.
    """
    name: str
    own_pos: tuple
    own_vel: tuple
    target_pos: tuple
    target_vel: tuple
    own_r: float = 10.0
    target_r: float = 10.0
    alpha: float = 0.25                 # sigma_pos = alpha * R
    beta: float = 0.10                  # sigma_vel = beta * vel_scale
    vel_scale: float | None = None      # override velocity anchor (e.g. low closing speed)
    n: int = 20000                      # Monte-Carlo samples (sampling / plots)
    horizon: float | None = None        # optional TCPA upper bound (collision corner)

    def ships(self):
        """(own, target) Ships with disc domains."""
        own = Ship(self.own_pos, self.own_vel, Circle(self.own_r))
        target = Ship(self.target_pos, self.target_vel, Circle(self.target_r))
        return own, target

    def metrics(self):
        """Deterministic CPA metrics for the nominal (no-noise) encounter."""
        own, target = self.ships()
        return Metrics(own, target)

    def covariances(self):
        """Scale-anchored (Sigma_pos, Sigma_vel) for this encounter."""
        own, target = self.ships()
        return anchored_covariances(own, target, self.alpha, self.beta, self.vel_scale)


# Predefined scenarios (r = 10 each, R = 20).  The comment under each is exact (no noise) (TCPA, DCPA). 

# ---- Head-on -----
HEAD_ON = Scenario("head_on", (0, 0), (0, 5), (0, 200), (0, -5))
# TCPA 20.0 s, DCPA 0 m

HEAD_ON_FASTER = Scenario("head_on_faster", (0, 0), (0, 8), (0, 200), (0, -5))
# TCPA 15.4 s, DCPA 0 m -- head-on, own faster

# ---- Crossing -----
CROSSING = Scenario("crossing", (0, 0), (5, 0), (100, 100), (0, -5))
# TCPA 20.0 s, DCPA 0 m -- perpendicular paths meeting at (100, 0)

CROSSING_WIDE = Scenario("crossing_wide", (0, 0), (5, 0), (0, 200), (0, -5))
# TCPA 20.0 s, DCPA 141.4 m -- perpendicular but a wide safe pass (DCPA >> R).

NORTH_EAST = Scenario("north_east", (0, 0), (5, 5), (0, 200), (0, -5))
# TCPA 16.0 s, DCPA 89.4 m -- own heads NE

# ---- Overtaking / collinear same-direction ----
OVERTAKING = Scenario("overtaking", (0, 0), (0, 8), (0, 120), (0, 4))
# TCPA 30.0 s, DCPA 0 m -- own overtakes a slower target dead ahead.

COLLINEAR_SLOWER = Scenario("collinear_slower", (0, 0), (0, -3), (0, 200), (0, -5))
# TCPA 100.0 s, DCPA 0 m -- both southbound, own slower; target overtakes from astern.

# ---- Parallel (no relative motion) ----
COLLINEAR_EQUAL = Scenario("collinear_equal", (0, 0), (0, -5), (0, 200), (0, -5),
                           vel_scale=5.0)
# TCPA 0 s, DCPA 200 m -- identical velocities; constant 200 m gap.

# ---- Separating / already past CPA ----
COLLINEAR_FALLING_BEHIND = Scenario("collinear_falling_behind", (0, 0), (0, 3), (0, 200), (0, 5))
# TCPA -100.0 s, DCPA 0 m -- both northbound, own slower

DIVERGING = Scenario("diverging", (0, 0), (0, -5), (0, 100), (0, 5))
# TCPA -10.0 s, DCPA 0 m 

# ---- Miscellaneous ----
STATIONARY = Scenario("stationary", (0, 0), (0, 5), (0, 200), (0, 0))
# TCPA 40.0 s, DCPA 0 m -- baseline own moves towards stationary target

LOW_CLOSING = Scenario("low_closing", (0, 0), (0, 5), (2, 120), (0, 4.5),
                       vel_scale=5.0)
# TCPA 240.0 s, DCPA 2 m -- slow near parallel

BEAM = Scenario("beam", (0, 0), (0, 5), (50, 0), (0, -5))
# TCPA 0 s, DCPA 50 m

CANONICAL_SCENARIOS = (
    HEAD_ON, HEAD_ON_FASTER,
    CROSSING, CROSSING_WIDE, NORTH_EAST,
    OVERTAKING, COLLINEAR_SLOWER,
    COLLINEAR_EQUAL,
    COLLINEAR_FALLING_BEHIND, DIVERGING,
    STATIONARY, LOW_CLOSING, BEAM,
)
