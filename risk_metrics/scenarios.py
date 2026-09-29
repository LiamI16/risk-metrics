"""Ships and the named two-ship scenarios."""

from dataclasses import dataclass

import numpy as np

from .geometry import Circle, Shape


@dataclass(eq=False)
class Ship:
    """A ship with position (m), constant velocity (m/s) and a convex domain."""

    pos: np.ndarray
    vel: np.ndarray
    domain: Shape

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float)
        self.vel = np.asarray(self.vel, dtype=float)


@dataclass
class Scenario:
    """One named two-ship encounter: both states, both radii, and ``R`` their sum."""

    name: str
    own_pos: tuple
    own_vel: tuple
    target_pos: tuple
    target_vel: tuple
    own_r: float = 10.0
    target_r: float = 10.0

    def ships(self):
        """(own, target) with Circle domains."""
        return (Ship(self.own_pos, self.own_vel, Circle(self.own_r)),
                Ship(self.target_pos, self.target_vel, Circle(self.target_r)))

    @property
    def R(self):
        return self.own_r + self.target_r


# Each comment gives the nominal (TCPA, signed DCPA). Radii are 10 m, so R = 20 m.

# (20.0 s, 0.0 m)
HEAD_ON = Scenario("head_on", (0, 0), (0, 5), (0, 200), (0, -5))
# (15.4 s, 0.0 m) own faster
HEAD_ON_FASTER = Scenario("head_on_faster", (0, 0), (0, 8), (0, 200), (0, -5))
# (20.0 s, 0.0 m) perpendicular paths meeting at (100, 0)
CROSSING = Scenario("crossing", (0, 0), (5, 0), (100, 100), (0, -5))
# (20.0 s, -141.4 m) perpendicular, wide pass
CROSSING_WIDE = Scenario("crossing_wide", (0, 0), (5, 0), (0, 200), (0, -5))
# (16.0 s, -89.4 m) own heads northeast
NORTH_EAST = Scenario("north_east", (0, 0), (5, 5), (0, 200), (0, -5))
# (30.0 s, 0.0 m) own overtakes a slower target dead ahead
OVERTAKING = Scenario("overtaking", (0, 0), (0, 8), (0, 120), (0, 4))
# (100.0 s, 0.0 m) both southbound, target overtakes from astern
COLLINEAR_SLOWER = Scenario("collinear_slower", (0, 0), (0, -3), (0, 200), (0, -5))
# (0.0 s, 200.0 m) identical velocities, constant gap
COLLINEAR_EQUAL = Scenario("collinear_equal", (0, 0), (0, -5), (0, 200), (0, -5))
# (-100.0 s, 0.0 m) both northbound, own slower; CPA is past
COLLINEAR_FALLING_BEHIND = Scenario("collinear_falling_behind", (0, 0), (0, 3), (0, 200), (0, 5))
# (-10.0 s, 0.0 m) moving apart
DIVERGING = Scenario("diverging", (0, 0), (0, -5), (0, 100), (0, 5))
# (40.0 s, 0.0 m) stationary target
STATIONARY = Scenario("stationary", (0, 0), (0, 5), (0, 200), (0, 0))
# (240.0 s, 2.0 m) slow, nearly parallel closing
LOW_CLOSING = Scenario("low_closing", (0, 0), (0, 5), (2, 120), (0, 4.5))
# (0.0 s, 50.0 m) abeam, passing now
BEAM = Scenario("beam", (0, 0), (0, 5), (50, 0), (0, -5))

CANONICAL_SCENARIOS = (
    HEAD_ON, HEAD_ON_FASTER, CROSSING, CROSSING_WIDE, NORTH_EAST,
    OVERTAKING, COLLINEAR_SLOWER, COLLINEAR_EQUAL, COLLINEAR_FALLING_BEHIND,
    DIVERGING, STATIONARY, LOW_CLOSING, BEAM,
)
