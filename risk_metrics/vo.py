"""Velocity obstacles for convex domains, with and without uncertainty."""

import numpy as np
from scipy.optimize import brentq
from .cpa import MIN_REL_SPEED
from .geometry import Circle, MinkowskiSum

_N_ARC = 720 # direction sampling, arc spans at most 180 deg * 4 per deg


def _direction(phi):
    return np.array([np.cos(phi), np.sin(phi)])

def config_space_obstacle(own_domain, target_domain, relpos):
    r"""$C = D_{target} \oplus (-D_{own})$ shifted to ``relpos``: the ships overlap at time $t$
    when $-relvel\,t$ lies in C."""
    return MinkowskiSum(target_domain, own_domain.reflect(),
                        Circle(0.0, center=np.asarray(relpos, dtype=float)))


def _collision_cone(C, n_grid=1440):
    """Normals, unit edges and contacts of the cone from the origin over C; None if 0 is in C."""
    # The tangent lines through the origin are the directions n with $h_C(n) = 0$: bracket
    # the sign changes of $h_C$ on an angular grid, then refine with brentq. C lies on the
    # side $n \cdot x \le 0$ of both lines.
    h_of = lambda angle: C.support(_direction(angle))
    phi = np.linspace(0, 2 * np.pi, n_grid)
    h = np.array([h_of(a) for a in phi])
    idx = np.where(np.sign(h[:-1]) != np.sign(h[1:]))[0] + 1
    if len(idx) == 0:
        return None
    normals = np.array([_direction(brentq(h_of, phi[i - 1], phi[i])) for i in idx])
    contacts = np.array([C.support_point(d) for d in normals])
    edges = np.array([s / np.linalg.norm(s) for s in contacts])
    return normals, edges, contacts


def _in_cone(w, edges):
    """Whether w (..., 2) lies in the cone spanned by ``edges``."""
    # Inside means within the half-angle of the bisector.
    e1, e2 = edges
    b = e1 + e2
    b = b / np.linalg.norm(b)
    wn = np.linalg.norm(w, axis=-1)
    moving = wn > MIN_REL_SPEED
    cos_w = (w / np.where(moving, wn, 1.0)[..., None]) @ b
    return moving & (cos_w >= e1 @ b - 1e-9)


def _scalar_if_single(mask, v):
    return bool(mask) if np.ndim(v) == 1 else mask


class VelocityObstacle:
    """Own velocities that lead to a collision with a constant-velocity target.

    VO = target.vel + cone(C), the cone from the origin over the config-space obstacle C.
        apex: the target velocity.
        edges: (2, 2) unit edge directions, measured from the apex.
        normals: (2, 2) outward unit normals of the two edges.
        tangent_points: (2, 2) tangent points on C, in relative-position space.
        half_angle: half the opening angle, radians.
        already_in_collision: True when the ships overlap at t = 0 (the origin lies in C).
            Every own velocity is then in the VO, and edges, normals and tangent_points are None.
    """

    def __init__(self, own, target, n_grid=1440):
        self.apex = np.asarray(target.vel, dtype=float)
        C = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
        cone = _collision_cone(C, n_grid=n_grid)
        self.already_in_collision = cone is None
        self.normals, self.edges, self.tangent_points = (None, None, None) if cone is None else cone

    def contains(self, v_own):
        """Whether own velocities ``v_own``, shape (2,) or (..., 2), are on a collision course."""
        v = np.asarray(v_own, dtype=float)
        if self.already_in_collision:
            return _scalar_if_single(np.ones(v.shape[:-1], dtype=bool), v)
        return _scalar_if_single(_in_cone(v - self.apex, self.edges), v)

    @property
    def half_angle(self):
        return _half_angle(self.edges)

    def outline(self, length):
        """Boundary points for plotting, with the infinite edges cut off at ``length``."""
        e0, e1 = self.edges
        return np.array([self.apex + length * e0, self.apex, self.apex + length * e1])


def _half_angle(edges):
    return None if edges is None else float(np.arccos(np.clip(edges[0] @ edges[1], -1.0, 1.0)) / 2)


def _outer_arc(a_from, a_to, a_through, n):
    """``n`` angles from a_from to a_to along the arc that passes through a_through."""
    span = (a_to - a_from) % (2 * np.pi)
    if (a_through - a_from) % (2 * np.pi) <= span:
        return np.linspace(a_from, a_from + span, n)
    return np.linspace(a_from, a_from - (2 * np.pi - span), n)


class UncertaintyAwareVelocityObstacle:
    r"""The velocity obstacle when the relative state is uncertain.

    Holds every own velocity that collides for some pair of errors: some $k_p \in K_p$ and
    $k_v \in K_v$ with relpos + $k_p$, relvel + $k_v$ colliding. As a set it is
    $target.vel + cone(C \oplus K_p) \oplus K_v$.

    K_p, K_v: the error sets on relpos and relvel, convex Shapes centered at the origin
    (e.g. ``Gaussian(Sigma).uncertainty_set(2)``), or None for no uncertainty in that channel. K_p widens
    the cone; K_v offsets its edges without rotating them and rounds the apex into ``cap``.

    Attributes are as in ``VelocityObstacle``, for the cone over $C \oplus K_p$, plus (also
    None when ``already_in_collision``):
        contacts: (2, 2) points where the offset edges meet the cap.
        cap: (n_cap, 2) the rounded apex, from contacts[0] to contacts[1].
    """

    def __init__(self, own, target, K_p=None, K_v=None, n_grid=1440, n_cap=64):
        self.K_p, self.K_v = K_p, K_v
        self.apex = a = np.asarray(target.vel, dtype=float)
        C = config_space_obstacle(own.domain, target.domain, target.pos - own.pos)
        if K_p is not None:
            C = MinkowskiSum(C, K_p)
        cone = _collision_cone(C, n_grid=n_grid)
        self.already_in_collision = cone is None
        if cone is None:
            self.normals = self.edges = self.tangent_points = self.contacts = self.cap = None
            return
        self.normals, self.edges, self.tangent_points = cone

        # K_v is the error set on relvel, so in own-velocity space it adds to the cone: each edge
        # offsets by $h_{K_v}(n)$ along its normal, and the apex rounds into cap.
        K = Circle(0.0) if K_v is None else K_v
        (n1, n2), (e1, e2) = self.normals, self.edges
        self.contacts = np.array([a + K.support_point(n1), a + K.support_point(n2)])
        b = e1 + e2
        b = b / np.linalg.norm(b)
        arc = lambda n: _outer_arc(np.arctan2(n1[1], n1[0]), np.arctan2(n2[1], n2[0]),
                                   np.arctan2(-b[1], -b[0]), n)
        self.cap = np.array([a + K.support_point(_direction(t)) for t in arc(n_cap)])
        if K_v is not None:
            self._arc_dirs = np.array([_direction(t) for t in arc(_N_ARC)])
            self._arc_h = np.array([K_v.support(d) for d in self._arc_dirs])

    def contains(self, v_own):
        """Whether own velocities ``v_own``, shape (2,) or (..., 2), lie in the inflated VO."""
        v = np.asarray(v_own, dtype=float)
        if self.already_in_collision:
            return _scalar_if_single(np.ones(v.shape[:-1], dtype=bool), v)
        x = v - self.apex
        if self.K_v is None:
            return _scalar_if_single(_in_cone(x, self.edges), v)
        # cone $\oplus K_v$ has no closed form, so test membership by support functions: x is
        # inside when $d \cdot x \le h_{K_v}(d)$ for every direction d of the polar arc (off the
        # arc the cone's support is infinite, so those directions constrain nothing). Each d gives
        # one half-plane and the true set is the intersection of all of them; intersecting only
        # _N_ARC leaves the tested set a little too large, so a velocity that just misses can
        # come back as a collision. A real collision is never reported as a miss. The margin is
        # under 2.5e-6 r, for r the farthest point of K_v from the origin.
        inside = np.all(x @ self._arc_dirs.T <= self._arc_h + 1e-9, axis=-1)
        return _scalar_if_single(inside, v)

    @property
    def half_angle(self):
        return _half_angle(self.edges)

    def outline(self, length):
        """Boundary points for plotting, with the infinite edges cut off at ``length``."""
        (c0, c1), (e0, e1) = self.contacts, self.edges
        return np.vstack([c0 + length * e0, self.cap, c1 + length * e1])
