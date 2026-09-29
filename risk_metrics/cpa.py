"""Closest point of approach and the collision test for two circles."""

import numpy as np

# Relative speeds below this (m/s) count as no relative motion
MIN_REL_SPEED = 1e-8


def _dot(a, b):
    return np.einsum("...i,...i->...", a, b)


def cpa(relpos, relvel):
    r"""TCPA and signed DCPA, broadcast over (..., 2) inputs.

    The signed DCPA is undefined for relvel at or below MIN_REL_SPEED. The second return is
    then the unsigned distance ||relpos||, and TCPA is 0.
    """
    relpos = np.asarray(relpos, dtype=float)
    relvel = np.asarray(relvel, dtype=float)
    speed_sq = _dot(relvel, relvel)
    moving = speed_sq > MIN_REL_SPEED ** 2
    safe = np.where(moving, speed_sq, 1.0)
    tcpa = np.where(moving, -_dot(relpos, relvel) / safe, 0.0)
    cross = relvel[..., 0] * relpos[..., 1] - relvel[..., 1] * relpos[..., 0]
    dcpa = np.where(moving, cross / np.sqrt(safe), np.linalg.norm(relpos, axis=-1))
    return tcpa[()], dcpa[()]           # unwraps the np.where output


def collides(relpos, relvel, R, horizon=None):
    r"""Whether $\|relpos + relvel\,t\|^2 \le R^2$ for some $t \in [0, horizon]$ (None: unbounded).
    """
    relpos = np.asarray(relpos, dtype=float)
    relvel = np.asarray(relvel, dtype=float)
    a = _dot(relvel, relvel)
    b = 2.0 * _dot(relpos, relvel)
    c = _dot(relpos, relpos) - R ** 2
    moving = a > MIN_REL_SPEED ** 2
    discriminant = b ** 2 - 4.0 * a * c
    root = np.sqrt(np.maximum(discriminant, 0.0))
    two_a = 2.0 * np.where(moving, a, 1.0)
    hit = (discriminant >= 0.0) & ((-b + root) / two_a >= 0.0)
    if horizon is not None:
        hit &= (-b - root) / two_a <= horizon
    return np.where(moving, hit, c <= 0.0)[()]
