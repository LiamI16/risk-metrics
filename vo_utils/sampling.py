from typing import NamedTuple

import numpy as np


class CpaSamples(NamedTuple):
    dcpa: np.ndarray    # (n,)  closest-approach distances (>= 0)
    tcpa: np.ndarray    # (n,)  times to closest approach (negative => already past)
    relpos: np.ndarray  # (n,2) sampled relative positions r
    relvel: np.ndarray  # (n,2) sampled relative velocities v


def _as_cov(Sigma):
    """A 2x2 covariance, or zero matrix when Sigma is None."""
    return np.zeros((2, 2)) if Sigma is None else np.asarray(Sigma, dtype=float)


def anchored_covariances(own, target, alpha, beta, vel_scale=None):
    """Isotropic relative covariances anchored to the encounter scale.

    ``sigma_pos = alpha * R`` (with ``R`` the summed radii) and ``sigma_vel = beta *
    vel_scale``.

    Parameters
    ----------
    own, target : Ship
        The two ships.
    alpha : float
        Position-noise scale, as a fraction of ``R``.
    beta : float
        Velocity-noise scale, as a fraction of ``vel_scale``.
    vel_scale : float, optional
        Velocity anchor. Defaults to the closing speed ``||v_rel||``; pass an
        absolute speed for low-closing-speed encounters where that vanishes.

    Returns
    -------
    Sigma_pos, Sigma_vel : ndarray, shape (2, 2)
        ``(sigma_pos^2 I, sigma_vel^2 I)``.
    """
    R = own.domain.r + target.domain.r
    if vel_scale is None:
        vel_scale = float(np.linalg.norm(np.asarray(target.vel) - np.asarray(own.vel)))
    sp, sv = alpha * R, beta * vel_scale
    return sp ** 2 * np.eye(2), sv ** 2 * np.eye(2)


def shaped_covariance(sigma, ecc, major_axis):
    """An anisotropic 2x2 covariance from scale, eccentricity, and major-axis direction.

    Parameters
    ----------
    sigma : float
        Geometric-mean standard deviation.
    ecc : float
        Eccentricity in [0, 1); ``ecc = 0`` returns ``sigma^2 I``.
    major_axis : array_like, shape (2,)
        Major-axis direction (pass a perpendicular direction to flip the ellipse).

    Returns
    -------
    ndarray, shape (2, 2)
    """
    u = np.asarray(major_axis, dtype=float)
    u = u / np.linalg.norm(u)
    perp = np.array([-u[1], u[0]])
    f = (1.0 - ecc ** 2) ** 0.25
    s_major, s_minor = sigma / f, sigma * f
    return s_major ** 2 * np.outer(u, u) + s_minor ** 2 * np.outer(perp, perp)


def sample_cpa(own, target, Sigma_pos=None, Sigma_vel=None, n=10000, rng=None):
    """Monte-Carlo (DCPA, TCPA) for one encounter under Gaussian relative uncertainty.

    Means are ``r_hat = target.pos - own.pos`` and ``v_hat = target.vel - own.vel``.
    Degenerate draws with ``||v|| ~ 0`` follow the Metrics convention: TCPA = 0,
    DCPA = ``||r||``.

    Parameters
    ----------
    own, target : Ship
        The two ships.
    Sigma_pos, Sigma_vel : array_like, shape (2, 2), optional
        Relative covariances; None leaves that channel exact.
    n : int
        Number of samples.
    rng : numpy.random.Generator, optional
        Random generator; defaults to a fresh one.

    Returns
    -------
    CpaSamples
        Arrays of length ``n``, also carrying the sampled ``r``, ``v``.
    """
    rng = np.random.default_rng() if rng is None else rng

    r_hat = np.asarray(target.pos, dtype=float) - np.asarray(own.pos, dtype=float)
    v_hat = np.asarray(target.vel, dtype=float) - np.asarray(own.vel, dtype=float)

    r = rng.multivariate_normal(r_hat, _as_cov(Sigma_pos), size=n)   # (n,2)
    v = rng.multivariate_normal(v_hat, _as_cov(Sigma_vel), size=n)   # (n,2)

    speed_sq = np.einsum("ij,ij->i", v, v)                           # ||v||^2
    moving = speed_sq > 1e-12

    tcpa = np.zeros(n)                                               # 0 where no relative motion
    np.divide(-np.einsum("ij,ij->i", r, v), speed_sq, out=tcpa, where=moving)

    cross = r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]                    # r x v
    speed = np.where(moving, np.sqrt(speed_sq), 1.0)                 # avoid 0/0 at static draws
    dcpa = np.where(moving, np.abs(cross) / speed,
                    np.linalg.norm(r, axis=1))                       # -> ||r|| when static

    return CpaSamples(dcpa, tcpa, r, v)
