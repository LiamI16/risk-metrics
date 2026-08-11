"""Monte-Carlo sampling of the (TCPA, DCPA) distribution induced by uncertainty."""

from typing import NamedTuple

import numpy as np


class CpaSamples(NamedTuple):
    """One encounter's sampled closest-approach metrics, with the draws behind them."""

    dcpa: np.ndarray    # (n,)  SIGNED closest-approach distances
    tcpa: np.ndarray    # (n,)  times to closest approach (negative => already past)
    relpos: np.ndarray  # (n,2) sampled relative positions r
    relvel: np.ndarray  # (n,2) sampled relative velocities v

    @property
    def miss_distance(self):
        """Unsigned closest-approach distance ``|dcpa|``."""
        return np.abs(self.dcpa)


def _draw(spec, mean, n, rng):
    """Draw ``n`` samples about ``mean`` from a channel's noise spec.

    ``spec`` is None (exact), an UncertaintyModel, or a bare 2x2 covariance, which is
    treated as ``Gaussian(spec)`` so callers holding a plain Sigma (scenarios,
    ``anchored_covariances``) need not wrap it.
    """
    mean = np.asarray(mean, dtype=float)
    if spec is None:
        return np.tile(mean, (n, 1))
    if hasattr(spec, "sample"):
        return mean + spec.sample(n, rng)
    return rng.multivariate_normal(mean, np.asarray(spec, dtype=float), size=n)


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


def uniform_like(Sigma, kind="ellipse"):
    """A bounded uniform model with the same covariance as ``N(0, Sigma)``.

    Equal-covariance anchoring, so a Gaussian-vs-uniform comparison isolates
    distribution *shape* at matched spread rather than confounding it with scale.

    Parameters
    ----------
    Sigma : array_like, shape (2, 2)
        Covariance to match. Its eigenvectors set the body's axes.
    kind : {'ellipse', 'box'}
        ``'ellipse'`` gives semi-axes ``2*sqrt(lambda)`` (a uniform ellipse of
        semi-axis ``2*sigma`` has per-axis variance ``sigma^2``), so its hard edge is
        the Gaussian 2-sigma ellipse. ``'box'`` gives half-widths ``sqrt(3)*sqrt(lambda)``
        (a uniform interval of half-width ``h`` has variance ``h^2/3``).

    Returns
    -------
    UniformEllipse or UniformBox

    Raises
    ------
    ValueError
        If ``kind`` is neither ``'ellipse'`` nor ``'box'``.
    """
    from .uncertainty_models import UniformEllipse, UniformBox
    vals, vecs = np.linalg.eigh(np.asarray(Sigma, dtype=float))
    order = np.argsort(vals)[::-1]                       # major (largest) axis first
    vals, vecs = vals[order], vecs[:, order]
    theta = float(np.arctan2(vecs[1, 0], vecs[0, 0]))    # angle of the major eigenvector
    sig = np.sqrt(np.maximum(vals, 0.0))
    if kind == "ellipse":
        return UniformEllipse(a=2.0 * sig[0], b=2.0 * sig[1], theta=theta)
    if kind == "box":
        return UniformBox(half_widths=np.sqrt(3.0) * sig, theta=theta)
    raise ValueError(f"kind must be 'ellipse' or 'box', got {kind!r}")


def sample_cpa(own, target, Sigma_pos=None, Sigma_vel=None, n=10000, rng=None):
    """Monte-Carlo (signed DCPA, TCPA) for one encounter under relative uncertainty.

    Draws are taken about ``r_hat = target.pos - own.pos`` and
    ``v_hat = target.vel - own.vel``; only the relative pair matters, so own and
    target-side uncertainty enter through their sum.

    Parameters
    ----------
    own, target : Ship
        The two ships.
    Sigma_pos, Sigma_vel : array_like or UncertaintyModel, optional
        Per-channel relative noise: None (exact), a 2x2 covariance (Gaussian), or a
        model exposing ``sample(n, rng)``. None leaves that channel exact.
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

    r = _draw(Sigma_pos, r_hat, n, rng)                             # (n,2)
    v = _draw(Sigma_vel, v_hat, n, rng)                             # (n,2)

    speed_sq = np.einsum("ij,ij->i", v, v)                           # ||v||^2
    moving = speed_sq > 1e-12

    tcpa = np.zeros(n)                                               # 0 where no relative motion
    np.divide(-np.einsum("ij,ij->i", r, v), speed_sq, out=tcpa, where=moving)

    cross = v[:, 0] * r[:, 1] - v[:, 1] * r[:, 0]                    # v x r (signed)
    speed = np.where(moving, np.sqrt(speed_sq), 1.0)                 # avoid 0/0 at static draws
    dcpa = np.where(moving, cross / speed,                           # signed miss distance
                    np.linalg.norm(r, axis=1))                       # -> ||r|| (unsigned) when static

    return CpaSamples(dcpa, tcpa, r, v)
