from typing import NamedTuple
import numpy as np
from .cpa import cpa

class CPA_Samples(NamedTuple):
    """Sampled CPA metrics, shape (n,), with the relative states (n, 2) behind them."""

    tcpa: np.ndarray
    dcpa: np.ndarray
    relpos: np.ndarray
    relvel: np.ndarray


def sample_cpa(own, target, pos_noise=None, vel_noise=None, n=10000, rng=None):
    rng = np.random.default_rng() if rng is None else rng

    def draw(noise, mean):
        if noise is None:
            return np.tile(mean, (n, 1))
        return mean + noise.sample(n, rng)

    relpos = draw(pos_noise, target.pos - own.pos)
    relvel = draw(vel_noise, target.vel - own.vel)
    tcpa, dcpa = cpa(relpos, relvel)
    return CPA_Samples(tcpa, dcpa, relpos, relvel)
