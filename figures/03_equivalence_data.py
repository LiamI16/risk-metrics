"""
Script to generate csv data for Fig 3.
"""

import sys
from pathlib import Path
import numpy as np
from risk_metrics import collides, cpa

OUT = Path("output/figures/equivalence.csv")
SEED = 0
RADIUS = 10.0     # per ship, so R = 20
RELPOS = np.array([48.0, 36.0])     # target relative to own, m
V_TARGET = np.array([-4.0, 2.0])    # m/s
D_FRAC = 0.6     # the pair's shared |DCPA|, as a fraction of R
SPEED = 10.0     # the pair's closing speed, m/s
N = 8000     # candidate own velocities
PAD = 0.25     # frame padding, and the sampling window


def equal_dcpa_pair(relpos, v_target, R, d_frac=D_FRAC, speed=SPEED):
    p = np.asarray(relpos, dtype=float)
    dist = float(np.linalg.norm(p))
    phi = np.arcsin(d_frac * R / dist)
    ct, st = np.cos(phi), np.sin(phi)
    u = (-p / dist) @ np.array([[ct, st], [-st, ct]])
    w = speed * u
    return np.asarray(v_target) - w, np.asarray(v_target) + w


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    R = RADIUS + RADIUS

    # We need a frame that includes the points that have to be visible. Compute the counterexample pair,
    # the velocity pair that is inside and outside the VO, then construct a box around those points.
    v_conflict, v_safe = equal_dcpa_pair(RELPOS, V_TARGET, R)
    must_show = np.array([(0.0, 0.0), V_TARGET, v_conflict, v_safe])
    min, max = must_show.min(axis=0), must_show.max(axis=0)
    center, half = 0.5 * (min + max), 0.5 * float((max - min).max()) * (1 + PAD)
    v_own = np.random.default_rng(SEED).uniform(center - half, center + half, size=(N, 2))

    relvel = V_TARGET - v_own
    tcpa, dcpa_s = cpa(RELPOS, relvel)
    dcpa = np.abs(dcpa_s)     # we want DCPA
    truth = collides(RELPOS, relvel, R)     # Theorem 1's left-hand side

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, np.column_stack([v_own[:, 0], v_own[:, 1], dcpa, tcpa, truth]),
               delimiter=",", header="x,y,dcpa,tcpa,collides", comments="",
               fmt=["%.3f"] * 4 + ["%d"])
    
    tests = {"DCPA <= R alone": dcpa <= R,
             "TCPA > 0 alone": tcpa > 0,
             "both together": (dcpa <= R) & (tcpa > 0)}
    print(f"wrote {out} ({N} samples, seed {SEED}, {int(truth.sum())} colliding)\n")
    print(f"  {'as a VO test':<18}{'wrong':>7}{'of ' + str(N):>10}")
    print(f"  {'-' * 35}")
    for label, holds in tests.items():
        n = int((holds & ~truth).sum())
        print(f"  {label:<18}{n:>7}{n / N:>9.1%}")
