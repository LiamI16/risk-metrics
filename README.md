# risk-metrics

Collision risk between two constant-velocity agents: the closest-point-of-approach (CPA) metrics (TCPA, DCPA), the velocity obstacle (VO), and how position and velocity uncertainty propagate through each. Agents can have any compact convex shape. Uncertainty is handled two ways:
- **set-based:** convex uncertainty sets inflate the VO;
- **sampling-based:** Monte Carlo draws give the distribution of (TCPA, DCPA) and P[collision].

This repository accompanies the paper *Systematic Comparison of Collision Risk Metrics* (title not finalized yet?).

## Installation

Needs Python 3.12+. The library itself uses only numpy and scipy; matplotlib is used by the figure and example scripts, and pytest by the tests.

```bash
git clone <repository-url> && cd risk-metrics
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # versions used for the published figures
pip install -e .
```

## Quickstart

```python
import numpy as np
from risk_metrics import (Ship, Circle, Gaussian, UniformEllipse, cpa, collides,
                          sample_cpa, VelocityObstacle,
                          UncertaintyAwareVelocityObstacle)

own = Ship(pos=(0, 0), vel=(5, 0), domain=Circle(10))
target = Ship(pos=(100, 100), vel=(0, -5), domain=Circle(10))
R = 20.0  # sum of the two radii

tcpa, dcpa = cpa(target.pos - own.pos, target.vel - own.vel)  # 20.0 s, 0.0 m
collides(target.pos - own.pos, target.vel - own.vel, R, horizon=10.0)  # False

vo = VelocityObstacle(own, target)
vo.contains(own.vel)  # True; also accepts (..., 2) arrays

Sp, Sv = 25 * np.eye(2), 0.25 * np.eye(2)  # position [m^2], velocity [(m/s)^2]
uvo = UncertaintyAwareVelocityObstacle(
    own, target,
    K_p=Gaussian(Sp).uncertainty_set(2),
    K_v=Gaussian(Sv).uncertainty_set(2))

s = sample_cpa(own, target, pos_noise=Gaussian(Sp),
               vel_noise=UniformEllipse.from_covariance(Sv),
               n=5000, rng=np.random.default_rng(0))
p_collision = collides(s.relpos, s.relvel, R).mean()
```

## Conventions

- **Axes and units.** x = East, y = North, for both positions and velocities. m, m/s, s; angles in radians.
- **Relative states.** `relpos = p_target - p_own` and `relvel = v_target - v_own`. In paper notation own is the ego E and target the obstacle O, so these are $\mathbf{p}_\mathrm{rel}$ and $\mathbf{v}_\mathrm{rel}$.
- **Signed DCPA.** $\mathrm{DCPA}_s = (\mathbf{v}_\mathrm{rel} \times \mathbf{p}_\mathrm{rel}) / \lVert \mathbf{v}_\mathrm{rel} \rVert$. Its magnitude $\mathrm{DCPA} = |\mathrm{DCPA}_s|$ is the distance at closest approach; the sign records which side the target passes.
- **Which side is positive.** $\mathrm{DCPA}_s > 0$ when the target lies to the left of $\mathbf{v}_\mathrm{rel}$, equivalently to the right of $-\mathbf{v}_\mathrm{rel}$ (the own ship's motion relative to the target). For a closing scenario such as head-on or crossing that is a pass to starboard. It does not hold in general, for instance when being overtaken from astern.
- **Degenerate cases.** A negative TCPA means the closest approach is already past. With `relvel = 0` there is no track direction, so $\mathrm{TCPA} = 0$ and $\mathrm{DCPA}$ is the unsigned $\lVert \mathbf{p}_\mathrm{rel} \rVert$. Note: $\mathrm{DCPA}_s$ is undefined for `relvel = 0`.
- **Names in the paper.** The code's `dcpa` is the paper's signed $\mathrm{DCPA}_s$; the paper's plain $\mathrm{DCPA}$ is `abs(dcpa)`. Likewise `relpos`/`relvel` are $\mathbf{p}_\mathrm{rel}$/$\mathbf{v}_\mathrm{rel}$, and a ship's `domain` is its convex set $\mathcal{E}$ or $\mathcal{O}$. The code writes `C` for the config-space obstacle, kept distinct from the paper's $\mathcal{O}$.
- **CPA treats the agents as points.** `cpa()` uses the reference positions only, so for non-circular domains TCPA/DCPA ignore the shapes.
- **Horizons.** The VO and the uncertainty-aware VO are infinite-horizon; a finite horizon is available only in `collides`.
- **Velocity obstacles live in own-velocity space.** A VO is the set of own velocities that collide, with its apex at the target velocity. Because $\mathbf{v}_\mathrm{own} = \mathbf{v}_\mathrm{target} - \mathbf{v}_\mathrm{rel}$, sets map with a sign flip between the two spaces:
  - velocity uncertainty $\mathcal{K}_v$ is a set of errors on $\mathbf{v}_\mathrm{rel}$. The VO grows by $+\mathcal{K}_v$ in own-velocity space. That same growth is $\oplus(-\mathcal{K}_v)$ in relative-velocity space.
  - the paper's collision set is $\mathcal{E} \oplus (-\mathcal{O})$. The code's `config_space_obstacle` builds its reflection, $\mathcal{O} \oplus (-\mathcal{E})$, and tests $-\mathbf{v}_\mathrm{rel}$ against it. The two sign flips cancel, so the answers match.
- **Per-ship errors compose with a reflection.** $\mathcal{K}_p$ and $\mathcal{K}_v$ are errors on the relative state, so a pair of per-ship errors enters as $e_\mathrm{rel} = e_\mathrm{target} - e_\mathrm{own}$: compose them as `target_noise + own_noise.reflect()`. Plain `a + b` is exact only for centrally symmetric errors.
- **Reflect once.** `own_noise.reflect().uncertainty_set(k)` and `own_noise.uncertainty_set(k).reflect()` give the same set, and doing both cancels back to the unreflected one. `reflect()` returns `self` for the three models here and a genuine reflection for any other convex body.
- **Uncertainty levels.** `Gaussian(Sigma).uncertainty_set(k)` is the $k\sigma$ confidence ellipse; for the bounded models `uncertainty_set(1)` is the support. Neither the $k\sigma$ set nor the inflated VO is a probability level set: the inflated VO is a conservative outer bound.

## Package

| module | contents |
|---|---|
| `geometry` | `Shape` and `Circle`, `Ellipse`, `Polygon`, `MinkowskiSum`, all described by their support functions; `covariance_ellipse` |
| `distributions` | `Distribution` and `Gaussian`, `UniformEllipse`, `UniformBox` (`.from_covariance` gives equal-covariance uniform models); `a + b` combines independent errors |
| `cpa` | `cpa` (TCPA and signed DCPA) and `collides` (exact collision test over an infinite or finite horizon, independent of TCPA/DCPA) |
| `vo` | `VelocityObstacle`, `UncertaintyAwareVelocityObstacle`, `config_space_obstacle` |
| `sampling` | `sample_cpa`, which returns `CPA_Samples(tcpa, dcpa, relpos, relvel)` |
| `scenarios` | `Ship`, `Scenario`, 13 named scenarios in `CANONICAL_SCENARIOS` |

Shapes are held as support functions rather than as outlines, so a Minkowski sum costs one query per part instead of constructing the total Minkowski sum.

**How the pieces fit.** Two layers, and one crossing point between them.

```
Shape                      Circle, Ellipse, Polygon, MinkowskiSum
  |
  |-- a ship's domain      Ship(domain=Circle(10))
  '-- an uncertainty set   UncertaintyAwareVelocityObstacle(K_p=, K_v=)

Distribution               Gaussian, UniformEllipse, UniformBox
  |
  |-- .sample(n, rng)      sample_cpa(pos_noise=, vel_noise=)
  '-- .uncertainty_set(k)  a Shape, so a noise model can serve as K_p or K_v
```

`sample_cpa` takes `Distribution`s and the uncertainty-aware VO takes `Shape`s, so
`uncertainty_set(k)` is how one noise model serves both. Note `k` differs by family: for
`Gaussian` it counts standard deviations, so `uncertainty_set(2)` is the $2\sigma$ ellipse,
while for the bounded models `uncertainty_set(1)` is already the entire support.

**Extending.** Two things are designed to be extended, both shown in `examples/`:
- **A new convex shape:** subclass `Shape` and implement `support(d)`, `support_point(d)` and `reflect()`. Minkowski sums and VOs then work with it unchanged.
- **A new noise model:** subclass `Distribution` and implement `sample(n, rng)`; `sample_cpa` needs only that. Also implement `uncertainty_set(k)`, returning a convex `Shape`, to use the model in `UncertaintyAwareVelocityObstacle`. `reflect()` is inherited correctly for any model; override it with `return self` if yours is centrally symmetric.

## Examples

Run from the repository root; each script prints its results and saves a figure to `output/examples/`.

| script | shows |
|---|---|
| `examples/00_shapes.py` | convex shapes, reflection and Minkowski sums, and the config-space obstacle they build |
| `examples/01_cpa_and_vo.py` | TCPA/DCPA, the collision test, VO membership, in absolute, relative and velocity space; flags `--scenario NAME`, `--all` (every preset scenario drawn) |
| `examples/02_convex_domains.py` | polygon/ellipse agents, and why two noise models of equal covariance give the same uncertainty-aware VO |
| `examples/03_uncertainty_sampling.py` | (TCPA, DCPA) and P[collision] under Gaussian vs. uniform noise of equal covariance |
| `examples/04_scenario_grid.py` | all 13 scenarios on one shared scale; flags `--constant-cov`, `--symlog` |
| `examples/05_uncertainty_aware_vo.py` | nested $k\sigma$ uncertainty-aware VOs: position vs. velocity uncertainty, and both combined |

## Reproducing the figures

main.py (Explain here later)

## Tests

```bash
pytest
```

93 tests, covering the exact collision test against the TCPA/DCPA characterization over both horizons, VO membership against simulation, and the uncertainty-aware VO against brute force with asymmetric uncertainty sets.
