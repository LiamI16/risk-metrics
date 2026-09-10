# risk-metrics

Collision risk metrics for unstructured environments — no lanes or paths, so risk is
read off the relative kinematics and geometry of nearby agents. Two representations of
the same risk, and the relationship between them:

- **CPA metrics.** DCPA (minimum separation over the horizon) and TCPA (when it occurs)
  collapse an encounter to two numbers, evaluated at one candidate velocity.
- **The velocity obstacle.** The *set* of ego velocities that collide at some `t`, which
  is what evasive control needs.

Under disc geometry and exact observations the two are equivalent, and the equivalence
is a closed-form membership test. They come apart under uncertainty, which is what the
sampling half of the repo measures.

Supporting code for *Systematic Comparison of Collision Risk Metrics* (WIP, ACC
submission).

## Packages

**`minkowski_utils`** — convex 2-D geometry via support functions: `Circle`, `Ellipse`,
`Polygon`, `MinkowskiSum`. Compact convex bodies only.

**`vo_utils`** — risk metrics on top of it:

| module | |
|---|---|
| `ship`, `metrics` | agents (position, constant velocity, convex domain); DCPA/TCPA and `.vo()` |
| `cone`, `obstacle` | collision cone for a general convex body; `VelocityObstacle` |
| `uncertainty`, `uncertainty_models` | confidence bodies (`Gaussian`, `UniformBox`, `UniformEllipse`, `IndependentSum`) and the inflated VOs they induce |
| `sampling`, `scenario`, `scales` | `sample_cpa`, 13 canonical encounters, catalog-wide shared plot scales |
| `plotting` | matplotlib lives only here — three/four/six/nine-panel figures |

Compute core is numpy/scipy; matplotlib is needed only for the two `plotting` modules.

The paper writes ego `E` and obstacle `O`, with $r = \mathbf{p}_O - \mathbf{p}_E$;
in code these are `own` and `target`.

## What it implements

**CPA metrics**, with relative position $r$ and velocity $v$:

$$\mathrm{TCPA} = -\frac{r \cdot v}{\lVert v \rVert^2}, \qquad \mathrm{DCPA}_s = \frac{v \times r}{\lVert v \rVert}$$

DCPA is kept **signed** in `sampling`, so the side of the encounter survives;
$\mathrm{DCPA} = |\mathrm{DCPA}_s|$ folds the two sides together, which changes the shape
of the distribution under uncertainty.

**Equivalence.** For disc agents with combined radius $R$ and exact observations,

$$\exists\, t \ge 0 : \lVert r + vt \rVert \le R \iff (\mathrm{TCPA} > 0 \wedge \mathrm{DCPA} \le R) \vee (\mathrm{TCPA} \le 0 \wedge \lVert r \rVert \le R)$$

with a third branch on a finite horizon $[0,\tau]$: $\mathrm{TCPA} > \tau \wedge \lVert r + v\tau \rVert \le R$

**Support functions.** Minkowski sums are pointwise addition, $h_{A \oplus B} = h_A +
h_B$, so every convex body reduces to two queries — $h_S(d)$ and a support point. The
collision cone is then the two roots of $g(\phi) = h_\mathcal{C}(\hat d(\phi))$, bracketed
on a grid and refined by Brent's method. Discs have those roots in closed form; the
root-finding is what buys arbitrary convex geometry.

**Bounded convex uncertainty.** Position error inflates the collision set;
velocity error inflates the resulting velocity obstacle:

$$\mathrm{VO} = \big[\text{cone of } \mathcal{E} \oplus (-\mathcal{O}) \oplus \mathcal{K}_p \big] \oplus \mathcal{K}_v$$

Each inflation is one added support term. $\mathcal{K}$ may be any compact convex body;
for Gaussian error at level $k$ it is the $k\sigma$ ellipse, $h(d) = k\sqrt{d^\top \Sigma
d}$. The velocity channel is an exact Minkowski sum rather than a cone enclosing it, so
the apex is not over-approximated.

**Sampling.** That inflation only *bounds* the induced risk. Drawing $(r, v)$ directly
gives the (TCPA, DCPA) distribution and `P[collision]`: the nominal cone measures as the
~50% contour, and the $k\sigma$-inflated VO is an outer bound over realizations, not a
probability level set.

## Run

    pip install -e ".[plot]"
    python3 main.py
