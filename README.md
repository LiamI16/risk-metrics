# risk-metrics

Convex Minkowski-sum geometry and velocity obstacles under uncertainty.

Velocity obstacles for maritime encounters, extended to handle **arbitrary
convex ship domains** (not just discs) and **Gaussian positional uncertainty**.
Both fall out of one idea: representing convex bodies by their support
functions, so Minkowski sums add.

## Packages

- **`minkowski_utils`** : convex 2-D geometry via support functions
  (`Circle`, `Ellipse`, `Polygon`, `MinkowskiSum`). numpy/scipy only.
- **`vo_utils`** : velocity obstacles built on top: `Ship`, `Metrics`
  (DCPA/TCPA), the deterministic collision cone / `VelocityObstacle`, and
  `k`-sigma probabilistic cones under positional covariance.

## Key identities

Minkowski sum = support-function sum:

$$h_{A \oplus B}(d) = h_A(d) + h_B(d)$$

$k$-sigma-inflated obstacle (positional uncertainty $\Sigma$):

$$O_k = O \oplus k\, E_\Sigma, \qquad h_{O_k}(d) = h_O(d) + k\sqrt{d^\top \Sigma\, d}$$

## Run

    pip install -e ".[plot]"
    python3 main.py
