"""Minkowski sums of convex shapes via support-function addition.

The Minkowski sum  A (+) B = { a + b : a in A, b in B }. For convex bodies the
support function linearizes it

    h_{A(+)B}(d) = h_A(d) + h_B(d)
    s_{A(+)B}(d) = s_A(d) + s_B(d)
"""

import numpy as np

from .shapes import Shape


class MinkowskiSum(Shape):
    """Minkowski sum of convex shapes.  Itself a Shape, so sums compose."""

    def __init__(self, *shapes):
        parts = []
        for s in shapes:
            if isinstance(s, MinkowskiSum):
                parts.extend(s.parts)          # flatten nested sums
            elif isinstance(s, Shape):
                parts.append(s)
            else:
                raise TypeError(f"MinkowskiSum parts must be Shape, got {type(s)}")
        if not parts:
            raise ValueError("MinkowskiSum needs at least one shape")
        self.parts = parts

    def support(self, d):
        return np.sum([p.support(d) for p in self.parts])

    def support_point(self, d):
        return np.sum([p.support_point(d) for p in self.parts], axis=0)
    

    def reflect(self):                         # -(A(+)B) = (-A)(+)(-B)
        return MinkowskiSum(*[s.reflect() for s in self.parts])

    def __repr__(self):
        return "MinkowskiSum(" + ", ".join(repr(p) for p in self.parts) + ")"

def minkowski_sum(*shapes):
    """Convenience constructor: minkowski_sum(A, B, ...) -> MinkowskiSum."""
    return MinkowskiSum(*shapes)
