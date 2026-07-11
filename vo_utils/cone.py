"""Collision cones for velocity obstacles
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import brentq

from minkowski_utils import Shape


def _dir(phi):
    """Unit direction at angle phi."""
    return np.array([np.cos(phi), np.sin(phi)])


@dataclass
class CollisionCone:
    contains_origin: bool
    normals: np.ndarray | None = None    # (2,2) tangent-line outward normals d1, d2
    edges: np.ndarray | None = None      # (2,2) unit rays from origin along cone edges
    contacts: np.ndarray | None = None   # (2,2) tangent contact points on O


def _cone_from_normals(O, normals):
    """Package tangent normals into a CollisionCone (contacts + edge rays)."""
    contacts = np.array([O.support_point(d) for d in normals])
    edges = np.array([s / np.linalg.norm(s) for s in contacts])
    return CollisionCone(False, normals=normals, edges=edges, contacts=contacts)


def sign_change(arr):
    return np.where(np.sign(arr[:-1]) != np.sign(arr[1:]))[0] + 1


def collision_cone(O, n_grid=1440):
    """Collision cone for convex obstacle O, via the two tangents from the origin.

    Returns a CollisionCone.  If the origin is inside O, contains_origin is True
    and the geometric fields are None.
    """
    g_of = lambda p: O.support(_dir(p))

    phi = np.linspace(0, 2 * np.pi, n_grid)
    g = np.array([g_of(p) for p in phi])
    idx = sign_change(g)
    if len(idx) == 0:
        return CollisionCone(contains_origin=True)

    roots = [brentq(g_of, phi[i - 1], phi[i]) for i in idx] #refines roots using brentq
    normals = np.array([_dir(r) for r in roots])
    return _cone_from_normals(O, normals)
