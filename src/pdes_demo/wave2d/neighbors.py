"""Nearest-neighbour search on the doubly periodic unit square.

The MATLAB (``findTiledNearestNeighbors.m``) ghosted every node within two
stencil radii of an edge to the opposite side and ran ``knnsearch`` on the
augmented set. ``scipy.spatial.cKDTree`` handles the torus directly through
its ``boxsize`` argument, so the port is a thin wrapper plus a minimal-image
rule for turning neighbour coordinates into displacements from the centre.
"""

import numpy as np
from scipy.spatial import cKDTree

BOX = 1.0


def wrap(xy: np.ndarray) -> np.ndarray:
    """Map coordinates onto [0, 1) componentwise.

    ``np.mod`` of a negative number smaller than the ULP at 1 rounds up to
    exactly 1.0, which ``cKDTree(boxsize=...)`` rejects; clamp it back.
    """
    return np.minimum(np.mod(xy, BOX), np.nextafter(BOX, 0.0))


def minimal_image(disp: np.ndarray) -> np.ndarray:
    """Wrap displacements into [-1/2, 1/2] so they measure the short way round.

    Exact half-period ties are left where they are; they cannot occur for
    nodes in general position.
    """
    return disp - BOX * np.round(disp / BOX)


def periodic_knn(
    xy: np.ndarray, k: int, query: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Indices and distances of the ``k`` nearest nodes under periodic wrap.

    With ``query`` unset every node is queried against the set it belongs to,
    so column 0 is the node itself at distance 0. Returns ``(idx, dist)`` of
    shape ``(n_query, k)`` each.
    """
    xy = wrap(np.asarray(xy, dtype=float))
    if k > xy.shape[0]:
        raise ValueError(f"k={k} exceeds the number of nodes ({xy.shape[0]})")
    tree = cKDTree(xy, boxsize=[BOX, BOX])
    pts = xy if query is None else wrap(np.asarray(query, dtype=float))
    dist, idx = tree.query(pts, k=k)
    return idx, dist


def stencil_offsets(
    xy: np.ndarray, idx: np.ndarray, centers: np.ndarray | None = None
) -> np.ndarray:
    """Displacements ``xy[idx] - centre`` for every stencil, shape ``(s, k, 2)``.

    ``centers`` defaults to the nodes themselves (``xy[idx[:, 0]]``), which
    is the "evaluation node is its own nearest neighbour" convention used by
    the operator construction.
    """
    xy = np.asarray(xy, dtype=float)
    if centers is None:
        centers = xy[idx[:, 0]]
    return minimal_image(xy[idx] - centers[:, None, :])
