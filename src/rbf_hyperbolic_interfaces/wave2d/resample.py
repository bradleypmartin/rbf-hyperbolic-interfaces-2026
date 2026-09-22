"""Interpolation from a node set to other points, one material at a time.

Both 2-D drivers need field values away from the nodes they were computed
on: on a pixel grid to draw colour maps, and on a coarse node set to measure
a coarse run against a fine reference. The interpolant is the local
Gaussian-plus-polynomial fit behind the RBF-FD weights
(:mod:`rbf_hyperbolic_interfaces.wave2d.rbf`), evaluated at the target point instead of
differentiated at a node, so it is exact for the same polynomials as the
solver and converges at the same order.

No stencil crosses an interface. The fields are smooth on each side of an
interface but not across it: v has a kink there and f jumps (dissertation
§3.3). A stencil mixing both sides would add a low-order error of its own
in exactly the band where the naive and interface-aware solvers differ, and
the cure would be to give the interpolant the interface-aware treatment as
well. Restricting every stencil to source nodes in the target's own
material sidesteps that; the stencils near an interface are merely
lopsided, which costs nothing in order.
"""

import numpy as np
import scipy.sparse as sp

from .domain import LayeredMedium2D, NodeSet
from .neighbors import periodic_knn, stencil_offsets, wrap
from .rbf import rbf_interpolation_weights


def pixel_grid(m: int) -> np.ndarray:
    """Centres of an ``m x m`` pixel grid on the unit square, shape ``(m*m, 2)``.

    Row-major in ``y`` then ``x``, so values on it reshape to ``(m, m)`` and
    draw with ``imshow(origin="lower", extent=(0, 1, 0, 1))``.
    """
    g = (np.arange(m) + 0.5) / m
    gx, gy = np.meshgrid(g, g, indexing="xy")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1)


def _material_groups(
    src: np.ndarray, tgt: np.ndarray, medium: LayeredMedium2D
) -> list[tuple[np.ndarray, np.ndarray]]:
    in_src = medium.in_layer(src[:, 0], src[:, 1])
    in_tgt = medium.in_layer(tgt[:, 0], tgt[:, 1])
    return [
        (np.flatnonzero(in_src == inside), np.flatnonzero(in_tgt == inside))
        for inside in (False, True)
    ]


def resample_matrix(
    source: NodeSet | np.ndarray,
    target: np.ndarray,
    medium: LayeredMedium2D,
    *,
    stencil_size: int = 30,
    poly_degree: int = 4,
    shape: float = 0.4,
    one_sided: bool = True,
) -> sp.csr_array:
    """Sparse ``(n_target, n_source)`` matrix taking nodal values to ``target``.

    Each target point gets the ``stencil_size`` nearest source nodes (under
    periodic wrap) from its own material and the interpolation weights of
    :func:`rbf_hyperbolic_interfaces.wave2d.rbf.rbf_interpolation_weights`.
    ``one_sided=False`` lets stencils cross the interfaces; it exists to demonstrate why
    the default does not.
    """
    src = wrap(source.xy if isinstance(source, NodeSet) else np.asarray(source))
    tgt = wrap(np.asarray(target, dtype=float))
    if tgt.ndim != 2 or tgt.shape[1] != 2:
        raise ValueError("target must have shape (n_target, 2)")
    if one_sided:
        groups = _material_groups(src, tgt, medium)
    else:
        groups = [(np.arange(len(src)), np.arange(len(tgt)))]
    rows, cols, vals = [], [], []
    for s_idx, t_idx in groups:
        if t_idx.size == 0:
            continue
        if s_idx.size < stencil_size:
            raise ValueError("too few source nodes in one material for a stencil")
        idx, _ = periodic_knn(src[s_idx], stencil_size, query=tgt[t_idx])
        offsets = stencil_offsets(src[s_idx], idx, centers=tgt[t_idx])
        w = rbf_interpolation_weights(offsets, shape=shape, poly_degree=poly_degree)
        rows.append(np.repeat(t_idx, stencil_size))
        cols.append(s_idx[idx].ravel())
        vals.append(w.ravel())
    return sp.csr_array(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(tgt), len(src)),
    )
