"""Sparse RBF-FD operators for the 2-D elastic wave equation.

``build_operators`` turns a node set into

- ``dx``, ``dy``: ``n x n`` first-derivative matrices,
- ``hyper``: ``n x n`` hyperviscosity matrix ``(-1)^(k+1) Delta^k`` so that
  adding ``gamma * hyper`` to the right-hand side damps (for ``k = 3`` the
  sign is ``+Delta^3``, whose Fourier symbol is ``-|xi|^6``),
- ``elastic``: the ``5n x 5n`` block operator of dissertation eq. 32 acting
  on the stacked state ``(u, v, f, g, h)``, with the material coefficients
  evaluated at each stencil's centre node (as in ``createRBFLoperator1``),
- ``hyper_block``: ``hyper`` repeated on the diagonal for all five fields.

``mode="naive"`` uses the same plain RBF-FD stencils everywhere, including
across the interfaces; that is the wrong-but-standard treatment the demo
contrasts against. ``mode="aware"`` (dissertation §3.3) is a later milestone.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp

from .domain import LayeredMedium2D, NodeSet
from .neighbors import periodic_knn, stencil_offsets
from .rbf import StencilWeights, rbf_fd_weights

Mode = Literal["naive", "aware"]


@dataclass(frozen=True)
class Operators:
    dx: sp.csr_array
    dy: sp.csr_array
    hyper: sp.csr_array
    elastic: sp.csr_array
    hyper_block: sp.csr_array
    stencils: np.ndarray  # (n, stencil_size) neighbour indices
    weights: StencilWeights
    hyper_power: int


def _scatter(nodes: NodeSet, idx: np.ndarray, w: np.ndarray) -> sp.csr_array:
    rows = np.repeat(np.arange(nodes.n), idx.shape[1])
    return sp.csr_array((w.ravel(), (rows, idx.ravel())), shape=(nodes.n, nodes.n))


def elastic_block(
    dx: sp.csr_array,
    dy: sp.csr_array,
    lam: np.ndarray,
    mu: np.ndarray,
    rho: np.ndarray,
) -> sp.csr_array:
    """Assemble the ``5n x 5n`` operator of eq. 32 from derivative matrices."""
    d = sp.diags_array
    inv_rho = d(1.0 / rho)
    lam2mu = d(lam + 2 * mu)
    lam_d, mu_d = d(lam), d(mu)
    blocks = [
        [None, None, inv_rho @ dx, inv_rho @ dy, None],
        [None, None, None, inv_rho @ dx, inv_rho @ dy],
        [lam2mu @ dx, lam_d @ dy, None, None, None],
        [mu_d @ dy, mu_d @ dx, None, None, None],
        [lam_d @ dx, lam2mu @ dy, None, None, None],
    ]
    return sp.csr_array(sp.block_array(blocks, format="csr"))


def build_operators(
    nodes: NodeSet,
    medium: LayeredMedium2D,
    *,
    stencil_size: int = 30,
    poly_degree: int = 4,
    shape: float = 0.4,
    hyper_power: int = 3,
    mode: Mode = "naive",
) -> Operators:
    if mode == "aware":
        raise NotImplementedError("interface-aware stencils are issue #8")
    if mode != "naive":
        raise ValueError(f"unknown mode {mode!r}")
    idx, _ = periodic_knn(nodes.xy, stencil_size)
    offsets = stencil_offsets(nodes.xy, idx)
    weights = rbf_fd_weights(
        offsets, shape=shape, poly_degree=poly_degree, hyper_power=hyper_power
    )
    dx = _scatter(nodes, idx, weights.dx)
    dy = _scatter(nodes, idx, weights.dy)
    hyper = _scatter(nodes, idx, (-1) ** (hyper_power + 1) * weights.hyper)
    lam = medium.lam_at(nodes.x, nodes.y)
    mu = medium.mu_at(nodes.x, nodes.y)
    rho = medium.rho_at(nodes.x, nodes.y)
    elastic = elastic_block(dx, dy, lam, mu, rho)
    hyper_block = sp.csr_array(sp.block_diag([hyper] * 5, format="csr"))
    return Operators(
        dx=dx,
        dy=dy,
        hyper=hyper,
        elastic=elastic,
        hyper_block=hyper_block,
        stencils=idx,
        weights=weights,
        hyper_power=hyper_power,
    )


def hyperviscosity_gamma(
    h: float, hyper_power: int = 3, gamma0: float = 2.4e-11, h0: float = 0.02
) -> float:
    """MATLAB scaling ``gamma = gamma0 (h/h0)^(2k-1)``.

    From ``executeRbfSimulationDriver.m``: ``2.4e-11 * relH^(2*dampLk-1)``
    with ``relH = sqrt(2500/N)``, i.e. ``h0 = 0.02``.

    ``Delta^k`` weights grow like ``h^-2k``, so this keeps ``gamma * hyper``
    proportional to ``1/h`` like the wave operator itself: the eigenvalue
    cloud scales uniformly under refinement and one CFL number keeps RK4
    stable at every resolution.
    """
    return gamma0 * (h / h0) ** (2 * hyper_power - 1)
