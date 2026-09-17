"""RBF-FD stencil weights: Gaussians plus polynomials, batched over stencils.

For a stencil with nodes x_1..x_n around an evaluation point x_c, the
weights w for a linear operator L solve the saddle-point system
(dissertation eq. 33; Geophysics 2015 eq. 5-6)

    [ A   P ] [ w  ]   [ L phi(|x - x_i|) |_{x_c} ]
    [ P^T 0 ] [ w* ] = [ L p_m(x)          |_{x_c} ]

with ``A_ij = phi(|x_i - x_j|)``, ``P_im = p_m(x_i)`` over all bivariate
monomials up to a chosen degree, and ``w*`` discarded. The constraint rows
make the weights exact for every polynomial in the augmentation, and the
polynomials "take over" from the RBFs under refinement.

Conventions follow ``createRBFLoperator1`` in ``EWE2DRbfPrep.m``:

- ``phi(r) = exp(-(eps r)^2)`` with ``eps = shape / d``, ``d`` the distance
  from the centre to its 3rd-nearest neighbour (``knnsearch`` column 4), so
  one relative ``shape`` (0.4) serves every stencil;
- polynomials are evaluated in ``(x - x_c) / r_max`` with ``r_max`` the
  distance to the farthest stencil node, for conditioning;
- powers of the Laplacian of a Gaussian, needed for hyperviscosity, come
  from the Laguerre recurrence of Fornberg & Lehto (2011).

Only Gaussians are implemented: that is what the MATLAB used and what Brad
knows to work here.
"""

from dataclasses import dataclass
from math import comb, factorial

import numpy as np

DIM = 2


def monomial_exponents(degree: int) -> np.ndarray:
    """Exponent pairs ``(a, b)`` of all ``x^a y^b`` with ``a + b <= degree``.

    Ordered by total degree, then descending in ``a`` (MATLAB order).
    """
    if degree < 0:
        raise ValueError("degree must be >= 0")
    return np.array(
        [(k - j, j) for k in range(degree + 1) for j in range(k + 1)], dtype=int
    )


def laplacian_power_of_gaussian(k: int, s: np.ndarray) -> np.ndarray:
    """``p_k(s)`` with ``Delta^k exp(-eps^2 r^2) = eps^(2k) p_k(s) exp(-s)``.

    ``s = (eps r)^2`` and the Laplacian is the 2-D one. Three-term recurrence
    coded in ``augHypwithpolys1``; equals ``(-4)^k k! L_k^{(0)}(s)`` with
    ``L`` the generalised Laguerre polynomial.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    s = np.asarray(s, dtype=float)
    p_prev = np.ones_like(s)
    if k == 0:
        return p_prev
    p = 4 * s - 2 * DIM
    for j in range(2, k + 1):
        p, p_prev = (
            4 * (s - 2 * (j - 1) - DIM / 2) * p
            - 8 * (j - 1) * (2 * (j - 1) - 2 + DIM) * p_prev,
            p,
        )
    return p


def laplacian_power_of_monomials(k: int, exps: np.ndarray) -> np.ndarray:
    """``Delta^k (x^a y^b)`` evaluated at the origin, for each exponent pair."""
    out = np.zeros(len(exps))
    for m, (a, b) in enumerate(exps):
        if a % 2 == 0 and b % 2 == 0 and a + b == 2 * k:
            out[m] = comb(k, a // 2) * factorial(a) * factorial(b)
    return out


@dataclass(frozen=True)
class StencilWeights:
    dx: np.ndarray  # (s, n)
    dy: np.ndarray  # (s, n)
    hyper: np.ndarray | None  # (s, n): Delta^k weights (raw sign), or None
    shape_scale: np.ndarray  # (s,) eps**2 actually used per stencil
    condition: np.ndarray | None  # (s,) 1-norm condition estimates if requested


def _saddle_point_system(
    off: np.ndarray, shape: float, shape_neighbor: int, exps: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Left-hand sides of the saddle-point systems for a chunk of stencils.

    ``off`` is ``(c, n, 2)``. Returns ``(lhs, eps2, r, r_max)``: the
    ``(c, n + m, n + m)`` matrices, the squared shape parameter per stencil,
    the node distances from the evaluation point ``(c, n)`` and the
    polynomial scaling distance ``(c,)``.
    """
    n = off.shape[1]
    m = len(exps)
    r = np.linalg.norm(off, axis=2)  # (c, n)
    r_sorted = np.sort(r, axis=1)
    d_shape = r_sorted[:, shape_neighbor]
    r_max = r_sorted[:, -1]
    if np.any(d_shape <= 0) or np.any(r_max <= 0):
        raise ValueError("degenerate stencil (coincident nodes)")
    eps2 = (shape / d_shape) ** 2  # (c,)

    diff = off[:, :, None, :] - off[:, None, :, :]  # (c, n, n, 2)
    r2 = np.sum(diff**2, axis=3)
    r2_off = r2 + np.diag(np.full(n, np.inf))
    if np.any(r2_off.min(axis=(1, 2)) <= 0):
        raise ValueError("degenerate stencil (coincident nodes)")
    a = np.exp(-eps2[:, None, None] * r2)
    xt = off / r_max[:, None, None]  # scaled coordinates (c, n, 2)
    p = xt[:, :, 0:1] ** exps[:, 0] * xt[:, :, 1:2] ** exps[:, 1]  # (c, n, m)

    c = off.shape[0]
    lhs = np.zeros((c, n + m, n + m))
    lhs[:, :n, :n] = a
    lhs[:, :n, n:] = p
    lhs[:, n:, :n] = np.transpose(p, (0, 2, 1))
    return lhs, eps2, r, r_max


def rbf_fd_weights(
    offsets: np.ndarray,
    *,
    shape: float = 0.4,
    shape_neighbor: int = 3,
    poly_degree: int = 4,
    hyper_power: int | None = 3,
    with_condition: bool = False,
    chunk: int = 2048,
) -> StencilWeights:
    """Weights for ``d/dx``, ``d/dy`` and optionally ``Delta^k`` on every stencil.

    ``offsets`` has shape ``(s, n, 2)``: displacements of the ``n`` stencil
    nodes from the evaluation point of each of the ``s`` stencils (the
    centre node itself may be one of them, at offset 0). Nodes need not be
    sorted by distance. ``shape_neighbor`` indexes the sorted distances that
    set ``eps``; the default 3 is the 3rd-nearest neighbour when the centre
    node is in the stencil at distance 0.
    """
    offsets = np.asarray(offsets, dtype=float)
    if offsets.ndim != 3 or offsets.shape[2] != 2:
        raise ValueError("offsets must have shape (s, n, 2)")
    s_total, n, _ = offsets.shape
    exps = monomial_exponents(poly_degree)
    m = len(exps)
    if n < m:
        raise ValueError(f"{n} nodes cannot support {m} polynomial terms")
    if shape_neighbor >= n:
        raise ValueError("shape_neighbor must index a stencil node")

    dx = np.empty((s_total, n))
    dy = np.empty((s_total, n))
    hyper = np.empty((s_total, n)) if hyper_power is not None else None
    eps2_all = np.empty(s_total)
    cond = np.empty(s_total) if with_condition else None

    for start in range(0, s_total, chunk):
        sl = slice(start, min(start + chunk, s_total))
        off = offsets[sl]
        lhs, eps2, r, r_max = _saddle_point_system(off, shape, shape_neighbor, exps)
        eps2_all[sl] = eps2
        c = off.shape[0]

        phi = np.exp(-eps2[:, None] * r**2)  # (c, n), phi(|x_c - x_i|)
        n_rhs = 3 if hyper_power is not None else 2
        rhs = np.zeros((c, n + m, n_rhs))
        # d/dx of phi(|x - x_i|) at x_c is -2 eps^2 (x_c - x_i) phi = 2 eps^2 off phi.
        rhs[:, :n, 0] = 2 * eps2[:, None] * off[:, :, 0] * phi
        rhs[:, :n, 1] = 2 * eps2[:, None] * off[:, :, 1] * phi
        # Polynomial rows: derivatives of the scaled monomials at the origin.
        is_x = (exps[:, 0] == 1) & (exps[:, 1] == 0)
        is_y = (exps[:, 0] == 0) & (exps[:, 1] == 1)
        rhs[:, n:, 0] = is_x / r_max[:, None]
        rhs[:, n:, 1] = is_y / r_max[:, None]
        if hyper_power is not None:
            k = hyper_power
            lag = laplacian_power_of_gaussian(k, eps2[:, None] * r**2)
            rhs[:, :n, 2] = eps2[:, None] ** k * lag * phi
            rhs[:, n:, 2] = laplacian_power_of_monomials(k, exps) / r_max[:, None] ** (
                2 * k
            )

        sol = np.linalg.solve(lhs, rhs)
        dx[sl] = sol[:, :n, 0]
        dy[sl] = sol[:, :n, 1]
        if hyper is not None:
            hyper[sl] = sol[:, :n, 2]
        if cond is not None:
            cond[sl] = np.linalg.cond(lhs, p=1)

    return StencilWeights(
        dx=dx, dy=dy, hyper=hyper, shape_scale=eps2_all, condition=cond
    )


def rbf_interpolation_weights(
    offsets: np.ndarray,
    *,
    shape: float = 0.4,
    shape_neighbor: int = 2,
    poly_degree: int = 4,
    chunk: int = 2048,
) -> np.ndarray:
    """Weights ``(s, n)`` interpolating nodal values to each stencil's evaluation point.

    The saddle-point system of :func:`rbf_fd_weights` with the identity in
    place of the differential operator: the right-hand side is
    ``phi(|x_e - x_i|)`` over the nodes and the monomials at the evaluation
    point, ``(1, 0, ..., 0)``. The weights are exact for every polynomial in
    the augmentation and reduce to a Kronecker delta when the evaluation
    point is a stencil node. Evaluation points are generally not nodes, so
    ``shape_neighbor`` defaults to 2: ``eps = shape / d`` with ``d`` the
    distance to the third-nearest node, as for the derivative stencils.
    """
    offsets = np.asarray(offsets, dtype=float)
    if offsets.ndim != 3 or offsets.shape[2] != 2:
        raise ValueError("offsets must have shape (s, n, 2)")
    s_total, n, _ = offsets.shape
    exps = monomial_exponents(poly_degree)
    m = len(exps)
    if n < m:
        raise ValueError(f"{n} nodes cannot support {m} polynomial terms")
    if shape_neighbor >= n:
        raise ValueError("shape_neighbor must index a stencil node")

    w = np.empty((s_total, n))
    for start in range(0, s_total, chunk):
        sl = slice(start, min(start + chunk, s_total))
        off = offsets[sl]
        lhs, eps2, r, _ = _saddle_point_system(off, shape, shape_neighbor, exps)
        rhs = np.zeros((off.shape[0], n + m, 1))
        rhs[:, :n, 0] = np.exp(-eps2[:, None] * r**2)
        rhs[:, n, 0] = 1.0  # the constant monomial comes first in exps
        w[sl] = np.linalg.solve(lhs, rhs)[:, :n, 0]
    return w
