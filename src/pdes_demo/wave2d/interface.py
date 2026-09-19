"""Interface-aware RBF-FD stencils (dissertation §3.3; JCP 2017 preprint).

A stencil near a material interface is treated in a local frame: origin at
the nearest interface point, x' along the tangent, y' along the normal. The
isotropic elastic system keeps its form under rotation, so in that frame
the interface is (locally) ``y' = 0`` and the rotated fields
``(u', v', f', g', h')`` obey eq. 32 unchanged. Plain polynomials are then
replaced, on stencils that see both sides, by piecewise polynomials that
satisfy the interface physics:

1. Expand ``(u', v')`` on each side in monomials of ``(x', y')`` up to
   degree ``p + 1``. Velocity ``(u', v')`` and traction ``(g', h')`` are
   continuous at ``y' = 0`` for all time, and their time derivatives follow
   from powers of the PDE operator ``D`` (eq. 35): even powers map ``(u, v)``
   coefficients to ``(u, v)`` rates, odd powers to ``(f, g, h)`` rates.
   Matching the coefficients of ``1, x', ..., x'^(p+1-k)`` at level ``k``
   gives a square continuity matrix ``C`` per side (eq. 41).
2. Keep the standard monomial basis on one side and translate it across
   with ``C_other^-1 C_standard`` (eq. 42). Truncate to degree ``p``.
3. The stress fields have no complete set of conditions (``f'`` may jump),
   so their basis is generated: apply the ``(f, g, h)`` rows of ``D`` on each
   side to every velocity basis function, drop the three columns that come
   from rigid motions (constants in u and v, and one of the two shear
   terms that give the same result), and truncate to degree ``p``
   (§3.3.2). For ``p = 3`` this is the 27-dimensional space the
   dissertation counts.
4. Solve the usual saddle-point systems, now coupled across fields:
   ``(f, g, h)`` rates need weights on ``u`` and ``v`` data jointly
   (``2n`` unknowns), ``(u, v)`` rates need weights on ``f, g, h`` jointly
   (``3n``). Hyperviscosity uses the same piecewise bases (dissertation
   p. 41). Weights come out in the rotated frame and are rotated back with
   ``R_uv`` and ``R_fgh`` (eq. 37-38), as ``EWE2DRbfPrep.m`` does. This
   step (:func:`gaussian_rows`, :func:`coupled_weights`) takes any
   augmenting basis; the seed stencils of :mod:`.seeds` reuse it.

With constant coefficients and the locally flat interface (the MATLAB's
choice; curvature terms are a possible refinement, JCP Fig. 10) the bases
depend only on the two materials, so they are built once per interface.
"""

from dataclasses import dataclass

import numpy as np

from .domain import ElasticMaterial, SineInterface
from .rbf import (
    laplacian_power_of_gaussian,
    monomial_exponents,
)

FIELD_ROWS_UV = ("f", "g", "h")  # rates computed from (u, v) data
FIELD_ROWS_FGH = ("u", "v")  # rates computed from (f, g, h) data


# --- polynomial algebra ------------------------------------------------------------


def derivative_matrices(exps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``Dx``, ``Dy`` acting on coefficient vectors over the monomials ``exps``.

    ``(Dx c)`` holds the coefficients of ``d/dx`` of the polynomial with
    coefficients ``c``; derivatives of the top degree land inside the set.
    """
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(exps)}
    m = len(exps)
    dx = np.zeros((m, m))
    dy = np.zeros((m, m))
    for j, (a, b) in enumerate(exps):
        if a > 0:
            dx[index[(a - 1, b)], j] = a
        if b > 0:
            dy[index[(a, b - 1)], j] = b
    return dx, dy


def pde_operator(material: ElasticMaterial, exps: np.ndarray) -> np.ndarray:
    """The ``5m x 5m`` block operator ``D`` of eq. 32/35 on stacked coefficients.

    Field order ``(u, v, f, g, h)``; one application gives the coefficients
    of the time derivative of each field from those of the others.
    """
    dx, dy = derivative_matrices(exps)
    m = len(exps)
    lam, mu, rho = material.lam, material.mu, material.rho
    z = np.zeros((m, m))
    return np.block(
        [
            [z, z, dx / rho, dy / rho, z],
            [z, z, z, dx / rho, dy / rho],
            [(lam + 2 * mu) * dx, lam * dy, z, z, z],
            [mu * dy, mu * dx, z, z, z],
            [lam * dx, (lam + 2 * mu) * dy, z, z, z],
        ]
    )


def continuity_matrix(material: ElasticMaterial, exps: np.ndarray) -> np.ndarray:
    """Rows expressing velocity/traction continuity at ``y = 0`` (eq. 40-41).

    ``exps`` are the monomials up to the expansion degree ``q``. For each
    power ``k = 0..q`` of ``D``, the coefficients of ``x^j`` (``j <= q - k``)
    in ``(u, v)`` (even ``k``) or ``(g, h)`` (odd ``k``) are collected; the
    result is a square ``2m x 2m`` matrix acting on stacked ``(u, v)``
    coefficients such that ``C_side_a c_a = C_side_b c_b``.
    """
    m = len(exps)
    q = int(exps[:, 0].max())
    d = pde_operator(material, exps)
    idx_x = {int(a): i for i, (a, b) in enumerate(exps) if b == 0}
    rows = []
    dk = np.eye(5 * m)
    for k in range(q + 1):
        if k % 2 == 0:
            blk = dk[0 : 2 * m, 0 : 2 * m]
        else:
            blk = dk[3 * m : 5 * m, 0 : 2 * m]
        for j in range(q - k + 1):
            r = idx_x[j]
            rows.append(blk[r])
            rows.append(blk[m + r])
        dk = dk @ d
    c = np.array(rows)
    if c.shape != (2 * m, 2 * m):
        raise AssertionError("continuity matrix is not square")
    return c


@dataclass(frozen=True)
class InterfaceBasis:
    """Piecewise polynomial bases across one flat interface, in its own frame.

    ``uv[side]`` has shape ``(2, m, n_uv)``: coefficient of monomial
    ``exps[i]`` in the u (0) or v (1) component of basis function ``j`` on
    that side (``side`` is ``True`` above the interface). ``fgh[side]`` has
    shape ``(3, m, n_fgh)`` for the stress fields.
    """

    degree: int
    exps: np.ndarray
    uv: dict[bool, np.ndarray]
    fgh: dict[bool, np.ndarray]
    below: ElasticMaterial
    above: ElasticMaterial

    @property
    def n_uv(self) -> int:
        return self.uv[True].shape[2]

    @property
    def n_fgh(self) -> int:
        return self.fgh[True].shape[2]

    def material(self, above: bool) -> ElasticMaterial:
        return self.above if above else self.below


def interface_basis(
    below: ElasticMaterial,
    above: ElasticMaterial,
    degree: int,
    standard_above: bool = True,
) -> InterfaceBasis:
    """Build the coupled ``(u, v)`` and ``(f, g, h)`` bases across an interface.

    The standard monomial basis lives on the ``standard_above`` side (the
    MATLAB keeps it inside the layer); the other side is derived from the
    continuity matrices. Expansions are carried to ``degree + 1`` so the
    generated stress basis is exact to ``degree``.
    """
    exps1 = monomial_exponents(degree + 1)
    exps = monomial_exponents(degree)
    m1, m = len(exps1), len(exps)
    mats = {False: below, True: above}
    c = {side: continuity_matrix(mats[side], exps1) for side in (False, True)}
    d = {side: pde_operator(mats[side], exps1) for side in (False, True)}
    std, der = standard_above, not standard_above
    u_full = {std: np.eye(2 * m1), der: np.linalg.solve(c[der], c[std])}

    keep_uv = list(range(m)) + list(range(m1, m1 + m))
    # Rigid motions generate no stress: constants in u and v, and the two
    # shear terms u = y, v = x give identical (f, g, h) rates. Drop u_1, v_1
    # and v_x (MATLAB's choice).
    keep_fgh = [j for j in range(2 * m1) if j not in (0, m1, m1 + 1)]
    uv: dict[bool, np.ndarray] = {}
    fgh: dict[bool, np.ndarray] = {}
    for side in (False, True):
        u_side = u_full[side][:, keep_uv]
        uv[side] = np.stack([u_side[0:m], u_side[m1 : m1 + m]])
        f_side = (d[side][2 * m1 : 5 * m1, 0 : 2 * m1] @ u_full[side])[:, keep_fgh]
        fgh[side] = np.stack(
            [f_side[0:m], f_side[m1 : m1 + m], f_side[2 * m1 : 2 * m1 + m]]
        )
    return InterfaceBasis(
        degree=degree, exps=exps, uv=uv, fgh=fgh, below=below, above=above
    )


def evaluate_basis(
    coeffs: np.ndarray, exps: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Values of basis functions at ``points`` ``(..., 2)``: ``(..., n_basis)``.

    ``coeffs`` is ``(m, n_basis)`` over the monomials ``exps``.
    """
    mono = points[..., 0:1] ** exps[:, 0] * points[..., 1:2] ** exps[:, 1]
    return mono @ coeffs


# --- local frames ------------------------------------------------------------------


def rotation_matrices(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``R_uv`` (2x2) and ``R_fgh`` (3x3) of eq. 37-38, batched over ``theta``."""
    c, s = np.cos(theta), np.sin(theta)
    r_uv = np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)
    r_fgh = np.stack(
        [
            np.stack([c**2, 2 * s * c, s**2], -1),
            np.stack([-s * c, c**2 - s**2, s * c], -1),
            np.stack([s**2, -2 * s * c, c**2], -1),
        ],
        -2,
    )
    return r_uv, r_fgh


def closest_point(
    interface: SineInterface, x: np.ndarray, y: np.ndarray, iterations: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Foot point ``x0`` on the curve nearest ``(x, y)`` and the tangent angle there.

    Same fixed-point scheme as ``pointFinder1/2``: project onto the tangent
    line at the current guess and move the guess to the projection.
    Converges in a few steps for the mild curves used here; exact
    immediately for a flat interface.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x0 = x.copy()
    for _ in range(iterations):
        y0 = interface.height(x0)
        th = interface.angle(x0)
        # Distance along the tangent from the current foot point.
        t = np.cos(th) * (x - x0) + np.sin(th) * (y - y0)
        x0 = x0 + t * np.cos(th)
    return x0, interface.angle(x0)


# --- stencil weights ---------------------------------------------------------------


@dataclass(frozen=True)
class InterfaceWeights:
    """Lab-frame weights for interface stencils, one entry per stencil.

    ``fgh_from_uv[s, r, c, i]``: contribution of field ``c`` (u, v) at stencil
    node ``i`` to the rate of stress field ``r`` (f, g, h). ``uv_from_fgh``
    likewise ``(s, 2, 3, n)``. ``hyper_uv`` ``(s, 2, 2, n)`` and ``hyper_fgh``
    ``(s, 3, 3, n)`` hold ``Delta^k`` weights (raw sign).
    """

    fgh_from_uv: np.ndarray
    uv_from_fgh: np.ndarray
    hyper_uv: np.ndarray
    hyper_fgh: np.ndarray


@dataclass(frozen=True)
class GaussianRows:
    """The Gaussian part of a coupled stencil, relative to the evaluation node.

    ``a`` ``(s, n, n)`` is the interpolation block; ``dx``, ``dy`` and
    ``lap`` ``(s, n)`` are ``d/dx``, ``d/dy`` and ``Delta^k`` of every
    node's Gaussian at the evaluation node; ``r_max`` ``(s,)`` is the
    distance to the farthest node, which scales the augmenting basis.
    """

    a: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    lap: np.ndarray
    r_max: np.ndarray


@dataclass(frozen=True)
class Augmentation:
    """The augmenting basis of a coupled stencil, in the stencil coordinate
    ``(x' - x_e, y' - y_e) / r_max``.

    ``uv`` ``(s, 2n, n_uv)``: u then v values of every velocity basis
    function at every node; ``fgh`` ``(s, 3n, n_fgh)`` likewise for the
    stress basis. ``uv_jet`` ``(s, n_uv, 4)`` holds ``(u_X, u_Y, v_X, v_Y)``
    and ``fgh_jet`` ``(s, n_fgh, 4)`` holds ``(f_X, g_X, g_Y, h_Y)`` at the
    evaluation node; ``lap_uv`` ``(s, n_uv, 2)`` and ``lap_fgh``
    ``(s, n_fgh, 3)`` hold ``Delta^k`` of each component there. All
    derivatives are per unit of the stencil coordinate.
    """

    uv: np.ndarray
    fgh: np.ndarray
    uv_jet: np.ndarray
    fgh_jet: np.ndarray
    lap_uv: np.ndarray
    lap_fgh: np.ndarray


def _laplacian_power(exps: np.ndarray, k: int) -> np.ndarray:
    dx, dy = derivative_matrices(exps)
    lap = dx @ dx + dy @ dy
    return np.linalg.matrix_power(lap, k)


def gaussian_rows(
    local: np.ndarray,
    *,
    shape: float = 0.4,
    shape_neighbor: int = 3,
    hyper_power: int = 3,
) -> GaussianRows:
    """Gaussian block and right-hand sides for stencils ``local`` ``(s, n, 2)``,
    node 0 the evaluation node; rotation invariant, so any frame will do."""
    local = np.asarray(local, dtype=float)
    k = hyper_power
    off = local - local[:, 0:1, :]
    r = np.linalg.norm(off, axis=2)
    r_sorted = np.sort(r, axis=1)
    d_shape = r_sorted[:, shape_neighbor]
    r_max = r_sorted[:, -1]
    if np.any(d_shape <= 0) or np.any(r_max <= 0):
        raise ValueError("degenerate interface stencil")
    eps2 = (shape / d_shape) ** 2
    diff = off[:, :, None, :] - off[:, None, :, :]
    a = np.exp(-eps2[:, None, None] * np.sum(diff**2, axis=3))
    phi = np.exp(-eps2[:, None] * r**2)
    rhs_dx = 2 * eps2[:, None] * off[:, :, 0] * phi
    rhs_dy = 2 * eps2[:, None] * off[:, :, 1] * phi
    rhs_lap = (
        eps2[:, None] ** k * laplacian_power_of_gaussian(k, eps2[:, None] * r**2) * phi
    )
    return GaussianRows(a=a, dx=rhs_dx, dy=rhs_dy, lap=rhs_lap, r_max=r_max)


def interface_weights(
    local: np.ndarray,
    above: np.ndarray,
    theta: np.ndarray,
    basis: InterfaceBasis,
    *,
    shape: float = 0.4,
    shape_neighbor: int = 3,
    hyper_power: int = 3,
) -> InterfaceWeights:
    """Coupled RBF-FD weights for stencils crossing one interface.

    ``local`` ``(s, n, 2)``: stencil node positions in the rotated frame,
    relative to the interface foot point; node 0 is the evaluation node.
    ``above`` ``(s, n)``: which side each node is on. ``theta`` ``(s,)``:
    frame angle, used only to rotate the finished weights back.
    """
    local = np.asarray(local, dtype=float)
    exps = basis.exps
    k = hyper_power
    dxm, dym = derivative_matrices(exps)
    lap_k = _laplacian_power(exps, k)
    gauss = gaussian_rows(
        local, shape=shape, shape_neighbor=shape_neighbor, hyper_power=k
    )

    # Polynomial part in coordinates scaled by r_max about the interface point.
    pts = local / gauss.r_max[:, None, None]
    mono = pts[..., 0:1] ** exps[:, 0] * pts[..., 1:2] ** exps[:, 1]  # (s, n, m)
    sel = above[..., None]  # (s, n, 1)

    def eval_side(coeffs: dict[bool, np.ndarray], comp: int) -> np.ndarray:
        """Values of component ``comp`` of every basis function at every node."""
        lo = mono @ coeffs[False][comp]
        hi = mono @ coeffs[True][comp]
        return np.where(sel, hi, lo)  # (s, n, n_basis)

    p_uv = np.concatenate([eval_side(basis.uv, 0), eval_side(basis.uv, 1)], axis=1)
    p_fgh = np.concatenate([eval_side(basis.fgh, c) for c in range(3)], axis=1)

    # Derivatives of the basis at the evaluation node, in its own material.
    mono_e = mono[:, 0, :]  # (s, m)
    above_e = above[:, 0]
    lam = np.where(above_e, basis.above.lam, basis.below.lam)
    mu = np.where(above_e, basis.above.mu, basis.below.mu)
    rho = np.where(above_e, basis.above.rho, basis.below.rho)

    def deriv_e(
        coeffs: dict[bool, np.ndarray], comp: int, op: np.ndarray
    ) -> np.ndarray:
        lo = mono_e @ (op @ coeffs[False][comp])
        hi = mono_e @ (op @ coeffs[True][comp])
        return np.where(above_e[:, None], hi, lo)  # (s, n_basis)

    aug = Augmentation(
        uv=p_uv,
        fgh=p_fgh,
        uv_jet=np.stack(
            [
                deriv_e(basis.uv, 0, dxm),
                deriv_e(basis.uv, 0, dym),
                deriv_e(basis.uv, 1, dxm),
                deriv_e(basis.uv, 1, dym),
            ],
            axis=-1,
        ),
        fgh_jet=np.stack(
            [
                deriv_e(basis.fgh, 0, dxm),
                deriv_e(basis.fgh, 1, dxm),
                deriv_e(basis.fgh, 1, dym),
                deriv_e(basis.fgh, 2, dym),
            ],
            axis=-1,
        ),
        lap_uv=np.stack([deriv_e(basis.uv, c, lap_k) for c in range(2)], axis=-1),
        lap_fgh=np.stack([deriv_e(basis.fgh, c, lap_k) for c in range(3)], axis=-1),
    )
    return coupled_weights(gauss, aug, (lam, mu, rho), theta, hyper_power=k)


def coupled_weights(
    gauss: GaussianRows,
    aug: Augmentation,
    material: tuple[np.ndarray, np.ndarray, np.ndarray],
    theta: np.ndarray,
    *,
    hyper_power: int,
) -> InterfaceWeights:
    """The coupled saddle-point solves of step 4, for any augmenting basis.

    ``material`` is ``(lam, mu, rho)`` at each evaluation node, ``(s,)``
    each. The five right-hand sides per block are the elastic rates of
    eq. 32 at the evaluation node (``f_t, g_t, h_t`` from ``(u, v)`` data,
    ``u_t, v_t`` from ``(f, g, h)`` data) and ``Delta^k`` of each field.
    """
    s_total, n, _ = gauss.a.shape
    k = hyper_power
    lam, mu, rho = material
    lam2mu = (lam + 2 * mu)[:, None]
    lam_c, mu_c, rho_c = lam[:, None], mu[:, None], rho[:, None]
    sc = (1.0 / gauss.r_max)[:, None]

    ux, uy, vx, vy = (aug.uv_jet[..., i] for i in range(4))
    fx, gx, gy, hy = (aug.fgh_jet[..., i] for i in range(4))
    poly_uv = np.stack(
        [
            (lam2mu * ux + lam_c * vy) * sc,  # f_t
            mu_c * (uy + vx) * sc,  # g_t
            (lam_c * ux + lam2mu * vy) * sc,  # h_t
            aug.lap_uv[..., 0] * sc ** (2 * k),  # Delta^k u
            aug.lap_uv[..., 1] * sc ** (2 * k),  # Delta^k v
        ],
        axis=-1,
    )  # (s, n_uv, 5)
    poly_fgh = np.stack(
        [
            (fx + gy) / rho_c * sc,  # u_t
            (gx + hy) / rho_c * sc,  # v_t
            aug.lap_fgh[..., 0] * sc ** (2 * k),
            aug.lap_fgh[..., 1] * sc ** (2 * k),
            aug.lap_fgh[..., 2] * sc ** (2 * k),
        ],
        axis=-1,
    )  # (s, n_fgh, 5)

    rhs_dx, rhs_dy, rhs_lap = gauss.dx, gauss.dy, gauss.lap
    zero = np.zeros_like(rhs_dx)
    rbf_uv = np.stack(
        [
            np.concatenate([lam2mu * rhs_dx, lam_c * rhs_dy], axis=1),
            np.concatenate([mu_c * rhs_dy, mu_c * rhs_dx], axis=1),
            np.concatenate([lam_c * rhs_dx, lam2mu * rhs_dy], axis=1),
            np.concatenate([rhs_lap, zero], axis=1),
            np.concatenate([zero, rhs_lap], axis=1),
        ],
        axis=-1,
    )  # (s, 2n, 5)
    rbf_fgh = np.stack(
        [
            np.concatenate([rhs_dx / rho_c, rhs_dy / rho_c, zero], axis=1),
            np.concatenate([zero, rhs_dx / rho_c, rhs_dy / rho_c], axis=1),
            np.concatenate([rhs_lap, zero, zero], axis=1),
            np.concatenate([zero, rhs_lap, zero], axis=1),
            np.concatenate([zero, zero, rhs_lap], axis=1),
        ],
        axis=-1,
    )  # (s, 3n, 5)

    w_uv = _solve_coupled(gauss.a, 2, aug.uv, rbf_uv, poly_uv)  # (s, 2n, 5)
    w_fgh = _solve_coupled(gauss.a, 3, aug.fgh, rbf_fgh, poly_fgh)  # (s, 3n, 5)

    # Reshape to (s, rows, fields, n) in the rotated frame.
    fgh_from_uv = w_uv[:, :, 0:3].reshape(s_total, 2, n, 3).transpose(0, 3, 1, 2)
    hyper_uv = w_uv[:, :, 3:5].reshape(s_total, 2, n, 2).transpose(0, 3, 1, 2)
    uv_from_fgh = w_fgh[:, :, 0:2].reshape(s_total, 3, n, 2).transpose(0, 3, 1, 2)
    hyper_fgh = w_fgh[:, :, 2:5].reshape(s_total, 3, n, 3).transpose(0, 3, 1, 2)

    # Back to lab-frame components: W = R_rows^-1 W' R_cols per node.
    r_uv, r_fgh = rotation_matrices(np.asarray(theta, dtype=float))
    r_uv_inv = np.linalg.inv(r_uv)
    r_fgh_inv = np.linalg.inv(r_fgh)

    def rotate(w: np.ndarray, rows_inv: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return np.einsum("sab,sbci,scd->sadi", rows_inv, w, cols)

    return InterfaceWeights(
        fgh_from_uv=rotate(fgh_from_uv, r_fgh_inv, r_uv),
        uv_from_fgh=rotate(uv_from_fgh, r_uv_inv, r_fgh),
        hyper_uv=rotate(hyper_uv, r_uv_inv, r_uv),
        hyper_fgh=rotate(hyper_fgh, r_fgh_inv, r_fgh),
    )


def _solve_coupled(
    a: np.ndarray,
    n_fields: int,
    p: np.ndarray,
    rbf_rhs: np.ndarray,
    poly_rhs: np.ndarray,
) -> np.ndarray:
    """Saddle-point solve with ``blockdiag(A, ..., A)`` and a coupled ``P``."""
    s, n, _ = a.shape
    n_poly = p.shape[2]
    size = n_fields * n + n_poly
    lhs = np.zeros((s, size, size))
    for f in range(n_fields):
        lhs[:, f * n : (f + 1) * n, f * n : (f + 1) * n] = a
    lhs[:, : n_fields * n, n_fields * n :] = p
    lhs[:, n_fields * n :, : n_fields * n] = np.transpose(p, (0, 2, 1))
    rhs = np.concatenate([rbf_rhs, poly_rhs], axis=1)
    sol = np.linalg.solve(lhs, rhs)
    return sol[:, : n_fields * n, :]
