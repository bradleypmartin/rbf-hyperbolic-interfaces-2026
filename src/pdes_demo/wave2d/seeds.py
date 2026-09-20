"""Elastic seed bases for a straight stiff feature (issue #38; Part 3, 2-D chain).

The 2-D twin of :mod:`pdes_demo.wave1d.stiff`. A stencil whose nodes see a
material edge too steep for the node spacing keeps its Gaussian RBF part and
swaps the polynomial augmentation of :mod:`.interface` for *seeds*: functions
that look like the monomials ``x'^a y'^b`` at the evaluation node and are
continued through the edge by what the elastic operator allows. They are the
``t = 0`` profiles of solutions polynomial in time (``docs/stiff-features.md``
§1, §4).

Everything happens in the local frame of :mod:`.interface`: origin at the
closest edge-centre point, ``x'`` tangential, ``y'`` normal, rotated fields
obeying eq. 32 unchanged, and the material depending on ``y'`` only: exact
for a flat edge, and for a curved one (#42, route (a)) the material along
the normal through the foot point, which the ansatz then extends
unchanged along ``x'``, zeroth order in the curvature like the jump
stencils of :mod:`.interface`. Eliminating the stresses gives the
second-order operator
``L`` on ``(u, v)``,

    rho u_tt = d/dx [(lam + 2 mu) u_x + lam v_y] + d/dy [mu (u_y + v_x)]
    rho v_tt = d/dx [mu (u_y + v_x)] + d/dy [lam u_x + (lam + 2 mu) v_y].

With ``u = sum_j a_j(y) x^j`` and ``v = sum_j b_j(y) x^j`` the coefficient
functions obey a chain of 1-D ODEs in ``y``, triangular from the top
``x``-degree down, written in first order with the tractions as flux
variables (``tau_j = mu (a_j' + (j+1) b_{j+1})``, ``sigma_j = lam (j+1)
a_{j+1} + (lam + 2 mu) b_j'``) so that ``lam, mu, rho`` are only ever
evaluated, never differentiated:

    a_j'     = tau_j / mu - (j+1) b_{j+1}
    b_j'     = (sigma_j - lam (j+1) a_{j+1}) / (lam + 2 mu)
    tau_j'   = rho R1_j - (lam + 2 mu)(j+2)(j+1) a_{j+2} - lam (j+1) b_{j+1}'
    sigma_j' = rho R2_j - (j+1) tau_{j+1}.

``(R1, R2)`` is the chain's right-hand side: with constant coefficients ``L``
sends a monomial pair to a combination of lower-degree pairs, and the seed of
that monomial satisfies the same relation with the coefficients frozen at
the anchor and the lower *seeds* on the right, ``L S_e = sum_e' C[e', e]
S_e'`` with ``C = (D^2)[uv, uv]`` from :func:`.interface.pde_operator`.
Initial data at the anchor are the monomial's jet (value and ``y``-slope of
the matching ``a_j`` or ``b_j``, all else zero), so constant coefficients
return the monomials exactly and a jump, in the limit ``edge_width -> 0``,
returns the translated basis of :func:`.interface.interface_basis`. All
``2 m`` seeds (``m`` monomials to degree ``p + 1``) march as one linear
system; the stress seeds ``(f, g, h) = ((lam + 2 mu) u_x + lam v_y,
mu (u_y + v_x), lam u_x + (lam + 2 mu) v_y)`` come from the marched state and
the flux variables, and the three rigid-motion columns are dropped as in
``interface_basis`` (27 stress functions for ``p = 3``).

Conventions, reconciled with :func:`.interface.interface_weights` (#38): the
seeds are anchored at the *evaluation node* in both coordinates and scaled by
``r_max``, the distance to the farthest stencil node, so ``S ~ X^a Y^b`` in
``(X, Y) = (x' - x_e, y' - y_e) / r_max`` and the interpolation block is
conditioned like the polynomial one. The polynomial basis keeps its origin at
the interface point; the two spans agree (shifting the origin is a change of
basis, :func:`shift_matrix`), and anchoring the *normal* coordinate at the
evaluation node is what the 1-D work found necessary for conditioning. The
stress seeds are ``r_max`` times the physical stress rates, exactly as the
polynomial ``fgh`` basis is, so a consumer multiplies first derivatives at
the anchor by ``1 / r_max`` once.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.integrate import solve_ivp

from .domain import ElasticMaterial, LayeredMedium2D, SineInterface
from .interface import (
    Augmentation,
    InterfaceWeights,
    coupled_weights,
    gaussian_rows,
    pde_operator,
)
from .rbf import monomial_exponents

# Tail of the tanh at which the ODE march gets a fresh start (as wave1d.stiff).
_EDGE_REACH = 10.0


# --- polynomial algebra ------------------------------------------------------------


def chain_matrix(material: ElasticMaterial, exps: np.ndarray) -> np.ndarray:
    """``C`` with ``L (monomial pair e) = sum_e' C[e', e] (monomial pair e')``.

    ``L`` is the constant-coefficient operator of the module docstring on
    stacked ``(u, v)`` coefficients over ``exps``: the ``(u, v)`` block of the
    square of eq. 35's ``D``. Degree drops by two, so ``C`` is strictly
    triangular in total degree.
    """
    m = len(exps)
    d = pde_operator(material, exps)
    return (d @ d)[: 2 * m, : 2 * m]


def shift_matrix(exps: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """``T`` with ``(x - x0)^a (y - y0)^b = sum_k' T[k', k] x^a' y^b'``.

    Re-expands the monomials ``exps`` about ``(x0, y0)``; a coefficient
    matrix over ``exps`` right-multiplied by ``T`` is the same basis with the
    origin moved. Used to compare a basis anchored at the evaluation node
    with one anchored at the interface point.
    """
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(exps)}
    m = len(exps)
    t = np.zeros((m, m))
    for k, (a, b) in enumerate(exps):
        for a1 in range(a + 1):
            for b1 in range(b + 1):
                t[index[(a1, b1)], k] = (
                    comb(a, a1) * comb(b, b1) * (-x0) ** (a - a1) * (-y0) ** (b - b1)
                )
    return t


def _shift(arr: np.ndarray, s: int) -> np.ndarray:
    """``out[..., j] = arr[..., j + s]``, zero past the end (``a_{j+s}``)."""
    out = np.zeros_like(arr)
    if s < arr.shape[-1]:
        out[..., : arr.shape[-1] - s] = arr[..., s:]
    return out


# --- the material along the normal ------------------------------------------------


@dataclass(frozen=True)
class NormalProfile:
    """``lam, mu, rho`` along the normal through a foot point of an interface.

    ``y'`` is the normal coordinate with origin at the foot point
    ``(x0, y0)`` and ``theta`` the tangent angle there, exactly the frame
    of :func:`.interface.closest_point`: the point at ``y'`` is
    ``(x0 - y' sin theta, y0 + y' cos theta)``, the vertical line through
    the foot point for a flat interface (``theta = 0``, bit for bit the
    #38 profile). With the medium's blend a function of the signed normal
    distance, the material along this line is exactly the medium's own
    edge profile in ``y'`` for the interface the line is normal to, out
    to its radius of curvature; the other interface's edge is crossed
    obliquely and enters through its true distance (route (a) of #42).
    """

    medium: LayeredMedium2D
    x0: float
    y0: float
    theta: float = 0.0

    def points(self, yp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        yp = np.asarray(yp, dtype=float)
        return self.x0 - yp * np.sin(self.theta), self.y0 + yp * np.cos(self.theta)

    def material(self, yp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.medium.material_at(*self.points(yp))

    def crossings(self, interface: SineInterface) -> np.ndarray:
        """``y'`` where the normal line meets ``interface`` and its images
        ``m = -1, 0, 1``: the root of ``y0 + y' cos theta = height(x0 - y'
        sin theta) + m`` by Newton from the vertical estimate (the closed
        form for a flat interface; five steps converge the curved case to
        rounding since the line is within 8 degrees of the vertical)."""
        c, s = np.cos(self.theta), np.sin(self.theta)
        m = np.arange(-1.0, 2.0)
        if self.theta == 0.0:
            return interface.y0 - self.y0 + m
        yp = (interface.height(self.x0) + m - self.y0) / c
        for _ in range(5):
            x = self.x0 - yp * s
            yp = yp - (self.y0 + yp * c - interface.height(x) - m) / (
                c + interface.slope(x) * s
            )
        return yp

    def stops(self, lo: float, hi: float) -> np.ndarray:
        """Edge centres and their ``+-_EDGE_REACH * edge_width`` flanks inside
        ``(lo, hi)``, periodic images included, as ``y'`` values."""
        reach = _EDGE_REACH * self.medium.edge_width
        out = []
        for ifc in self.medium.interfaces:
            for centre in self.crossings(ifc):
                for s in (centre - reach, centre, centre + reach):
                    if lo < s < hi:
                        out.append(s)
        return np.array(out)


def normal_profile(
    medium: LayeredMedium2D, interface: SineInterface, x0: float
) -> NormalProfile:
    """The profile along the normal through the foot point ``(x0,
    interface.height(x0))``.

    Smooth media only: a jump is :func:`.interface.interface_basis`'s job
    (and the ``delta -> 0`` limit of these seeds).
    """
    if not medium.is_smooth:
        raise ValueError("seeds need edge_width > 0; a jump uses interface_basis")
    if interface not in medium.interfaces:
        raise ValueError("interface is not one of the medium's")
    x0 = float(x0)
    return NormalProfile(
        medium, x0, float(interface.height(x0)), float(interface.angle(x0))
    )


# --- the ODE chain -----------------------------------------------------------------


class SeedChain:
    """The first-order system for all seeds of one anchor, in stencil coordinates.

    The state is ``(a, b, tau, sigma)``, each ``(n_seeds, levels)`` with
    ``levels = degree + 2`` (``x``-degrees ``0 .. degree + 1``), flattened.
    Seed ``e = comp * m + k`` is the monomial ``exps[k]`` in component
    ``comp`` (0 = u, 1 = v). ``Y = (y' - y_e) / scale``.
    """

    def __init__(
        self,
        profile: NormalProfile,
        y_e: float,
        scale: float,
        degree: int,
        rtol: float = 1e-13,
        atol: float = 1e-15,
    ) -> None:
        if degree < 1:
            raise ValueError("degree must be at least 1")
        self.profile = profile
        self.y_e = float(y_e)
        self.scale = float(scale)
        self.degree = degree
        self.rtol, self.atol = rtol, atol
        self.exps = monomial_exponents(degree + 1)
        self.m = len(self.exps)
        self.n_seeds = 2 * self.m
        self.levels = degree + 2
        self.j = np.arange(self.levels, dtype=float)
        lam, mu, rho = (float(v) for v in profile.material(self.y_e))
        self.anchor = ElasticMaterial(lam=lam, mu=mu, rho=rho)
        self.chain_t = np.ascontiguousarray(chain_matrix(self.anchor, self.exps).T)
        self.y0 = self._initial_state()

    @property
    def shape(self) -> tuple[int, int, int]:
        return (4, self.n_seeds, self.levels)

    def _initial_state(self) -> np.ndarray:
        a, b, da, db = (np.zeros((self.n_seeds, self.levels)) for _ in range(4))
        for e in range(self.n_seeds):
            comp, k = divmod(e, self.m)
            alpha, beta = (int(v) for v in self.exps[k])
            value, slope = (a, da) if comp == 0 else (b, db)
            if beta == 0:
                value[e, alpha] = 1.0
            elif beta == 1:
                slope[e, alpha] = 1.0
        lam, mu = self.anchor.lam, self.anchor.mu
        j1 = self.j + 1
        tau = mu * (da + j1 * _shift(b, 1))
        sigma = lam * j1 * _shift(a, 1) + (lam + 2 * mu) * db
        return np.stack([a, b, tau, sigma]).ravel()

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        lam, mu, rho = self.profile.material(self.y_e + self.scale * t)
        k = lam + 2 * mu
        a, b, tau, sigma = y.reshape(self.shape)
        j1 = self.j + 1
        a1, a2 = _shift(a, 1), _shift(a, 2)
        da = tau / mu - j1 * _shift(b, 1)
        db = (sigma - lam * j1 * a1) / k
        dtau = (
            rho * (self.chain_t @ a) - k * (j1 + 1) * j1 * a2 - lam * j1 * _shift(db, 1)
        )
        dsigma = rho * (self.chain_t @ b) - j1 * _shift(tau, 1)
        return np.stack([da, db, dtau, dsigma]).ravel()

    def march(self, targets: np.ndarray) -> np.ndarray:
        """State at each target ``Y``: ``(len(targets), 4, n_seeds, levels)``.

        One DOP853 march per side from ``Y = 0`` outward, restarted at every
        target (so the values there are integrated, not interpolated) and at
        the edge centres and their flanks (:meth:`NormalProfile.stops`) so
        the adaptive step never has to discover an edge inside a long step.
        """
        targets = np.asarray(targets, dtype=float)
        out = np.empty((targets.size,) + self.shape)
        out[targets == 0] = self.y0.reshape(self.shape)
        yp = self.y_e + self.scale * targets
        stops = (self.profile.stops(yp.min(), yp.max()) - self.y_e) / self.scale
        for side in (-1, 1):
            sel = np.flatnonzero(np.sign(targets) == side)
            if sel.size == 0:
                continue
            pts = targets[sel]
            far = np.abs(pts).max()
            bounds = stops[(np.sign(stops) == side) & (np.abs(stops) < far)]
            bounds = np.unique(np.concatenate([pts, bounds]))[::side]  # away from 0
            y, t0 = self.y0, 0.0
            for t1 in bounds:
                sol = solve_ivp(
                    self.rhs,
                    (t0, t1),
                    y,
                    method="DOP853",
                    rtol=self.rtol,
                    atol=self.atol,
                )
                if not sol.success:
                    raise RuntimeError(f"seed ODE failed at Y = {t1}: {sol.message}")
                y, t0 = sol.y[:, -1], t1
                out[sel[pts == t1]] = y.reshape(self.shape)
        return out

    def evaluate(
        self, xs: np.ndarray, ys: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(u, v, f, g, h)`` of every seed at the points ``(xs, ys)`` in the
        stencil coordinate, each ``(len(xs), n_seeds)``.

        The stresses are the constitutive rows of eq. 32 in that coordinate,
        read off the marched state and the flux variables: ``g_j = tau_j``,
        ``h_j = sigma_j`` and ``f_j = (lam + 2 mu)(j+1) a_{j+1} + lam b_j'``
        with ``b_j'`` from the ``b`` equation, so nothing is differentiated
        numerically.
        """
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        state = self.march(ys)  # (n, 4, n_seeds, levels)
        a, b, tau, sigma = (state[:, i] for i in range(4))
        material = self.profile.material(self.y_e + self.scale * ys)
        lam, mu, _ = (v[:, None, None] for v in material)
        j1 = self.j + 1
        a1 = _shift(a, 1)
        db = (sigma - lam * j1 * a1) / (lam + 2 * mu)
        f_coef = (lam + 2 * mu) * j1 * a1 + lam * db
        powers = xs[:, None] ** self.j  # (n, levels)
        return tuple(
            np.einsum("nel,nl->ne", c, powers) for c in (a, b, f_coef, tau, sigma)
        )


# --- the basis at the stencil nodes ---------------------------------------------


@dataclass(frozen=True)
class SeedBasis:
    """Seed values at one stencil's nodes, in the stencil coordinate.

    ``uv`` ``(2, n, n_uv)``: u (0) and v (1) components of the velocity seeds
    (monomials to ``degree``) at every node. ``fgh`` ``(3, n, n_fgh)``: the
    stress seeds (from monomials to ``degree + 1``, rigid motions dropped).
    ``uv_jet`` ``(n_uv, 4)`` holds ``(u_X, u_Y, v_X, v_Y)`` and ``fgh_jet``
    ``(n_fgh, 4)`` holds ``(f_X, g_X, g_Y, h_Y)`` at the anchor, per unit of
    the stencil coordinate; these are what the elastic rates need there.
    ``exps`` lists the monomials to ``degree + 1``; ``keep_uv`` and
    ``keep_fgh`` index the seeds (``comp * m + k``) behind each column.
    """

    degree: int
    exps: np.ndarray
    scale: float
    anchor: ElasticMaterial
    uv: np.ndarray
    fgh: np.ndarray
    uv_jet: np.ndarray
    fgh_jet: np.ndarray
    keep_uv: np.ndarray
    keep_fgh: np.ndarray

    @property
    def n_uv(self) -> int:
        return self.uv.shape[2]

    @property
    def n_fgh(self) -> int:
        return self.fgh.shape[2]


def basis_columns(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Which of the ``2 m`` seeds (monomials to ``degree + 1``) make the
    velocity and the stress bases, in :func:`.interface.interface_basis`'s
    order: velocity keeps the monomials to ``degree``; stress keeps all but
    the constants in u and v and ``v = x`` (whose stress is that of ``u = y``).
    """
    m1 = len(monomial_exponents(degree + 1))
    m = len(monomial_exponents(degree))
    keep_uv = np.array(list(range(m)) + list(range(m1, m1 + m)))
    keep_fgh = np.array([j for j in range(2 * m1) if j not in (0, m1, m1 + 1)])
    return keep_uv, keep_fgh


def frozen_profile(profile: NormalProfile, y_e: float) -> NormalProfile:
    """The profile with the material frozen at ``y' = y_e``: a uniform medium
    of the anchor material on the same geometry, whose seeds are the
    monomials."""
    lam, mu, rho = (float(v) for v in profile.material(y_e))
    anchor = ElasticMaterial(lam=lam, mu=mu, rho=rho)
    medium = profile.medium
    uniform = LayeredMedium2D(
        background=anchor,
        layer=anchor,
        lower=medium.lower,
        upper=medium.upper,
        edge_width=medium.edge_width,
    )
    return NormalProfile(uniform, profile.x0, profile.y0, profile.theta)


def seed_basis(
    local: np.ndarray,
    profile: NormalProfile,
    degree: int,
    *,
    tangential: bool = True,
    rtol: float = 1e-13,
    atol: float = 1e-15,
) -> SeedBasis:
    """Velocity and stress seeds at the nodes of one stencil.

    ``local`` ``(n, 2)``: node positions in the interface frame (origin at
    the foot point, ``x'`` tangential), node 0 the evaluation node, as
    ``operators._local_frames`` provides. ``profile`` gives the material
    along the normal through that foot point.

    ``tangential=False`` is the ablation of issue #41: only the seeds of
    the pure ``y'^b`` monomials are marched through the edge; every
    ``x'^a y'^b`` with ``a >= 1`` is the plain monomial with the anchor
    material's stress, in both components, from a second chain on
    :func:`frozen_profile` (a frozen column must see frozen lower seeds on
    its right-hand side, or ``x'^2 y'^2`` would still be driven by the
    marched ``y'^2``). Same span dimension, same jets.
    """
    local = np.asarray(local, dtype=float)
    off = local - local[0]
    scale = float(np.max(np.linalg.norm(off, axis=1)))
    if scale <= 0:
        raise ValueError("degenerate stencil")
    xs, ys = off[:, 0] / scale, off[:, 1] / scale
    chain = SeedChain(profile, local[0, 1], scale, degree, rtol=rtol, atol=atol)
    fields = chain.evaluate(xs, ys)
    if not tangential:
        frozen = SeedChain(
            frozen_profile(profile, local[0, 1]), local[0, 1], scale, degree
        )
        swap = np.tile(chain.exps[:, 0] >= 1, 2)  # x'-degree >= 1, u and v seeds
        fields = tuple(
            np.where(swap, plain, marched)
            for marched, plain in zip(fields, frozen.evaluate(xs, ys), strict=True)
        )
    u, v, f, g, h = fields
    keep_uv, keep_fgh = basis_columns(degree)

    y0 = chain.y0.reshape(chain.shape)
    dy0 = chain.rhs(0.0, chain.y0).reshape(chain.shape)
    lam_e, mu_e = chain.anchor.lam, chain.anchor.mu
    f_x = (lam_e + 2 * mu_e) * 2 * y0[0][:, 2] + lam_e * dy0[1][:, 1]
    uv_jet = np.stack([y0[0][:, 1], dy0[0][:, 0], y0[1][:, 1], dy0[1][:, 0]], -1)
    fgh_jet = np.stack([f_x, y0[2][:, 1], dy0[2][:, 0], dy0[3][:, 0]], -1)
    return SeedBasis(
        degree=degree,
        exps=chain.exps,
        scale=scale,
        anchor=chain.anchor,
        uv=np.stack([u[:, keep_uv], v[:, keep_uv]]),
        fgh=np.stack([f[:, keep_fgh], g[:, keep_fgh], h[:, keep_fgh]]),
        uv_jet=uv_jet[keep_uv],
        fgh_jet=fgh_jet[keep_fgh],
        keep_uv=keep_uv,
        keep_fgh=keep_fgh,
    )


# --- the weights -------------------------------------------------------------------


def seed_weights(
    local: np.ndarray,
    theta: np.ndarray,
    seeds: Sequence[SeedBasis],
    *,
    shape: float = 0.4,
    shape_neighbor: int = 3,
    hyper_power: int = 3,
) -> InterfaceWeights:
    """Coupled RBF-FD weights for stencils that see a smooth edge, one
    :class:`SeedBasis` per stencil.

    Same contract as :func:`.interface.interface_weights`: ``local``
    ``(s, n, 2)`` in the interface frame with node 0 the evaluation node,
    ``theta`` ``(s,)`` the frame angle. The Gaussian block is the same; the
    augmenting basis is the seeds, whose values at the nodes and jets at the
    anchor ``seed_basis`` already gives in the stencil coordinate. The
    material at the evaluation node is the seeds' anchor material, and the
    hyperviscosity right-hand sides on the seed columns are zero (module
    docstring).
    """
    local = np.asarray(local, dtype=float)
    s_total = local.shape[0]
    if len(seeds) != s_total:
        raise ValueError("one SeedBasis per stencil")
    gauss = gaussian_rows(
        local, shape=shape, shape_neighbor=shape_neighbor, hyper_power=hyper_power
    )
    scales = np.array([sb.scale for sb in seeds])
    if not np.allclose(scales, gauss.r_max, rtol=1e-12, atol=0):
        raise ValueError("seed bases were built for other stencils (r_max differs)")
    n_uv, n_fgh = seeds[0].n_uv, seeds[0].n_fgh
    aug = Augmentation(
        uv=np.stack([np.concatenate(sb.uv) for sb in seeds]),
        fgh=np.stack([np.concatenate(sb.fgh) for sb in seeds]),
        uv_jet=np.stack([sb.uv_jet for sb in seeds]),
        fgh_jet=np.stack([sb.fgh_jet for sb in seeds]),
        lap_uv=np.zeros((s_total, n_uv, 2)),
        lap_fgh=np.zeros((s_total, n_fgh, 3)),
    )
    material = tuple(
        np.array([getattr(sb.anchor, attr) for sb in seeds])
        for attr in ("lam", "mu", "rho")
    )
    return coupled_weights(gauss, aug, material, theta, hyper_power=hyper_power)
