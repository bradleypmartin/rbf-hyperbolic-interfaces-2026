"""Elastic seed bases (#38) and seed-augmented weights (#39) for a straight
stiff feature."""

import numpy as np
import pytest

from pdes_demo.fd_weights import fornberg_weights
from pdes_demo.wave2d import (
    ElasticMaterial,
    LayeredMedium2D,
    make_node_set,
    minimal_image,
    periodic_knn,
)
from pdes_demo.wave2d.interface import (
    evaluate_basis,
    interface_basis,
    interface_weights,
)
from pdes_demo.wave2d.rbf import monomial_exponents, rbf_fd_weights
from pdes_demo.wave2d.seeds import (
    SeedChain,
    basis_columns,
    chain_matrix,
    normal_profile,
    seed_basis,
    seed_weights,
    shift_matrix,
)

BG = ElasticMaterial(lam=1.0, mu=1.0, rho=1.0)
LAYER = ElasticMaterial(lam=4.0, mu=4.0, rho=2.0)
FLAT = LayeredMedium2D()
DEGREE = 3
EXPS1 = monomial_exponents(DEGREE + 1)
EXPS = monomial_exponents(DEGREE)
M1, M = len(EXPS1), len(EXPS)
KEEP_UV, KEEP_FGH = basis_columns(DEGREE)


@pytest.fixture(scope="module")
def nodes():
    return make_node_set(FLAT, 400)


def _stencil(nodes, row_offset: float, x_near: float = 0.3):
    """A 19-node interface stencil centred on a fixed row of the lower
    interface, in the interface frame (flat: no rotation)."""
    fixed = np.flatnonzero(nodes.fixed)
    off = FLAT.lower.vertical_offset(nodes.x[fixed], nodes.y[fixed])
    cand = fixed[np.isclose(off, row_offset * nodes.h)]
    centre = int(cand[np.argmin(np.abs(nodes.x[cand] - x_near))])
    idx, _ = periodic_knn(nodes.xy, 19, query=nodes.xy[[centre]])
    origin = np.array([nodes.x[centre], FLAT.lower.y0])
    return minimal_image(nodes.xy[idx[0]] - origin), centre


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


# --- algebra -----------------------------------------------------------------------


def test_chain_matrix_is_the_constant_coefficient_operator_on_monomials() -> None:
    # L (x^a y^b, 0) = (1/rho) [K a(a-1) x^(a-2) y^b + mu b(b-1) x^a y^(b-2),
    #                           (lam + mu) a b x^(a-1) y^(b-1)], and the mirror
    # image for (0, x^a y^b); K = lam + 2 mu.
    c = chain_matrix(LAYER, EXPS1)
    lam, mu, rho = LAYER.lam, LAYER.mu, LAYER.rho
    k = lam + 2 * mu
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(EXPS1)}
    expected = np.zeros_like(c)
    for i, (a, b) in enumerate(EXPS1):
        for comp in (0, 1):
            col = comp * M1 + i
            same, other = (k, mu) if comp == 0 else (mu, k)
            if a >= 2:
                row = comp * M1 + index[(a - 2, b)]
                expected[row, col] += same * a * (a - 1) / rho
            if b >= 2:
                row = comp * M1 + index[(a, b - 2)]
                expected[row, col] += other * b * (b - 1) / rho
            if a >= 1 and b >= 1:
                row = (1 - comp) * M1 + index[(a - 1, b - 1)]
                expected[row, col] += (lam + mu) * a * b / rho
    np.testing.assert_allclose(c, expected, atol=1e-13)


def test_shift_matrix_re_expands_polynomials_about_a_new_origin() -> None:
    rng = np.random.default_rng(1)
    coeffs = rng.normal(size=(M1, 3))
    x0, y0 = 0.3, -0.7
    t = shift_matrix(EXPS1, x0, y0)
    pts = rng.normal(size=(6, 2))
    shifted = evaluate_basis(coeffs, EXPS1, pts - [x0, y0])
    np.testing.assert_allclose(evaluate_basis(t @ coeffs, EXPS1, pts), shifted)


def test_guards() -> None:
    with pytest.raises(ValueError, match="edge_width"):
        normal_profile(FLAT, FLAT.lower, 0.3)
    smooth = LayeredMedium2D(edge_width=0.01)
    with pytest.raises(ValueError, match="not one of"):
        normal_profile(smooth, FLAT.upper.__class__(0.3), 0.3)
    with pytest.raises(ValueError, match="degree"):
        SeedChain(normal_profile(smooth, smooth.lower, 0.3), 0.02, 0.05, 0)


# --- the seeds ---------------------------------------------------------------------


def test_constant_material_seeds_are_the_monomial_basis(nodes) -> None:
    same = ElasticMaterial(lam=1.3, mu=0.7, rho=0.9)
    uniform = LayeredMedium2D(background=same, layer=same, edge_width=0.01)
    local, centre = _stencil(nodes, 0.5)
    seeds = seed_basis(local, normal_profile(uniform, uniform.lower, 0.0), DEGREE)
    basis = interface_basis(same, same, DEGREE)
    assert seeds.n_uv == basis.n_uv == 20 and seeds.n_fgh == basis.n_fgh == 27
    pts = (local - local[0]) / seeds.scale
    for comp in range(2):
        poly = evaluate_basis(basis.uv[True][comp], EXPS, pts)
        np.testing.assert_allclose(seeds.uv[comp], poly, rtol=0, atol=1e-11)
    for comp in range(3):
        poly = evaluate_basis(basis.fgh[True][comp], EXPS, pts)
        np.testing.assert_allclose(seeds.fgh[comp], poly, rtol=0, atol=1e-11)


def _shifted_jump_basis(local: np.ndarray, above_e: bool, scale: float):
    """``interface_basis`` re-expanded about the evaluation node and scaled
    like the seeds, evaluated on each node's own side."""
    basis = interface_basis(BG, LAYER, DEGREE, standard_above=above_e)
    xe, ye = local[0] / scale
    t1 = shift_matrix(EXPS1, xe, ye)
    t = t1[:M, :M]
    block = np.kron(np.eye(2), t)
    block1 = np.kron(np.eye(2), t1)
    pts = local / scale
    above = local[:, 1] > 0
    uv = np.empty((2, len(local), 2 * M))
    fgh = np.empty((3, len(local), 2 * M1 - 3))
    for side in (False, True):
        sel = above == side
        for c in range(2):
            uv[c, sel] = evaluate_basis(basis.uv[side][c] @ block, EXPS, pts[sel])
        # The shift mixes in the dropped columns: constants (no stress) and
        # v = x, whose stress is that of u = y (kept column 1).
        full = np.zeros((3, M, 2 * M1))
        full[:, :, KEEP_FGH] = basis.fgh[side]
        full[:, :, M1 + 1] = basis.fgh[side][:, :, 1]
        for c in range(3):
            coeffs = (full[c] @ block1)[:, KEEP_FGH]
            fgh[c, sel] = evaluate_basis(coeffs, EXPS, pts[sel])
    return uv, fgh


@pytest.mark.parametrize("row", [0.5, -0.5])
def test_jump_limit_recovers_the_interface_basis_at_first_order(nodes, row) -> None:
    local, centre = _stencil(nodes, row)
    errors_uv, errors_fgh = [], []
    for width in (1e-3, 1e-4, 1e-5):
        medium = LayeredMedium2D(edge_width=width)
        profile = normal_profile(medium, medium.lower, nodes.x[centre])
        seeds = seed_basis(local, profile, DEGREE)
        uv, fgh = _shifted_jump_basis(local, row > 0, seeds.scale)
        errors_uv.append(_rel(seeds.uv, uv))
        errors_fgh.append(_rel(seeds.fgh, fgh))
    for errors in (errors_uv, errors_fgh):
        ratios = np.array(errors[:-1]) / np.array(errors[1:])
        assert np.all(ratios > 8), errors
        assert errors[-1] < 1e-3, errors


def test_rigid_motions_have_no_stress_and_the_shear_pair_shares_one(nodes) -> None:
    # Through a real edge: u = 1 and v = 1 stay stress-free, and the seeds of
    # u = y and v = x, (int mu_e / mu, 0) and (int (mu_e / mu - 1), x), both
    # carry the constant shear g = mu_e and nothing else, which is why the
    # same three columns are dropped as in interface_basis.
    medium = LayeredMedium2D(edge_width=0.01)
    local, centre = _stencil(nodes, 0.5)
    off = local - local[0]
    scale = np.max(np.linalg.norm(off, axis=1))
    chain = SeedChain(
        normal_profile(medium, medium.lower, nodes.x[centre]), local[0, 1], scale, 3
    )
    u, v, f, g, h = chain.evaluate(off[:, 0] / scale, off[:, 1] / scale)
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(EXPS1)}
    u_const, v_const = index[(0, 0)], M1 + index[(0, 0)]
    u_y, v_x = index[(0, 1)], M1 + index[(1, 0)]
    for col in (u_const, v_const):
        assert np.abs(np.stack([f[:, col], g[:, col], h[:, col]])).max() < 1e-13
    mu_e = chain.anchor.mu
    for col in (u_y, v_x):
        np.testing.assert_allclose(g[:, col], mu_e, rtol=1e-12)
        assert np.abs(f[:, col]).max() < 1e-12 and np.abs(h[:, col]).max() < 1e-12
    assert np.abs(u[:, u_y] - u[:, v_x]).max() > 1e-3  # the seeds themselves differ
    assert list(KEEP_FGH) == [j for j in range(2 * M1) if j not in (0, M1, M1 + 1)]
    assert u_const == 0 and v_const == M1 and v_x == M1 + 1


def test_seeds_satisfy_the_2d_operator_applied_by_finite_differences() -> None:
    # Independent of every formula in the module except the chain matrix
    # (checked against its closed form above): evaluate each seed as a 2-D
    # function on a tensor grid through a delta = 0.01 edge, apply the
    # second-order elastic operator of domain.py's docstring with 8th-order
    # finite differences in both x and y (exact in x, where the seeds are
    # polynomials of degree <= 4), and compare with rho sum_e' C[e', e] S_e'.
    medium = LayeredMedium2D(edge_width=0.01)
    chain = SeedChain(normal_profile(medium, medium.lower, 0.3), 0.02, 0.06, 3)
    nx, ny = 25, 401
    xg = np.linspace(-0.6, 0.6, nx)
    yg = np.linspace(-1.0, 1.0, ny)
    xx, yy = np.meshgrid(xg, yg, indexing="ij")
    fields = chain.evaluate(xx.ravel(), yy.ravel())
    u, v = (c.reshape(nx, ny, chain.n_seeds) for c in fields[:2])
    lam, mu, rho = (
        v_[None, :, None] for v_ in chain.profile.material(chain.y_e + chain.scale * yg)
    )
    k = lam + 2 * mu

    def fd(arr: np.ndarray, axis: int) -> np.ndarray:
        h = (xg if axis == 0 else yg)[1] - (xg if axis == 0 else yg)[0]
        w = fornberg_weights(0.0, np.arange(-4, 5) * h, 1)[1]
        n = arr.shape[axis]
        out = np.zeros_like(arr)
        core = [slice(None)] * arr.ndim
        core[axis] = slice(4, n - 4)
        for i, wi in enumerate(w):
            src = [slice(None)] * arr.ndim
            src[axis] = slice(i, n - 8 + i)
            out[tuple(core)] += wi * arr[tuple(src)]
        return out

    u_x, u_y, v_x, v_y = fd(u, 0), fd(u, 1), fd(v, 0), fd(v, 1)
    sxx = k * u_x + lam * v_y
    sxy = mu * (u_y + v_x)
    syy = lam * u_x + k * v_y
    lhs_u = fd(sxx, 0) + fd(sxy, 1)
    lhs_v = fd(sxy, 0) + fd(syy, 1)
    c = chain.chain_t.T  # C[e', e]
    rhs_u = rho * (u @ c)
    rhs_v = rho * (v @ c)
    interior = (slice(8, nx - 8), slice(8, ny - 8))
    err = max(
        np.abs(lhs_u - rhs_u)[interior].max(), np.abs(lhs_v - rhs_v)[interior].max()
    )
    scale = max(np.abs(rhs_u).max(), np.abs(rhs_v).max())
    assert scale > 100  # the degree-4 seeds drive O(K * 12) right-hand sides
    assert err / scale < 1e-9
    # The seeds of degree <= 1 are in ker L: nothing drives them.
    low = [i for i, (p, q) in enumerate(EXPS1) if p + q <= 1]
    assert np.abs(rhs_u[:, :, low + [M1 + i for i in low]]).max() == 0
    assert np.abs(lhs_u[interior][:, :, low]).max() / scale < 1e-9


def test_jets_at_the_anchor_are_the_monomial_jets_and_the_chain_constants(
    nodes,
) -> None:
    medium = LayeredMedium2D(edge_width=0.0025)
    local, centre = _stencil(nodes, 0.5)
    seeds = seed_basis(local, normal_profile(medium, medium.lower, 0.0), DEGREE)
    # Velocity seeds: only X and Y have a first derivative at the anchor.
    expected = np.zeros((seeds.n_uv, 4))
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(EXPS)}
    expected[index[(1, 0)], 0] = 1.0  # u = X: u_X
    expected[index[(0, 1)], 1] = 1.0  # u = Y: u_Y
    expected[M + index[(1, 0)], 2] = 1.0  # v = X: v_X
    expected[M + index[(0, 1)], 3] = 1.0  # v = Y: v_Y
    np.testing.assert_allclose(seeds.uv_jet, expected, atol=1e-15)
    # Stress seeds: (f_X + g_Y) / rho_e and (g_X + h_Y) / rho_e are the
    # velocity rates of the seed at the anchor, i.e. the constant terms of
    # the chain's right-hand side.
    c = chain_matrix(seeds.anchor, EXPS1)
    f_x, g_x, g_y, h_y = seeds.fgh_jet.T
    np.testing.assert_allclose((f_x + g_y) / seeds.anchor.rho, c[0, KEEP_FGH])
    np.testing.assert_allclose((g_x + h_y) / seeds.anchor.rho, c[M1, KEEP_FGH])
    assert np.abs(seeds.fgh_jet).max() > 1  # the degree-2 seeds have unit rates


def test_seed_blocks_are_conditioned_like_the_polynomial_block(nodes) -> None:
    local, centre = _stencil(nodes, 0.5)
    scale = np.max(np.linalg.norm(local - local[0], axis=1))
    uv, fgh = _shifted_jump_basis(local, True, scale)
    cond_uv = np.linalg.cond(np.concatenate(uv))
    cond_fgh = np.linalg.cond(np.concatenate(fgh))
    assert cond_uv < 100 and cond_fgh < 100
    for width in (nodes.h / 8, nodes.h / 2, nodes.h):  # h / 8 to the band's limit
        medium = LayeredMedium2D(edge_width=width)
        profile = normal_profile(medium, medium.lower, nodes.x[centre])
        seeds = seed_basis(local, profile, DEGREE)
        assert np.linalg.cond(np.concatenate(seeds.uv)) < 2 * cond_uv, width
        assert np.linalg.cond(np.concatenate(seeds.fgh)) < 2 * cond_fgh, width


# --- the weights (#39) -------------------------------------------------------------

BLOCKS = ("fgh_from_uv", "uv_from_fgh", "hyper_uv", "hyper_fgh")


def _weights_for(local, medium, x0, degree=DEGREE):
    seeds = seed_basis(local, normal_profile(medium, medium.lower, x0), degree)
    return seed_weights(local[None], np.zeros(1), [seeds]), seeds


def test_seed_weights_reduce_to_naive_in_constant_material(nodes) -> None:
    # Acceptance criterion 1: no contrast, so the seeds are the monomials and
    # the stress-rate and Delta^3 rows on (u, v) data are the plain degree-3
    # RBF-FD weights; every block equals the polynomial interface weights,
    # whose constraint spaces are the same (the 27-dimensional stress space
    # is not the full degree-3 space, dissertation p. 39).
    same = ElasticMaterial(lam=1.3, mu=0.7, rho=0.9)
    uniform = LayeredMedium2D(background=same, layer=same, edge_width=0.01)
    local, _ = _stencil(nodes, 0.5)
    w, _ = _weights_for(local, uniform, 0.0)
    naive = rbf_fd_weights((local - local[0])[None], poly_degree=3, hyper_power=3)
    lam, mu = same.lam, same.mu
    rows = {
        0: ((lam + 2 * mu) * naive.dx[0], lam * naive.dy[0]),
        1: (mu * naive.dy[0], mu * naive.dx[0]),
        2: (lam * naive.dx[0], (lam + 2 * mu) * naive.dy[0]),
    }
    tol = 1e-10 * np.abs(naive.dx[0]).max()
    for r, (on_u, on_v) in rows.items():
        np.testing.assert_allclose(w.fgh_from_uv[0, r, 0], on_u, atol=tol)
        np.testing.assert_allclose(w.fgh_from_uv[0, r, 1], on_v, atol=tol)
    htol = 1e-10 * np.abs(naive.hyper[0]).max()
    for r in range(2):
        np.testing.assert_allclose(w.hyper_uv[0, r, r], naive.hyper[0], atol=htol)
        np.testing.assert_allclose(w.hyper_uv[0, r, 1 - r], 0.0, atol=htol)
    above = (local[:, 1] > 0)[None]
    w_poly = interface_weights(
        local[None], above, np.zeros(1), interface_basis(same, same, DEGREE)
    )
    for name in BLOCKS:
        a, b = getattr(w, name), getattr(w_poly, name)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-10 * np.abs(b).max())


@pytest.mark.parametrize("row", [0.5, -0.5])
@pytest.mark.parametrize("standard_above", [True, False])
def test_seed_weights_tend_to_the_interface_weights_at_first_order(
    nodes, row, standard_above
) -> None:
    # Acceptance criterion 2, all five fields: the weights are set by the
    # span, and the seeds' span tends to that of the piecewise polynomials,
    # whichever side carries the standard monomials (the translation keeps
    # the degree, so both choices span the same 20 velocity and 27 stress
    # functions).
    local, centre = _stencil(nodes, row)
    above = (local[:, 1] > 0)[None]
    basis = interface_basis(BG, LAYER, DEGREE, standard_above=standard_above)
    w_jump = interface_weights(local[None], above, np.zeros(1), basis)
    errors = []
    for width in (1e-3, 1e-4, 1e-5):
        w, _ = _weights_for(local, LayeredMedium2D(edge_width=width), nodes.x[centre])
        errors.append(max(_rel(getattr(w, k), getattr(w_jump, k)) for k in BLOCKS))
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all(ratios > 8), errors
    assert errors[-1] < 1e-3, errors


def test_seed_weights_are_exact_on_the_seed_space(nodes) -> None:
    # Apply every weight block to every seed (values at the nodes) and
    # compare with the elastic rate of that seed at the evaluation node,
    # from the jets; the hyperviscosity blocks must annihilate the seeds.
    local, centre = _stencil(nodes, 0.5)
    medium = LayeredMedium2D(edge_width=nodes.h / 4)
    w, seeds = _weights_for(local, medium, nodes.x[centre])
    sc = 1.0 / seeds.scale
    lam, mu, rho = seeds.anchor.lam, seeds.anchor.mu, seeds.anchor.rho
    u, v = seeds.uv
    f, g, h = seeds.fgh
    ux, uy, vx, vy = seeds.uv_jet.T * sc
    fx, gx, gy, hy = seeds.fgh_jet.T * sc

    def check(weights: np.ndarray, data: list[np.ndarray], expected: np.ndarray):
        got = sum(weights[c] @ d for c, d in enumerate(data))
        scale = sum(np.abs(weights[c])[:, None] * np.abs(d) for c, d in enumerate(data))
        residual = np.abs(got - expected) / scale.sum(axis=0)
        assert residual.max() < 1e-9, residual.max()

    check(w.fgh_from_uv[0, 0], [u, v], (lam + 2 * mu) * ux + lam * vy)
    check(w.fgh_from_uv[0, 1], [u, v], mu * (uy + vx))
    check(w.fgh_from_uv[0, 2], [u, v], lam * ux + (lam + 2 * mu) * vy)
    check(w.uv_from_fgh[0, 0], [f, g, h], (fx + gy) / rho)
    check(w.uv_from_fgh[0, 1], [f, g, h], (gx + hy) / rho)
    for r in range(2):
        check(w.hyper_uv[0, r], [u, v], np.zeros(seeds.n_uv))
    for r in range(3):
        check(w.hyper_fgh[0, r], [f, g, h], np.zeros(seeds.n_fgh))
    # Through an edge the u_t row draws on h data (coupled stress seeds).
    assert (
        np.abs(w.uv_from_fgh[0, 0, 2]).max()
        > 0.01 * np.abs(w.uv_from_fgh[0, 0, 0]).max()
    )


def test_seed_weights_guards(nodes) -> None:
    local, centre = _stencil(nodes, 0.5)
    medium = LayeredMedium2D(edge_width=0.01)
    seeds = seed_basis(
        local, normal_profile(medium, medium.lower, nodes.x[centre]), DEGREE
    )
    with pytest.raises(ValueError, match="per stencil"):
        seed_weights(local[None], np.zeros(1), [seeds, seeds])
    with pytest.raises(ValueError, match="r_max"):
        seed_weights(1.1 * local[None], np.zeros(1), [seeds])
