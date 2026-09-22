import numpy as np
import pytest

from rbf_hyperbolic_interfaces.wave2d import (
    ElasticMaterial,
    LayeredMedium2D,
    SineInterface,
    build_operators,
    exact_plane_wave,
    make_node_set,
    run,
)
from rbf_hyperbolic_interfaces.wave2d.interface import (
    closest_point,
    continuity_matrix,
    derivative_matrices,
    evaluate_basis,
    interface_basis,
    interface_weights,
    pde_operator,
    rotation_matrices,
)
from rbf_hyperbolic_interfaces.wave2d.rbf import monomial_exponents, rbf_fd_weights

BG = ElasticMaterial(lam=1.0, mu=1.0, rho=1.0)
LAYER = ElasticMaterial(lam=4.0, mu=4.0, rho=2.0)
FLAT = LayeredMedium2D()


def test_derivative_matrices_differentiate_polynomials() -> None:
    exps = monomial_exponents(3)
    dx, dy = derivative_matrices(exps)
    # p = x^2 y + 3 y^3 -> p_x = 2 x y, p_y = x^2 + 9 y^2
    index = {tuple(e): i for i, e in enumerate(exps)}
    c = np.zeros(len(exps))
    c[index[(2, 1)]] = 1.0
    c[index[(0, 3)]] = 3.0
    pts = np.array([[0.3, -0.7], [1.1, 0.2]])
    np.testing.assert_allclose(
        evaluate_basis((dx @ c)[:, None], exps, pts)[:, 0], 2 * pts[:, 0] * pts[:, 1]
    )
    np.testing.assert_allclose(
        evaluate_basis((dy @ c)[:, None], exps, pts)[:, 0],
        pts[:, 0] ** 2 + 9 * pts[:, 1] ** 2,
    )


def test_pde_operator_matches_the_equations_on_coefficients() -> None:
    exps = monomial_exponents(2)
    m = len(exps)
    d = pde_operator(LAYER, exps)
    rng = np.random.default_rng(0)
    coeffs = rng.normal(size=5 * m)
    out = d @ coeffs
    dx, dy = derivative_matrices(exps)
    u, v, f, g, h = coeffs.reshape(5, m)
    lam, mu, rho = LAYER.lam, LAYER.mu, LAYER.rho
    expected = np.concatenate(
        [
            (dx @ f + dy @ g) / rho,
            (dx @ g + dy @ h) / rho,
            (lam + 2 * mu) * (dx @ u) + lam * (dy @ v),
            mu * (dy @ u + dx @ v),
            lam * (dx @ u) + (lam + 2 * mu) * (dy @ v),
        ]
    )
    np.testing.assert_allclose(out, expected)


def test_continuity_matrix_is_square_and_no_contrast_gives_identity_basis() -> None:
    exps = monomial_exponents(4)
    c = continuity_matrix(BG, exps)
    assert c.shape == (2 * len(exps), 2 * len(exps))
    assert np.linalg.matrix_rank(c) == 2 * len(exps)
    basis = interface_basis(BG, BG, 3)
    np.testing.assert_allclose(basis.uv[False], basis.uv[True], atol=1e-12)
    np.testing.assert_allclose(basis.fgh[False], basis.fgh[True], atol=1e-12)
    assert basis.n_uv == 20 and basis.n_fgh == 27  # dissertation §3.3.2 counts


@pytest.mark.parametrize("standard_above", [True, False])
def test_basis_satisfies_the_interface_conditions(standard_above: bool) -> None:
    basis = interface_basis(BG, LAYER, 3, standard_above=standard_above)
    xs = np.linspace(-1.5, 1.5, 9)
    line = np.stack([xs, np.zeros_like(xs)], axis=-1)

    def on_line(coeffs: dict, comp: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            evaluate_basis(coeffs[False][comp], basis.exps, line),
            evaluate_basis(coeffs[True][comp], basis.exps, line),
        )

    # Velocity continuous.
    for comp in (0, 1):
        lo, hi = on_line(basis.uv, comp)
        np.testing.assert_allclose(lo, hi, atol=1e-12)
    # Traction (g, h) continuous; the parallel normal stress f jumps.
    for comp in (1, 2):
        lo, hi = on_line(basis.fgh, comp)
        np.testing.assert_allclose(lo, hi, atol=1e-10)
    lo, hi = on_line(basis.fgh, 0)
    assert np.abs(lo - hi).max() > 1.0
    # Rates of traction from the velocity basis are continuous too (k = 1).
    dx, dy = derivative_matrices(basis.exps)
    rates = {}
    for side in (False, True):
        mat = basis.material(side)
        u, v = basis.uv[side]
        rates[side] = (
            mat.mu * (dy @ u + dx @ v),
            mat.lam * (dx @ u) + (mat.lam + 2 * mat.mu) * (dy @ v),
        )
    for comp in (0, 1):
        lo = evaluate_basis(rates[False][comp], basis.exps, line)
        hi = evaluate_basis(rates[True][comp], basis.exps, line)
        np.testing.assert_allclose(lo, hi, atol=1e-10)
    # Full rank on both sides together.
    uv_all = np.concatenate(
        [basis.uv[s].reshape(-1, basis.n_uv) for s in (False, True)]
    )
    fgh_all = np.concatenate(
        [basis.fgh[s].reshape(-1, basis.n_fgh) for s in (False, True)]
    )
    assert np.linalg.matrix_rank(uv_all) == basis.n_uv
    assert np.linalg.matrix_rank(fgh_all) == basis.n_fgh


def test_rotation_matrices_are_the_tensor_transforms() -> None:
    theta = np.array([0.0, 0.4])
    r_uv, r_fgh = rotation_matrices(theta)
    np.testing.assert_allclose(r_uv[0], np.eye(2))
    np.testing.assert_allclose(r_fgh[0], np.eye(3))
    # Rotating a stress tensor: sigma' = Q sigma Q^T with Q = R_uv.
    q = r_uv[1]
    sigma = np.array([[1.3, 0.4], [0.4, -0.7]])
    prime = q @ sigma @ q.T
    fgh = r_fgh[1] @ np.array([sigma[0, 0], sigma[0, 1], sigma[1, 1]])
    np.testing.assert_allclose(fgh, [prime[0, 0], prime[0, 1], prime[1, 1]])


def test_closest_point_on_flat_and_curved_interfaces() -> None:
    flat = SineInterface(0.5)
    x0, th = closest_point(flat, np.array([0.3]), np.array([0.6]))
    assert x0[0] == 0.3 and th[0] == 0.0
    curved = SineInterface(0.5, 0.02)
    x = np.array([0.12, 0.61])
    y = curved.height(x) + 0.03
    x0, th = closest_point(curved, x, y)
    # The foot point is where the displacement is along the normal.
    disp = np.stack([x - x0, y - curved.height(x0)], axis=-1)
    tangent = np.stack([np.cos(th), np.sin(th)], axis=-1)
    np.testing.assert_allclose(np.sum(disp * tangent, axis=1), 0.0, atol=1e-10)


def _stencil_across_interface(seed: int = 5, n: int = 19, h: float = 0.02):
    """A random 19-node stencil straddling y' = 0; node 0 is the evaluation node."""
    rng = np.random.default_rng(seed)
    off = rng.normal(size=(1, n, 2)) * 1.3 * h
    off[0, 0] = 0.0
    local = off + np.array([0.004, -0.003])  # expansion origin off the node
    above = local[..., 1] > 0
    assert above.any() and (~above).any()
    return off, local, above


def test_weights_reduce_to_naive_without_contrast() -> None:
    # No contrast: the (u, v) constraint space is the full degree-3 space,
    # so every stress-rate row must be exactly the naive combination of
    # d/dx and d/dy weights, and (u, v) hyperviscosity must be the naive
    # Delta^3 weights with no cross-field coupling.
    off, local, above = _stencil_across_interface()
    basis = interface_basis(BG, BG, 3)
    w = interface_weights(local, above, np.zeros(1), basis)
    naive = rbf_fd_weights(off, poly_degree=3, hyper_power=3)
    lam, mu, rho = BG.lam, BG.mu, BG.rho
    expected_rows = {
        0: ((lam + 2 * mu) * naive.dx[0], lam * naive.dy[0]),  # f_t
        1: (mu * naive.dy[0], mu * naive.dx[0]),  # g_t
        2: (lam * naive.dx[0], (lam + 2 * mu) * naive.dy[0]),  # h_t
    }
    tol = 1e-8 * np.abs(naive.dx[0]).max()
    for r, (on_u, on_v) in expected_rows.items():
        np.testing.assert_allclose(w.fgh_from_uv[0, r, 0], on_u, atol=tol)
        np.testing.assert_allclose(w.fgh_from_uv[0, r, 1], on_v, atol=tol)
    htol = 1e-6 * np.abs(naive.hyper[0]).max()
    for r in range(2):
        np.testing.assert_allclose(w.hyper_uv[0, r, r], naive.hyper[0], atol=htol)
        np.testing.assert_allclose(w.hyper_uv[0, r, 1 - r], 0.0, atol=htol)
    # The stress-data rows do NOT reduce to naive even without contrast: their
    # constraint space is the 27-dimensional subspace the PDE generates, not
    # all 30 degree-3 polynomial triples (dissertation p. 39). They stay
    # close, and exactness on that subspace is tested separately below.
    scale = np.abs(naive.dx[0]).max()
    assert np.abs(w.uv_from_fgh[0, 0, 0] - naive.dx[0] / rho).max() < 0.15 * scale
    assert np.abs(w.uv_from_fgh[0, 0, 1] - naive.dy[0] / rho).max() < 0.15 * scale
    assert np.abs(w.uv_from_fgh[0, 0, 2]).max() < 0.15 * scale
    assert np.abs(w.uv_from_fgh[0, 1, 0]).max() < 0.15 * scale
    assert np.abs(w.uv_from_fgh[0, 1, 1] - naive.dx[0] / rho).max() < 0.15 * scale
    assert np.abs(w.uv_from_fgh[0, 1, 2] - naive.dy[0] / rho).max() < 0.15 * scale
    hscale = np.abs(naive.hyper[0]).max()
    for r in range(3):
        assert np.abs(w.hyper_fgh[0, r, r] - naive.hyper[0]).max() < 0.15 * hscale


@pytest.mark.parametrize("contrast", [False, True])
def test_weights_are_exact_on_the_piecewise_bases(contrast: bool) -> None:
    # Apply every weight block to every basis function of its constraint
    # space (values at the nodes, each on its own side) and compare with the
    # operator applied to that basis function at the evaluation node. This
    # is the check the end-to-end flat problem cannot make: there u = g = 0,
    # so a wrong g_y term or a swapped stress-rate formula never acts.
    _, local, above = _stencil_across_interface(seed=9)
    basis = interface_basis(BG, LAYER if contrast else BG, 3)
    w = interface_weights(local, above, np.zeros(1), basis)
    # The weights treat p_j(x' / r_max) as the function, with r_max the
    # farthest node from the evaluation node.
    r_max = np.linalg.norm(local[0] - local[0, 0], axis=1).max()
    pts = local[0] / r_max
    side_e = bool(above[0, 0])
    mat = basis.material(side_e)
    dx, dy = derivative_matrices(basis.exps)
    lap3 = np.linalg.matrix_power(dx @ dx + dy @ dy, 3)

    def values(coeffs: dict, comp: int) -> np.ndarray:
        lo = evaluate_basis(coeffs[False][comp], basis.exps, pts)
        hi = evaluate_basis(coeffs[True][comp], basis.exps, pts)
        return np.where(above[0][:, None], hi, lo)  # (n, n_basis)

    def at_eval(coeffs: dict, comp: int, op: np.ndarray, order: int) -> np.ndarray:
        c = op @ coeffs[side_e][comp]
        return evaluate_basis(c, basis.exps, pts[0:1])[0] / r_max**order

    def check(weights: np.ndarray, data: list[np.ndarray], expected: np.ndarray):
        got = sum(weights[c] @ d for c, d in enumerate(data))
        scale = sum(np.abs(weights[c])[:, None] * np.abs(d) for c, d in enumerate(data))
        residual = np.abs(got - expected) / scale.sum(axis=0)
        assert residual.max() < 1e-9, residual.max()

    lam, mu, rho = mat.lam, mat.mu, mat.rho
    u, v = values(basis.uv, 0), values(basis.uv, 1)
    ux, uy = at_eval(basis.uv, 0, dx, 1), at_eval(basis.uv, 0, dy, 1)
    vx, vy = at_eval(basis.uv, 1, dx, 1), at_eval(basis.uv, 1, dy, 1)
    check(w.fgh_from_uv[0, 0], [u, v], (lam + 2 * mu) * ux + lam * vy)
    check(w.fgh_from_uv[0, 1], [u, v], mu * (uy + vx))
    check(w.fgh_from_uv[0, 2], [u, v], lam * ux + (lam + 2 * mu) * vy)
    for r in range(2):
        check(w.hyper_uv[0, r], [u, v], at_eval(basis.uv, r, lap3, 6))

    f, g, hh = (values(basis.fgh, c) for c in range(3))
    fx, gx = at_eval(basis.fgh, 0, dx, 1), at_eval(basis.fgh, 1, dx, 1)
    gy, hy = at_eval(basis.fgh, 1, dy, 1), at_eval(basis.fgh, 2, dy, 1)
    check(w.uv_from_fgh[0, 0], [f, g, hh], (fx + gy) / rho)
    check(w.uv_from_fgh[0, 1], [f, g, hh], (gx + hy) / rho)
    for r in range(3):
        check(w.hyper_fgh[0, r], [f, g, hh], at_eval(basis.fgh, r, lap3, 6))
    if contrast:
        # With contrast the u_t row genuinely uses h data (coupled basis).
        assert (
            np.abs(w.uv_from_fgh[0, 0, 2]).max()
            > 0.01 * np.abs(w.uv_from_fgh[0, 0, 0]).max()
        )


def test_thin_layer_stencils_are_refused() -> None:
    # Interfaces 2h apart with single straddling rows: a 19-node stencil
    # near one interface reaches past the other, which the port does not
    # handle (MATLAB's "triple" stencils), so it must say so.
    medium = LayeredMedium2D(lower=SineInterface(0.4), upper=SineInterface(0.5))
    nodes = make_node_set(medium, 400, rows_per_side=1, repulsion_steps=10)
    build_operators(nodes, medium, mode="naive")  # naive path is fine
    with pytest.raises(NotImplementedError, match="both interfaces"):
        build_operators(nodes, medium, mode="aware")


def test_aware_operator_only_changes_rows_near_interfaces() -> None:
    nodes = make_node_set(FLAT, 900, repulsion_steps=20)
    naive = build_operators(nodes, FLAT, mode="naive")
    aware = build_operators(nodes, FLAT, mode="aware")
    dist = FLAT.distance_to_interfaces(nodes.x, nodes.y)
    near = dist <= 4 * nodes.h
    np.testing.assert_array_equal(np.sort(aware.interface_nodes), np.flatnonzero(near))
    diff = abs(aware.elastic - naive.elastic)
    changed = np.unique(diff.nonzero()[0] % nodes.n)
    assert set(changed) <= set(aware.interface_nodes)
    # u_t rows now also draw on h data through the coupled stress basis.
    i = aware.interface_nodes[0]
    row = aware.elastic[[i], :].toarray().ravel()
    assert np.abs(row[4 * nodes.n : 5 * nodes.n]).max() > 0


def test_aware_beats_naive_on_the_flat_two_interface_problem() -> None:
    errs = {"naive": [], "aware": []}
    for n in (2500, 4900):
        nodes = make_node_set(FLAT, n, repulsion_steps=40)
        exact = exact_plane_wave(nodes, 0.3, FLAT)
        for mode in errs:
            snaps = run(nodes, FLAT, mode=mode, t_end=0.3, n_snapshots=1)
            v = snaps.field("v")[-1]
            assert np.isfinite(v).all()
            errs[mode].append(np.linalg.norm(v - exact[1]) / np.linalg.norm(exact[1]))
    assert errs["aware"][1] < 0.6 * errs["naive"][1], errs
    assert errs["aware"][1] < 0.45 * errs["aware"][0], errs
