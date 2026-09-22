"""ODE-continued seed stencils for stiff smooth edges (issues #31, #32)."""

import numpy as np
import pytest
from scipy.integrate import quad

from rbf_hyperbolic_interfaces.fd_weights import fornberg_weights
from rbf_hyperbolic_interfaces.wave1d import (
    LayeredMedium,
    Material,
    build_operators,
    periodic_grid,
    run,
)
from rbf_hyperbolic_interfaces.wave1d.domain import Interface
from rbf_hyperbolic_interfaces.wave1d.operators import aware_rows, interface_weights
from rbf_hyperbolic_interfaces.wave1d.spectral import (
    interpolate,
    run_spectral,
    spectral_derivative,
)
from rbf_hyperbolic_interfaces.wave1d.stiff import seed_basis, stiff_weights

XS = np.array([-0.025, -0.015, -0.005, 0.005, 0.015])
XE = -0.005
LEFT, RIGHT = Material(c=1.0, rho=1.0), Material(c=2.0, rho=1.5)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


def test_constant_material_reduces_to_fornberg() -> None:
    same = Material(c=1.3, rho=0.7)
    medium = LayeredMedium(background=same, layer=same, edge_width=0.01)
    w_u, w_f = stiff_weights(XS, XE, medium, 4)
    expected = fornberg_weights(XE, XS, 1)[1]
    np.testing.assert_allclose(w_u, expected, rtol=0, atol=1e-11)
    np.testing.assert_allclose(w_f, expected, rtol=0, atol=1e-11)


def test_seeds_match_their_closed_forms() -> None:
    # With rho = 1: phi_1 = K_e int dx/K and phi_2 = 2 K_e int (s - x_e)/K(s) ds,
    # both from x_e, so the ODE march can be checked against quadrature.
    medium = LayeredMedium(
        background=Material(1.0, 1.0), layer=Material(2.0, 1.0), edge_width=0.004
    )
    xs = np.array([-0.02, -0.01, 0.003, 0.012, 0.02])
    xe = 0.003
    a_u, _, h_s = seed_basis(xs, xe, medium, 5)
    k_e = float(medium.c_at(xe) ** 2)

    def inv_k(s: float) -> float:
        return 1.0 / float(medium.c_at(s) ** 2)

    for j, xj in enumerate(xs):
        pts = sorted({0.0, medium.layer_start} & set(np.linspace(xe, xj, 3)))
        phi1 = k_e * quad(inv_k, xe, xj, points=pts or None, epsabs=1e-14)[0]
        phi2 = 2 * k_e * quad(lambda s: (s - xe) * inv_k(s), xe, xj, epsabs=1e-14)[0]
        assert a_u[0, j] == pytest.approx(1.0)
        assert a_u[1, j] * h_s == pytest.approx(phi1, abs=1e-12)
        assert a_u[2, j] * h_s**2 == pytest.approx(phi2, abs=1e-12)


def test_jump_limit_recovers_the_interface_weights_at_first_order() -> None:
    w_iu, w_if = interface_weights(XS, XE, [Interface(0.0, LEFT, RIGHT)], 4)
    errors_u, errors_f = [], []
    for width in (1e-3, 1e-4, 1e-5):
        medium = LayeredMedium(background=LEFT, layer=RIGHT, edge_width=width)
        w_u, w_f = stiff_weights(XS, XE, medium, 4)
        errors_u.append(_rel(w_u, w_iu))
        errors_f.append(_rel(w_f, w_if))
    for errors in (errors_u, errors_f):
        ratios = np.array(errors[:-1]) / np.array(errors[1:])
        assert np.all(ratios > 8), errors
        assert errors[-1] < 2e-3, errors


def test_jump_limit_holds_for_the_thin_layer_double_cross() -> None:
    xs = np.array([-0.015, -0.005, 0.005, 0.015, 0.025])
    xe = 0.005
    thin = LayeredMedium(background=LEFT, layer=RIGHT, layer_width=0.01)
    w_iu, w_if = interface_weights(xs, xe, list(thin.interfaces), 4)
    smooth = LayeredMedium(
        background=LEFT, layer=RIGHT, layer_width=0.01, edge_width=1e-5
    )
    w_u, w_f = stiff_weights(xs, xe, smooth, 4)
    assert _rel(w_u, w_iu) < 2e-3
    assert _rel(w_f, w_if) < 2e-3


def test_seed_interpolation_stays_well_conditioned_for_every_width() -> None:
    # Extended complete Chebyshev system: the seed matrix is nonsingular for
    # any width, and in the stencil coordinate its conditioning is that of a
    # small Vandermonde matrix.
    for width in (1e-5, 1e-3, 1e-1, 1.0):
        medium = LayeredMedium(background=LEFT, layer=RIGHT, edge_width=width)
        a_u, a_f, _ = seed_basis(XS, XE, medium, 5)
        assert np.linalg.cond(a_u) < 100, width
        assert np.linalg.cond(a_f) < 100, width


def test_aware_rows_reach_about_19_widths_and_the_rest_is_fornberg() -> None:
    grid = periodic_grid(400)
    medium = LayeredMedium(edge_width=0.005)
    rows = aware_rows(grid, medium)
    du_a, _ = build_operators(grid, medium, mode="aware")
    du_n, _ = build_operators(grid, medium, mode="naive")
    changed = np.abs(du_a - du_n).max(axis=1).toarray().ravel() > 1e-12
    # Every changed row is an aware row; the outermost aware rows see only a
    # 1e-16 material variation and so land back on Fornberg's weights.
    assert not np.any(changed & ~rows)
    near_edge = np.zeros(grid.n, dtype=bool)
    for edge in (medium.layer_start, medium.layer_end):
        near_edge |= np.abs(grid.x - edge) < 5 * medium.edge_width
    assert np.all(changed[near_edge])
    assert 60 < rows.sum() < 100


def test_semi_discrete_operator_has_no_growing_modes() -> None:
    grid = periodic_grid(400)
    medium = LayeredMedium(edge_width=0.005)
    du, df = build_operators(grid, medium, mode="aware")
    rho = medium.rho_at(grid.x)
    k = rho * medium.c_at(grid.x) ** 2
    n = grid.n
    m = np.block(
        [
            [np.zeros((n, n)), df.toarray() / rho[:, None]],
            [du.toarray() * k[:, None], np.zeros((n, n))],
        ]
    )
    ev = np.linalg.eigvals(m)
    assert ev.real.max() < 1e-6


@pytest.fixture(scope="module")
def reference_delta_001() -> tuple:
    medium = LayeredMedium(edge_width=0.01)
    grid, snaps = run_spectral(
        medium, 2048, t_end=1.0, dt=1e-4, n_snapshots=1, pulse_sharpness=60.0
    )
    return medium, grid, snaps


def test_seed_stencils_are_fourth_order_on_the_true_solution(
    reference_delta_001: tuple,
) -> None:
    # Local truncation error on the reference solution's profile, over the
    # rows the seed construction rebuilds: fourth order whether or not the
    # grid resolves the edge (h = 2 delta at n = 100).
    medium, gref, sref = reference_delta_001
    ux_ref = spectral_derivative(sref.u[-1])
    errors = {"naive": [], "aware": []}
    for n in (100, 200, 400):
        grid = periodic_grid(n)
        u_c = interpolate(sref.u[-1], gref, grid.x)
        ux_c = interpolate(ux_ref, gref, grid.x)
        rows = aware_rows(grid, medium)
        for mode in errors:
            du, _ = build_operators(grid, medium, mode=mode)
            errors[mode].append(
                np.abs(du @ u_c - ux_c)[rows].max() / np.abs(ux_c).max()
            )
    rates = np.log2(np.array(errors["aware"][:-1]) / np.array(errors["aware"][1:]))
    assert np.all(rates > 3.5), (errors, rates)
    assert errors["aware"][0] < errors["naive"][0] / 3, errors


def test_end_to_end_seed_stencils_beat_naive_through_an_unresolved_edge(
    reference_delta_001: tuple,
) -> None:
    medium, gref, sref = reference_delta_001
    errors = {"naive": [], "aware": []}
    for n in (100, 200):
        grid = periodic_grid(n)
        f_ref = interpolate(sref.f[-1], gref, grid.x)
        for mode in errors:
            snaps = run(
                grid, medium, mode=mode, t_end=1.0, n_snapshots=1, pulse_sharpness=60.0
            )
            errors[mode].append(
                np.linalg.norm(snaps.f[-1] - f_ref) / np.linalg.norm(f_ref)
            )
    rate = np.log2(errors["aware"][0] / errors["aware"][1])
    assert rate > 3.5, errors
    assert errors["aware"][0] < errors["naive"][0] / 4, errors
