import numpy as np
import pytest

from rbf_hyperbolic_interfaces.wave1d import (
    LayeredMedium,
    Material,
    build_operators,
    periodic_grid,
)
from rbf_hyperbolic_interfaces.wave1d.domain import Interface
from rbf_hyperbolic_interfaces.wave1d.operators import (
    continuity_matrices,
    interface_weights,
    piecewise_bases,
)


def test_continuity_matrices_constant_material_closed_form() -> None:
    # For constant (c, rho): D**(2m) = c**(2m) d^(2m)/dx^(2m) on both fields,
    # so even rows are k! c**k. Odd rows route through the other field:
    # rho c**2 u_x for u, f_x / rho for f.
    c, rho = 2.0, 1.5
    p = 5
    cu, cf = continuity_matrices(Material(c=c, rho=rho), p)
    k = np.arange(p)
    fact = np.array([1, 1, 2, 6, 24])
    expected_u = np.where(k % 2 == 0, fact * c**k, fact * rho * c ** (k + 1))
    expected_f = np.where(k % 2 == 0, fact * c**k, fact * c ** (k - 1) / rho)
    np.testing.assert_allclose(cu, np.diag(expected_u), atol=1e-12)
    np.testing.assert_allclose(cf, np.diag(expected_f), atol=1e-12)


def test_translated_basis_keeps_rho_c2_u_x_continuous() -> None:
    # Across an interface, rho c^2 u_x must be continuous (f_t continuous).
    left, right = Material(c=1.0, rho=1.0), Material(c=2.0, rho=1.5)
    bases_u, _, _ = piecewise_bases([Interface(0.0, left, right)], 5, 1.0)
    slope_left = bases_u[0][1]
    slope_right = bases_u[1][1]
    np.testing.assert_allclose(
        left.rho * left.c**2 * slope_left, right.rho * right.c**2 * slope_right
    )
    # Values themselves are continuous: constant terms agree.
    np.testing.assert_allclose(bases_u[0][0], bases_u[1][0])


def test_interface_weights_reduce_to_fornberg_without_contrast() -> None:
    same = Material(c=1.3, rho=0.7)
    xs = np.array([-0.025, -0.015, -0.005, 0.005, 0.015])
    w_u, w_f = interface_weights(xs, -0.005, [Interface(0.0, same, same)], 4)
    expected = np.array([1, -8, 0, 8, -1]) / (12 * 0.01)
    np.testing.assert_allclose(w_u, expected, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(w_f, expected, rtol=1e-10, atol=1e-9)


def test_aware_equals_naive_on_uniform_medium() -> None:
    grid = periodic_grid(64)
    medium = LayeredMedium(layer=Material(c=1.0, rho=1.0))
    du_a, df_a = build_operators(grid, medium, mode="aware")
    du_n, df_n = build_operators(grid, medium, mode="naive")
    assert abs(du_a - du_n).max() < 1e-12
    assert abs(df_a - df_n).max() < 1e-12


def test_only_interface_rows_differ() -> None:
    grid = periodic_grid(200)
    medium = LayeredMedium()
    du_a, _ = build_operators(grid, medium, mode="aware")
    du_n, _ = build_operators(grid, medium, mode="naive")
    diff_rows = np.unique(np.nonzero(abs(du_a - du_n).toarray() > 1e-12)[0])
    # 4th order: two nodes either side of each of the two interfaces.
    assert diff_rows.size == 8
    assert np.all(np.abs(grid.x[diff_rows]) < 0.6)


def test_rejects_odd_order() -> None:
    with pytest.raises(ValueError):
        build_operators(periodic_grid(50), LayeredMedium(), order=3)
