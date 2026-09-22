"""Fourier pseudo-spectral reference (issue #30)."""

import numpy as np
import pytest

from rbf_hyperbolic_interfaces.wave1d import (
    LayeredMedium,
    Material,
    gaussian,
    periodic_grid,
    run,
)
from rbf_hyperbolic_interfaces.wave1d.spectral import (
    energy,
    interpolate,
    reference_size,
    run_spectral,
    spectral_derivative,
)


def test_spectral_derivative_is_exact_on_trigonometric_polynomials() -> None:
    grid = periodic_grid(64)
    x = grid.x
    v = np.sin(3 * np.pi * x) + 0.5 * np.cos(7 * np.pi * x) + 2.0
    dv = 3 * np.pi * np.cos(3 * np.pi * x) - 3.5 * np.pi * np.sin(7 * np.pi * x)
    np.testing.assert_allclose(spectral_derivative(v), dv, atol=1e-12)


def test_interpolation_reproduces_samples_and_smooth_functions() -> None:
    grid = periodic_grid(256)
    g = np.exp(np.sin(np.pi * grid.x)) + np.cos(3 * np.pi * grid.x)
    np.testing.assert_allclose(interpolate(g, grid, grid.x), g, atol=1e-13)
    rng = np.random.default_rng(3)
    xt = rng.uniform(-1, 1, 500)
    expected = np.exp(np.sin(np.pi * xt)) + np.cos(3 * np.pi * xt)
    np.testing.assert_allclose(interpolate(g, grid, xt), expected, atol=1e-12)


def test_reference_size_grows_with_edge_stiffness() -> None:
    assert reference_size(0.04) == 1024
    assert reference_size(0.01) == 2048
    assert reference_size(0.0025) == 8192
    with pytest.raises(ValueError):
        reference_size(0.0)


def test_constant_coefficients_translate_the_pulse_exactly() -> None:
    medium = LayeredMedium(background=Material(1.5, 0.8), layer=Material(1.5, 0.8))
    grid, snaps = run_spectral(medium, 512, t_end=0.7, dt=5e-5)
    # Right-going pulse at speed 1.5: f(x, t) = g(x - 1.5 t), u = -f / Z.
    f_exact = gaussian(grid.x, center=-0.5 + 1.5 * 0.7)
    np.testing.assert_allclose(snaps.f[-1], f_exact, atol=1e-9)
    np.testing.assert_allclose(snaps.u[-1], -f_exact / 1.2, atol=1e-9)


def test_time_error_is_fourth_order_in_dt() -> None:
    medium = LayeredMedium(background=Material(1.5, 0.8), layer=Material(1.5, 0.8))
    errors = []
    for dt in (6.5e-4, 3.25e-4, 1.625e-4):
        grid, snaps = run_spectral(medium, 512, t_end=0.7, cfl=0.9, dt=dt)
        f_exact = gaussian(grid.x, center=-0.5 + 1.5 * 0.7)
        errors.append(np.abs(snaps.f[-1] - f_exact).max())
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(rates > 3.8), (errors, rates)


def test_energy_is_conserved_through_a_smooth_edge() -> None:
    medium = LayeredMedium(edge_width=0.02)
    grid, snaps = run_spectral(medium, 1024, t_end=0.8, n_snapshots=4)
    e0 = energy(snaps.u[0], snaps.f[0], grid, medium)
    for u, f in zip(snaps.u[1:], snaps.f[1:], strict=True):
        assert abs(energy(u, f, grid, medium) / e0 - 1) < 1e-9


def test_self_convergence_in_space() -> None:
    # delta = 0.02 is resolved on 1024 nodes (reference_size says so), so
    # doubling the grid at the same time step changes nothing above
    # round-off; the shared RK4 error cancels in the difference.
    medium = LayeredMedium(edge_width=0.02)
    probe = periodic_grid(400).x
    grid_a, a = run_spectral(medium, 1024, t_end=0.6, dt=5e-5)
    grid_b, b = run_spectral(medium, 2048, t_end=0.6, dt=5e-5)
    fa = interpolate(a.f[-1], grid_a, probe)
    fb = interpolate(b.f[-1], grid_b, probe)
    assert np.abs(fa - fb).max() < 1e-10


def test_agrees_with_resolved_naive_fd4() -> None:
    # delta = 0.04 spans 32 cells at n = 1600, so standard FD4 resolves the
    # edge and the two independent discretisations must agree to FD4's own
    # error level.
    medium = LayeredMedium(edge_width=0.04)
    grid_fd = periodic_grid(1600)
    fd = run(grid_fd, medium, mode="naive", t_end=1.0, n_snapshots=1)
    grid_sp, sp = run_spectral(medium, 1024, t_end=1.0)
    f_ref = interpolate(sp.f[-1], grid_sp, grid_fd.x)
    rel = np.linalg.norm(fd.f[-1] - f_ref) / np.linalg.norm(f_ref)
    assert rel < 3e-5, rel
