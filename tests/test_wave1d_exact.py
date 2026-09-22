import numpy as np
import pytest

from rbf_hyperbolic_interfaces.wave1d import (
    LayeredMedium,
    Material,
    exact_solution,
    right_going_pulse,
)
from rbf_hyperbolic_interfaces.wave1d.domain import periodic_grid

MEDIUM = LayeredMedium(layer=Material(c=2.0, rho=1.0), layer_width=0.5)  # Z: 1 -> 2


def test_matches_initial_condition() -> None:
    grid = periodic_grid(400)
    u0, f0 = right_going_pulse(grid.x, MEDIUM)
    u, f = exact_solution(grid.x, 0.0, MEDIUM)
    np.testing.assert_allclose(f, f0, atol=1e-14)
    np.testing.assert_allclose(u, u0, atol=1e-14)


def test_reflection_and_transmission_amplitudes() -> None:
    # Z1 = 1, Z2 = 2: R = 1/3, T = 4/3 into the layer; T' = 2/3, R' = -1/3 out.
    # At t = 1.2 the pulses are well separated: reflected (1/3) at -0.7,
    # transmitted (4/3 * 2/3) at 0.95, once-bounced back out (4/3 * -1/3 * 2/3)
    # at -0.2, twice-bounced inside the layer (4/3 * -1/3 * -1/3) at 0.4.
    _, f = exact_solution(np.array([-0.7, -0.2, 0.95]), 1.2, MEDIUM)
    np.testing.assert_allclose(f, [1 / 3, -8 / 27, 8 / 9], atol=1e-9)
    # Inside the layer the leading tail has already begun reflecting off the
    # far edge, so the peak is only approximately the ray amplitude.
    _, f_layer = exact_solution(np.array([0.4]), 1.2, MEDIUM)
    np.testing.assert_allclose(f_layer, [4 / 27], atol=2e-4)


@pytest.mark.parametrize("t", [0.5, 0.75, 1.0, 1.3, 1.9])
def test_u_and_f_continuous_at_interfaces(t: float) -> None:
    eps = 1e-9
    for xi in (MEDIUM.layer_start, MEDIUM.layer_end):
        ul, fl = exact_solution(np.array([xi - eps]), t, MEDIUM)
        ur, fr = exact_solution(np.array([xi + eps]), t, MEDIUM)
        assert abs(fl[0] - fr[0]) < 1e-6
        assert abs(ul[0] - ur[0]) < 1e-6


def test_energy_is_conserved() -> None:
    # Energy density = rho u^2 / 2 + f^2 / (2 rho c^2); constant in time for
    # the lossless problem, including after wrap-around on the periodic domain.
    grid = periodic_grid(4000)
    rho, c = MEDIUM.rho_at(grid.x), MEDIUM.c_at(grid.x)
    energies = []
    for t in (0.0, 0.6, 1.1, 2.3):
        u, f = exact_solution(grid.x, t, MEDIUM)
        energies.append(np.sum(0.5 * rho * u**2 + 0.5 * f**2 / (rho * c**2)) * grid.h)
    np.testing.assert_allclose(energies, energies[0], rtol=1e-6)


def test_uniform_medium_is_pure_translation() -> None:
    uniform = LayeredMedium(layer=Material(c=1.0, rho=1.0))
    grid = periodic_grid(400)
    _, f = exact_solution(grid.x, 0.3, uniform)
    _, f0 = right_going_pulse(grid.x - 0.3, uniform)
    np.testing.assert_allclose(f, f0, atol=1e-12)


@pytest.mark.parametrize("center", [-0.9, 0.2, 0.7])
def test_non_default_pulse_center(center: float) -> None:
    # Regression: the ray-pruning threshold once had the sign of ``center``
    # backwards, which silently zeroed the field for centers >= 0.
    uniform = LayeredMedium(layer=Material(c=1.0, rho=1.0), layer_start=-0.6)
    t = 1.2
    peak = (center + t + 1.0) % 2.0 - 1.0  # translate, then wrap onto [-1, 1)
    _, f = exact_solution(np.array([peak]), t, uniform, center=center)
    np.testing.assert_allclose(f, [1.0], atol=1e-12)
    # And through the layer: energy still conserved for a pulse starting right
    # of the layer and wrapping around into it.
    grid = periodic_grid(4000)
    rho, c = MEDIUM.rho_at(grid.x), MEDIUM.c_at(grid.x)
    energies = []
    for t in (0.0, 0.9, 1.7):
        u, f = exact_solution(grid.x, t, MEDIUM, center=0.7)
        energies.append(np.sum(0.5 * rho * u**2 + 0.5 * f**2 / (rho * c**2)) * grid.h)
    np.testing.assert_allclose(energies, energies[0], rtol=1e-6)


def test_pulse_must_start_outside_layer() -> None:
    with pytest.raises(ValueError):
        exact_solution(np.zeros(3), 0.1, MEDIUM, center=0.25)


@pytest.mark.parametrize("c_bg", [0.5, 2.0])
def test_background_speed_other_than_one(c_bg: float) -> None:
    # Regression: the ray phases are travel times, and the pulse centre and
    # width were once applied in those units directly, which is only right
    # for c = 1. Uniform medium: the solution is the translated initial pulse.
    uniform = LayeredMedium(
        background=Material(c=c_bg, rho=1.0), layer=Material(c=c_bg, rho=1.0)
    )
    grid = periodic_grid(400)
    u0, f0 = right_going_pulse(grid.x, uniform)
    u, f = exact_solution(grid.x, 0.0, uniform)
    np.testing.assert_allclose(f, f0, atol=1e-14)
    np.testing.assert_allclose(u, u0, atol=1e-14)
    t = 0.3
    _, f_t = exact_solution(grid.x, t, uniform)
    _, f_shift = right_going_pulse(grid.x - c_bg * t, uniform)
    np.testing.assert_allclose(f_t, f_shift, atol=1e-12)
    # And with a contrast: energy still conserved through the layer.
    medium = LayeredMedium(
        background=Material(c=c_bg, rho=1.0), layer=Material(c=2 * c_bg, rho=0.5)
    )
    grid = periodic_grid(4000)
    rho, c = medium.rho_at(grid.x), medium.c_at(grid.x)
    energies = []
    for t in (0.0, 0.4, 0.9):
        u, f = exact_solution(grid.x, t, medium)
        energies.append(np.sum(0.5 * rho * u**2 + 0.5 * f**2 / (rho * c**2)) * grid.h)
    np.testing.assert_allclose(energies, energies[0], rtol=1e-6)


def test_initial_pulse_near_the_seam_is_periodic() -> None:
    # A pulse whose tail wraps through x = -1 -> 1: the exact solution at
    # t = 0 must equal the (periodic) initial condition on both sides of the
    # seam, which needs the pre-image ray in the right region.
    uniform = LayeredMedium(layer=Material(c=1.0, rho=1.0), layer_start=-0.6)
    grid = periodic_grid(400)
    u0, f0 = right_going_pulse(grid.x, uniform, center=-0.9)
    u, f = exact_solution(grid.x, 0.0, uniform, center=-0.9)
    assert f0[grid.x > 0.9].max() > 1e-3  # the wrapped tail is not negligible
    np.testing.assert_allclose(f, f0, atol=1e-14)
    np.testing.assert_allclose(u, u0, atol=1e-14)
    # And it then translates as one periodic pulse.
    _, f_t = exact_solution(grid.x, 0.35, uniform, center=-0.9)
    _, f_shift = right_going_pulse(grid.x - 0.35, uniform, center=-0.9)
    np.testing.assert_allclose(f_t, f_shift, atol=1e-12)


def test_right_region_start_near_the_seam_and_final_snapshot() -> None:
    # A pulse starting right of the layer with its tail wrapping through
    # x = 1 -> -1: the main ray's seam child carries that tail.
    uniform = LayeredMedium(layer=Material(c=1.0, rho=1.0), layer_start=-0.6)
    grid = periodic_grid(400)
    u0, f0 = right_going_pulse(grid.x, uniform, center=0.9)
    u, f = exact_solution(grid.x, 0.0, uniform, center=0.9)
    assert f0[grid.x < -0.9].max() > 1e-3
    np.testing.assert_allclose(f, f0, atol=1e-14)
    np.testing.assert_allclose(u, u0, atol=1e-14)
    _, f_t = exact_solution(grid.x, 0.35, uniform, center=0.9)
    _, f_shift = right_going_pulse(grid.x - 0.35, uniform, center=0.9)
    np.testing.assert_allclose(f_t, f_shift, atol=1e-12)


def test_pulse_tail_across_an_interface_is_refused() -> None:
    # Centre 0.05 outside the layer with a wide pulse: the tail inside the
    # layer has no ray to carry it, so the reference refuses instead of
    # returning zeros there.
    with pytest.raises(ValueError, match="tail"):
        exact_solution(np.zeros(3), 0.0, MEDIUM, center=-0.05, sharpness=50.0)
    # Identical materials: no contrast, tails may cross freely.
    uniform = LayeredMedium(layer=Material(c=1.0, rho=1.0))
    exact_solution(np.zeros(3), 0.0, uniform, center=-0.05, sharpness=50.0)
