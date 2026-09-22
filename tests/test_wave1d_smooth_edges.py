"""Smooth-edged layer (issue demo#29): a tanh edge of width ``edge_width``."""

import numpy as np
import pytest

from rbf_hyperbolic_interfaces.wave1d import (
    LayeredMedium,
    Material,
    exact_solution,
    periodic_grid,
)
from rbf_hyperbolic_interfaces.wave1d.operators import _crossed_interfaces, _stencil


def test_zero_edge_width_is_the_jump_medium_bit_for_bit() -> None:
    x = periodic_grid(317).x
    jump = LayeredMedium(layer=Material(c=1.3, rho=0.7))
    smooth0 = LayeredMedium(layer=Material(c=1.3, rho=0.7), edge_width=0.0)
    assert not smooth0.is_smooth
    assert np.array_equal(jump.c_at(x), smooth0.c_at(x))
    assert np.array_equal(jump.rho_at(x), smooth0.rho_at(x))
    assert np.array_equal(jump.layer_fraction(x), jump.in_layer(x).astype(float))


def test_profile_is_periodic_and_symmetric_about_the_layer_centre() -> None:
    medium = LayeredMedium(edge_width=0.05)
    x = np.linspace(-1, 1, 2001)
    s = medium.layer_fraction(x)
    np.testing.assert_allclose(s, medium.layer_fraction(x + 2.0), atol=1e-15)
    np.testing.assert_allclose(s, medium.layer_fraction(x - 4.0), atol=1e-15)
    centre = 0.5 * (medium.layer_start + medium.layer_end)
    d = np.linspace(0, 0.9, 500)
    np.testing.assert_allclose(
        medium.layer_fraction(centre + d), medium.layer_fraction(centre - d), atol=1e-15
    )
    assert np.all(s >= 0) and np.all(s <= 1)


def test_edges_sit_at_the_midpoint_value_and_far_field_is_exact() -> None:
    medium = LayeredMedium(layer=Material(c=2.0, rho=1.5), edge_width=0.01)
    for xi in (medium.layer_start, medium.layer_end):
        assert medium.c_at(xi) == pytest.approx(1.5, abs=1e-12)
        assert medium.rho_at(xi) == pytest.approx(1.25, abs=1e-12)
    # Far from both edges the tanh tails round away completely.
    assert medium.c_at(-0.5) == 1.0
    assert medium.rho_at(0.25) == 1.5
    assert medium.c_max == 2.0


def test_smooth_edge_converges_to_the_jump_as_width_shrinks() -> None:
    grid = periodic_grid(400)
    jump = LayeredMedium()
    for width in (1e-2, 1e-3, 1e-4):
        smooth = LayeredMedium(edge_width=width)
        err = np.abs(smooth.c_at(grid.x) - jump.c_at(grid.x)).max()
        # Cell-centred nodes sit h/2 from each edge at the closest, where the
        # tanh has only reached tanh(h / (2 width)) of its full swing.
        expected = 0.5 * (1 - np.tanh(grid.h / (2 * width)))
        assert err == pytest.approx(expected, rel=1e-6, abs=1e-15), width
    assert 0.5 * (1 - np.tanh(grid.h / 2e-4)) < 1e-10


def test_ray_sum_refuses_smooth_edges() -> None:
    with pytest.raises(ValueError, match="smooth edges"):
        exact_solution(periodic_grid(50).x, 0.5, LayeredMedium(edge_width=0.01))


def test_negative_edge_width_rejected() -> None:
    with pytest.raises(ValueError):
        LayeredMedium(edge_width=-0.1)


def test_varies_over_matches_interface_crossing_for_a_jump() -> None:
    grid = periodic_grid(200)
    medium = LayeredMedium()
    for i in range(grid.n):
        _, xs = _stencil(grid, i, 2)
        assert medium.varies_over(xs) == bool(_crossed_interfaces(xs, medium))


def test_varies_over_reaches_about_19_edge_widths_for_a_smooth_edge() -> None:
    medium = LayeredMedium(edge_width=0.01)
    assert medium.varies_over(np.array([-0.02, -0.01, 0.0, 0.01, 0.02]))
    assert medium.varies_over(np.array([-0.18, -0.17]))
    assert not medium.varies_over(np.array([-0.30, -0.29]))
    assert not medium.varies_over(np.array([-0.7, -0.6, -0.5]))
