"""Coefficient treatments of the naive scheme's medium (issue demo#69)."""

import numpy as np
import pytest
from scipy.special import sici

from rbf_hyperbolic_interfaces.wave1d import LayeredMedium, Material, periodic_grid
from rbf_hyperbolic_interfaces.wave1d.treatments import TreatedMedium, widened

LAYER = Material(c=2.0, rho=1.5)


def compliance(medium, x):
    return 1.0 / (medium.rho_at(x) * medium.c_at(x) ** 2)


def test_widened_edge_is_never_narrower_than_the_cells_asked_for() -> None:
    medium = LayeredMedium(edge_width=0.0025)
    assert widened(medium, 0.02, 1).edge_width == 0.02
    assert widened(medium, 0.02, 2).edge_width == 0.04
    # A resolved edge is left alone.
    assert widened(medium, 0.001, 2).edge_width == 0.0025
    assert widened(medium, 0.02, 1).layer == medium.layer


def test_box_kernel_on_a_jump_is_tornberg_engquist_eq_18_and_22() -> None:
    """1/a and 1/b linear across one cell centred on the jump (MAA 2006, §4.1):
    compliance and density both ramp linearly over [x - h/2, x + h/2]."""
    medium = LayeredMedium(layer=LAYER)
    h = 0.02
    treated = TreatedMedium(medium, h, kernel="box")
    x = medium.layer_start + h * np.linspace(-0.5, 0.5, 11)
    ramp = (x - medium.layer_start + h / 2) / h
    comp_l, comp_r = 1.0, 1.0 / (LAYER.rho * LAYER.c**2)
    np.testing.assert_allclose(
        compliance(treated, x), comp_l + ramp * (comp_r - comp_l), rtol=1e-13
    )
    np.testing.assert_allclose(treated.rho_at(x), 1.0 + ramp * 0.5, rtol=1e-13)
    # Outside the cell the medium is untouched, on both sides.
    assert treated.c_at(medium.layer_start - 0.51 * h) == pytest.approx(1.0, abs=1e-14)
    assert treated.c_at(medium.layer_start + 0.51 * h) == pytest.approx(2.0, abs=1e-14)


@pytest.mark.parametrize("width_over_h", [1 / 8, 1 / 2, 2.0])
def test_box_kernel_on_a_tanh_edge_matches_the_closed_form(width_over_h) -> None:
    """The arithmetic cell mean of rho has the antiderivative delta ln cosh."""
    h = 0.02
    delta = width_over_h * h
    medium = LayeredMedium(layer=LAYER, edge_width=delta)
    treated = TreatedMedium(medium, h, kernel="box")
    x = medium.layer_start + h * np.linspace(-2, 2, 17)

    def log_cosh(t):
        return np.logaddexp(t, -t) - np.log(2.0)

    def primitive(y):
        # Both edges' m = 0 images; the periodic images are 1.5 away, where
        # the tails are below 1e-30 at these widths.
        return (
            0.5
            * delta
            * (
                log_cosh((y - medium.layer_start) / delta)
                - log_cosh((y - medium.layer_end) / delta)
            )
        )

    mean_s = (primitive(x + h / 2) - primitive(x - h / 2)) / h
    np.testing.assert_allclose(treated.rho_at(x), 1.0 + 0.5 * mean_s, rtol=1e-12)


@pytest.mark.parametrize("kernel", ["box", "sinc"])
def test_far_from_the_edges_the_treated_medium_is_the_true_one(kernel) -> None:
    medium = LayeredMedium(layer=LAYER, edge_width=0.0025)
    treated = TreatedMedium(medium, 0.02, kernel=kernel)
    x = np.array([-0.6, -0.5, 0.25, 0.8])
    np.testing.assert_allclose(treated.c_at(x), medium.c_at(x), rtol=1e-14)
    np.testing.assert_allclose(treated.rho_at(x), medium.rho_at(x), rtol=1e-14)
    np.testing.assert_allclose(
        treated.impedance_at(x), medium.impedance_at(x), rtol=1e-14
    )


def test_sinc_kernel_is_symmetric_and_normalised() -> None:
    """A symmetric normalised kernel maps a jump to a profile odd about the
    edge: the two half-jumps of compliance sum to the two far-field values."""
    medium = LayeredMedium(layer=LAYER)
    treated = TreatedMedium(medium, 0.02, kernel="sinc")
    z = 0.02 * np.linspace(0.05, 2.4, 20)
    left = compliance(treated, medium.layer_start - z)
    right = compliance(treated, medium.layer_start + z)
    np.testing.assert_allclose(left + right, 1.0 + compliance(medium, 0.25), rtol=1e-12)


def test_sinc_kernel_with_a_long_window_is_the_anti_aliased_step() -> None:
    """Koene et al. (2022) eq. 22: the step band-limited to the grid Nyquist is
    1/2 + Si(pi z / h) / pi. With the window pushed out to 20 cells (short of
    the far edge) and the cutoff at the Nyquist, the treated compliance
    matches it near the edge to the window's truncation of the sinc tail."""
    medium = LayeredMedium(layer=LAYER)
    h = 0.02
    treated = TreatedMedium(medium, h, kernel="sinc", cutoff=1.0, half_width=20.0)
    z = h * np.linspace(-3, 3, 25)
    step = 0.5 + sici(np.pi * z / h)[0] / np.pi
    comp_l, comp_r = 1.0, compliance(medium, 0.25)
    expected = comp_l + step * (comp_r - comp_l)
    np.testing.assert_allclose(
        compliance(treated, medium.layer_start + z), expected, atol=3e-3
    )
    # Gibbs: the band-limited compliance overshoots both far-field values.
    fine = medium.layer_start + h * np.linspace(-4, 4, 801)
    comp = compliance(treated, fine)
    assert comp.max() > comp_l + 0.05 * (comp_l - comp_r)
    assert comp.min() < comp_r - 0.05 * (comp_l - comp_r)


def test_c_max_sees_the_gibbs_overshoot_of_the_sinc_kernel_only() -> None:
    medium = LayeredMedium()
    assert TreatedMedium(medium, 0.02, kernel="box").c_max == pytest.approx(2.0)
    assert TreatedMedium(medium, 0.02, kernel="sinc").c_max > 2.02


def test_treated_medium_runs_through_the_naive_scheme() -> None:
    """The wrapper satisfies Medium1D: ``run(mode="naive")`` takes it as is."""
    from rbf_hyperbolic_interfaces.wave1d import run

    medium = LayeredMedium(edge_width=0.0025)
    grid = periodic_grid(100)
    treated = TreatedMedium(medium, grid.h, kernel="box")
    snaps = run(grid, treated, mode="naive", t_end=0.1, n_snapshots=1)
    assert np.all(np.isfinite(snaps.f[-1]))
    assert snaps.f.shape == (2, 100)


def test_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="positive"):
        TreatedMedium(LayeredMedium(), 0.0)
    with pytest.raises(ValueError, match="kernel"):
        TreatedMedium(LayeredMedium(), 0.02, kernel="gauss")  # type: ignore[arg-type]


# --- what the comparators do to the standard scheme (notes §2.1) ---------------


def _error(n: int, medium: LayeredMedium, run_medium, t_end: float = 1.0) -> float:
    """Relative l2 error in f at ``t_end`` of the naive scheme on
    ``run_medium`` against the true medium's reference (the notes' setup:
    sharpness 60, centre -0.6)."""
    from rbf_hyperbolic_interfaces.wave1d import exact_solution, run
    from rbf_hyperbolic_interfaces.wave1d.spectral import (
        interpolate,
        reference_size,
        run_spectral,
    )

    grid = periodic_grid(n)
    snaps = run(
        grid,
        run_medium,
        mode="naive",
        t_end=t_end,
        n_snapshots=1,
        pulse_center=-0.6,
        pulse_sharpness=60.0,
    )
    if medium.is_smooth:
        ref_grid, ref = run_spectral(
            medium,
            reference_size(medium.edge_width),
            t_end=t_end,
            dt=2e-4,
            pulse_center=-0.6,
            pulse_sharpness=60.0,
        )
        f_ref = interpolate(ref.f[-1], ref_grid, grid.x)
    else:
        _, f_ref = exact_solution(grid.x, t_end, medium, -0.6, 60.0)
    return float(np.linalg.norm(snaps.f[-1] - f_ref) / np.linalg.norm(f_ref))


def test_one_cell_average_of_a_jump_on_a_cell_boundary_changes_nothing() -> None:
    """The layer's edges sit midway between cell-centred nodes, so the
    one-cell ramp of Tornberg & Engquist ends exactly at the nearest nodes
    and the sampled coefficients are the true ones to rounding, so the runs
    agree to rounding. The two-cell average does reach those nodes."""
    from rbf_hyperbolic_interfaces.wave1d import run

    medium = LayeredMedium()
    grid = periodic_grid(200)
    cell = TreatedMedium(medium, grid.h)
    np.testing.assert_allclose(cell.c_at(grid.x), medium.c_at(grid.x), rtol=1e-13)
    np.testing.assert_allclose(cell.rho_at(grid.x), medium.rho_at(grid.x), rtol=1e-13)
    kw = dict(mode="naive", t_end=0.5, n_snapshots=1, pulse_center=-0.6)
    plain = run(grid, medium, pulse_sharpness=60.0, **kw)
    same = run(grid, cell, pulse_sharpness=60.0, **kw)
    wider = run(grid, TreatedMedium(medium, 2 * grid.h), pulse_sharpness=60.0, **kw)
    np.testing.assert_allclose(same.f[-1], plain.f[-1], rtol=1e-10, atol=1e-13)
    assert not np.allclose(wider.f[-1], plain.f[-1], rtol=1e-3, atol=1e-6)


def test_two_cell_average_is_second_order_at_a_jump_and_the_others_first() -> None:
    """Tornberg & Engquist's order for their Yee scheme, on collocated FD4 with
    the average widened to reach the nodes beside the jump; the sampled
    medium and the band-limited one stay first order there."""
    medium = LayeredMedium()
    ns = [100, 200, 400]
    errs = {
        label: [_error(n, medium, make(periodic_grid(n).h), 0.6) for n in ns]
        for label, make in {
            "naive": lambda h: medium,
            "cell2": lambda h: TreatedMedium(medium, 2 * h),
            "bandlimit": lambda h: TreatedMedium(medium, h, kernel="sinc"),
        }.items()
    }
    rate = {k: np.log2(v[1] / v[2]) for k, v in errs.items()}
    assert 1.8 < rate["cell2"] < 2.3
    assert 0.8 < rate["naive"] < 1.3 and 0.8 < rate["bandlimit"] < 1.3
    assert errs["cell2"][2] < 0.1 * errs["naive"][2]


def test_treatments_gain_little_through_an_unresolved_edge() -> None:
    """At h = 4 delta the cell and band-limited media cut the naive error by
    a factor between 1.2 and 3 while the seeds cut it by more than 50."""
    from rbf_hyperbolic_interfaces.wave1d import run

    medium = LayeredMedium(edge_width=0.0025)
    n = 200
    h = periodic_grid(n).h
    naive = _error(n, medium, medium, 0.6)
    cell = _error(n, medium, TreatedMedium(medium, h), 0.6)
    band = _error(n, medium, TreatedMedium(medium, h, kernel="sinc"), 0.6)
    for treated in (cell, band):
        assert 1.2 < naive / treated < 3.0
    grid = periodic_grid(n)
    seeds = run(
        grid,
        medium,
        mode="aware",
        t_end=0.6,
        n_snapshots=1,
        pulse_center=-0.6,
        pulse_sharpness=60.0,
    )
    ref_err = _error(n, medium, medium, 0.6)  # same reference path, naive
    assert ref_err == naive
    # The seeds against the same reference:
    from rbf_hyperbolic_interfaces.wave1d.spectral import (
        interpolate,
        reference_size,
        run_spectral,
    )

    ref_grid, ref = run_spectral(
        medium,
        reference_size(medium.edge_width),
        t_end=0.6,
        dt=2e-4,
        pulse_center=-0.6,
        pulse_sharpness=60.0,
    )
    f_ref = interpolate(ref.f[-1], ref_grid, grid.x)
    seed_err = float(np.linalg.norm(seeds.f[-1] - f_ref) / np.linalg.norm(f_ref))
    assert naive / seed_err > 50
