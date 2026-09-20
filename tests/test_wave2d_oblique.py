"""Oblique incidence on flat smooth edges (#41): the plane-wave train and the
Fourier-in-x reference."""

import numpy as np
import pytest

from pdes_demo.wave2d import (
    LayeredMedium2D,
    make_node_set,
    oblique_p_wave,
    plane_p_wave,
    run,
    run_fourier,
    spectral_plane_wave,
)

FLAT = LayeredMedium2D()
UNIFORM = LayeredMedium2D(layer=FLAT.background, edge_width=0.01)
PULSE = dict(center=0.875, sharpness=15.0)


@pytest.fixture(scope="module")
def nodes():
    return make_node_set(FLAT, 400)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _initial(medium: LayeredMedium2D, direction):
    return lambda xy: oblique_p_wave(xy, medium, direction, **PULSE)


# --- the pulse ---------------------------------------------------------------------


def test_normal_incidence_member_is_the_plane_pulse(nodes) -> None:
    for medium in (FLAT, LayeredMedium2D(edge_width=0.01)):
        a = plane_p_wave(nodes, medium, **PULSE)
        b = oblique_p_wave(nodes, medium, (0, 1), **PULSE)
        assert np.abs(a - b).max() < 1e-13
    with pytest.raises(ValueError):
        oblique_p_wave(nodes, FLAT, (1, 0))
    with pytest.raises(ValueError):
        run(nodes, FLAT, t_end=0.01, initial=np.zeros((5, 3)))


def test_train_is_the_exact_translate_in_a_uniform_medium(nodes) -> None:
    # The solver sees d/dx through the mode number and d/dy through the FFT;
    # the train's eigenvector is right only if both leave it a pure translate.
    # Its curl is zero at t = 0: a P wave.
    initial = _initial(UNIFORM, (1, 2))
    states = run_fourier(UNIFORM, initial, 0.3, snapshot_times=[0.0, 0.3], n_y=256)
    assert list(states[-1].modes) == list(range(13))
    exact = oblique_p_wave(nodes, UNIFORM, (1, 2), t=0.3, **PULSE)
    assert np.abs(states[-1].evaluate(nodes.xy) - exact).max() < 1e-10
    assert np.abs(states[0].curl(nodes.xy)).max() < 1e-11


# --- the Fourier-in-x reference ---------------------------------------------------


def test_fourier_reference_matches_the_1d_reference_at_normal_incidence(
    nodes,
) -> None:
    # Independent code paths (period-2 mapped grid against a period-1 grid in
    # y, complex mode arrays against real 1-D arrays) at delta = 0.01.
    medium = LayeredMedium2D(edge_width=0.01)
    state = run_fourier(medium, _initial(medium, (0, 1)), 0.3)[-1]
    assert list(state.modes) == [0]
    ref = spectral_plane_wave(nodes, 0.3, medium, **PULSE)
    got = state.evaluate(nodes.xy)
    assert np.abs(got - ref).max() < 1e-8
    assert np.abs(ref[1]).max() > 0.5  # the pulse is there


def test_fourier_reference_conserves_energy_and_self_converges(nodes) -> None:
    # Oblique through the edge (mode conversion included): the semi-discrete
    # energy is exact, so the step and grid set the accuracy; doubling the
    # grid and halving the step moves the answer by under 1e-7.
    medium = LayeredMedium2D(edge_width=0.02)
    initial = _initial(medium, (1, 2))
    coarse = run_fourier(medium, initial, 0.2, snapshot_times=[0.0, 0.2], n_y=512)
    fine = run_fourier(medium, initial, 0.2, n_y=1024, dt=2.5e-5)[-1]
    e0, e1 = (s.energy(medium) for s in coarse)
    assert abs(e1 / e0 - 1) < 1e-10
    a, b = coarse[-1].evaluate(nodes.xy), fine.evaluate(nodes.xy)
    assert np.abs(a - b).max() / np.abs(b).max() < 1e-7
    # S waves exist only because of the edge.
    assert np.abs(coarse[-1].curl(nodes.xy)).max() > 1.0
