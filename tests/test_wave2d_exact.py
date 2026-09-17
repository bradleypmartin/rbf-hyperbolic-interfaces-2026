import math

import numpy as np
import pytest

from pdes_demo.wave2d import (
    ElasticMaterial,
    LayeredMedium2D,
    SineInterface,
    exact_plane_wave,
    make_node_set,
    plane_p_wave,
)

FLAT = LayeredMedium2D()
Z1, Z2 = FLAT.background.p_impedance, FLAT.layer.p_impedance
R = (Z2 - Z1) / (Z1 + Z2)  # reflection coefficient for stress, eq. 49 up to sign
T_IN = 2 * Z2 / (Z1 + Z2)
T_OUT = 2 * Z1 / (Z1 + Z2)


def _column(x: float, ys: np.ndarray) -> np.ndarray:
    return np.stack([np.full_like(ys, x), ys], axis=-1)


def test_matches_initial_state() -> None:
    nodes = make_node_set(FLAT, 400, repulsion_steps=5)
    np.testing.assert_allclose(
        exact_plane_wave(nodes, 0.0, FLAT), plane_p_wave(nodes, FLAT), atol=1e-13
    )


@pytest.mark.parametrize("t", [0.1, 0.2, 0.3])
def test_velocity_and_traction_continuous_stress_f_not(t: float) -> None:
    eps = 1e-9
    for y_i in (FLAT.lower.y0, FLAT.upper.y0):
        below = exact_plane_wave(_column(0.3, np.array([y_i - eps])), t, FLAT)
        above = exact_plane_wave(_column(0.3, np.array([y_i + eps])), t, FLAT)
        assert abs(below[1, 0] - above[1, 0]) < 1e-6  # v
        assert abs(below[4, 0] - above[4, 0]) < 1e-6  # h (traction)
        # f = lam / (lam + 2 mu) h has the same ratio on both sides here, so
        # it happens to be continuous too; u and g vanish identically.
    ys = np.linspace(0, 1, 401)
    state = exact_plane_wave(_column(0.7, ys), t, FLAT)
    assert np.all(state[0] == 0) and np.all(state[3] == 0)
    np.testing.assert_allclose(state[2], state[4] / 3, atol=1e-14)


def test_reflected_and_transmitted_amplitudes_at_t_0_3() -> None:
    # Incident v-pulse (amplitude 1) hits y = 0.5 at t = 0.25 / sqrt(3).
    # By t = 0.3 the reflected pulse has travelled back up, the transmitted
    # one has crossed the band at sqrt(6) and left through y = 0.25.
    t = 0.3
    c1, c2 = FLAT.background.c_p, FLAT.layer.c_p
    t_hit = 0.25 / c1
    y_refl = 0.5 + c1 * (t - t_hit)
    y_trans = 0.25 - c1 * (t - t_hit - 0.25 / c2)
    ys = np.linspace(0, 1, 4001)
    v = exact_plane_wave(_column(0.5, ys), t, FLAT)[1]
    i_refl = np.argmin(np.abs(ys - y_refl))
    i_trans = np.argmin(np.abs(ys - y_trans))
    # Reflected velocity flips sign relative to stress; transmitted keeps it.
    assert v[i_refl] == pytest.approx(-R, abs=2e-3)
    assert v[i_trans] == pytest.approx(T_IN * T_OUT, abs=2e-3)
    assert R == pytest.approx(0.47759, abs=1e-5)
    assert T_IN * T_OUT == pytest.approx(0.77190, abs=1e-5)


def test_non_default_interfaces_and_center() -> None:
    medium = LayeredMedium2D(lower=SineInterface(0.3), upper=SineInterface(0.6))
    nodes = make_node_set(medium, 400, repulsion_steps=5)
    center = 0.85
    np.testing.assert_allclose(
        exact_plane_wave(nodes, 0.0, medium, center=center),
        plane_p_wave(nodes, medium, center=center),
        atol=1e-13,
    )
    eps = 1e-9
    for y_i in (0.3, 0.6):
        lo = _column(0.1, np.array([y_i - eps]))
        hi = _column(0.1, np.array([y_i + eps]))
        below = exact_plane_wave(lo, 0.25, medium, center=center)
        above = exact_plane_wave(hi, 0.25, medium, center=center)
        assert abs(below[4, 0] - above[4, 0]) < 1e-6
    # The default centre (0.75) is only 0.15 from the moved upper interface:
    # its tail there is 7e-6, which the ray sum cannot carry, so it refuses.
    with pytest.raises(ValueError):
        exact_plane_wave(lo, 0.25, medium)


def test_stress_ratio_follows_the_local_material() -> None:
    # Different lam / (lam + 2 mu) on the two sides: f = ratio * h with the
    # local ratio, because f_t and h_t share v_y and both start at zero in
    # the layer. (The defaults have the same ratio on both sides, which
    # would hide a wrong choice here.)
    medium = LayeredMedium2D(layer=ElasticMaterial(lam=2.0, mu=0.5, rho=1.5))
    nodes = make_node_set(medium, 400, repulsion_steps=5)
    np.testing.assert_allclose(
        exact_plane_wave(nodes, 0.0, medium), plane_p_wave(nodes, medium), atol=1e-13
    )
    ys = np.linspace(0.01, 0.99, 197)
    state = exact_plane_wave(_column(0.4, ys), 0.3, medium)
    inside = medium.in_layer(np.full_like(ys, 0.4), ys)
    np.testing.assert_allclose(state[2][inside], 2 / 3 * state[4][inside], atol=1e-14)
    np.testing.assert_allclose(state[2][~inside], 1 / 3 * state[4][~inside], atol=1e-14)
    assert np.abs(state[4][inside]).max() > 0.1  # the wave is actually there


def test_broad_pulse_whose_tail_crosses_an_interface_is_refused() -> None:
    with pytest.raises(ValueError, match="tails"):
        exact_plane_wave(np.zeros((3, 2)), 0.1, FLAT, sharpness=3.0)


def test_rejects_curved_interfaces() -> None:
    curved = LayeredMedium2D(lower=SineInterface(0.25, 0.02))
    with pytest.raises(ValueError):
        exact_plane_wave(np.zeros((3, 2)), 0.1, curved)


def test_pulse_speed_is_c_p_of_the_background() -> None:
    uniform = LayeredMedium2D(layer=FLAT.background)
    t = 0.1
    ys = np.linspace(0, 1, 2001)
    v = exact_plane_wave(_column(0.2, ys), t, uniform)[1]
    assert ys[np.argmax(v)] == pytest.approx(0.75 - math.sqrt(3) * t, abs=1e-3)
