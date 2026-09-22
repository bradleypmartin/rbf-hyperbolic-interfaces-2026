import math

import numpy as np
import pytest

from pdes_demo.wave2d import (
    ElasticMaterial,
    LayeredMedium2D,
    SineInterface,
    make_node_set,
    nearest_spacing,
    periodic_knn,
    plane_p_wave,
    straddling_rows,
)

FLAT = LayeredMedium2D()
CURVED = LayeredMedium2D(
    lower=SineInterface(0.25, 0.02), upper=SineInterface(0.5, 0.02)
)


def test_material_wave_speeds_and_impedance() -> None:
    bg, ly = FLAT.background, FLAT.layer
    assert bg.c_p == pytest.approx(math.sqrt(3))
    assert ly.c_p == pytest.approx(math.sqrt(6))
    assert bg.c_s == pytest.approx(1.0)
    # Reflection coefficient of dissertation eq. 49 for the default contrast,
    # Z: sqrt(3) -> 2 sqrt(6), evaluated by hand.
    z1, z2 = bg.p_impedance, ly.p_impedance
    assert (z1 - z2) / (z1 + z2) == pytest.approx(-0.47759, abs=1e-5)
    assert FLAT.c_max == pytest.approx(math.sqrt(6))


def test_medium_lookup_and_layer_membership() -> None:
    x = np.array([0.1, 0.1, 0.1, 0.3])
    y = np.array([0.1, 0.3, 0.6, 0.25])  # below, inside, above, on lower edge
    np.testing.assert_array_equal(FLAT.in_layer(x, y), [False, True, False, True])
    np.testing.assert_allclose(FLAT.rho_at(x, y), [1, 2, 1, 2])
    np.testing.assert_allclose(FLAT.lam_at(x, y), [1, 4, 1, 4])
    # Curved: membership follows the sine, not the mean height.
    x, y = np.array([0.25, 0.25]), np.array([0.26, 0.275])  # lower edge is at 0.27
    np.testing.assert_array_equal(CURVED.in_layer(x, y), [False, True])
    assert FLAT.is_flat and not CURVED.is_flat


def test_rejects_crossing_interfaces() -> None:
    with pytest.raises(ValueError):
        LayeredMedium2D(lower=SineInterface(0.5), upper=SineInterface(0.25))
    with pytest.raises(ValueError):
        LayeredMedium2D(lower=SineInterface(0.4, 0.1), upper=SineInterface(0.5, 0.1))


def test_interface_normal_is_unit_and_orthogonal_to_tangent() -> None:
    ifc = SineInterface(0.5, 0.02)
    x = np.linspace(0, 1, 17)
    nrm = ifc.normal(x)
    np.testing.assert_allclose(np.linalg.norm(nrm, axis=1), 1.0)
    tangent = np.stack([np.ones_like(x), ifc.slope(x)], axis=-1)
    np.testing.assert_allclose(np.sum(nrm * tangent, axis=1), 0.0, atol=1e-14)
    assert np.all(nrm[:, 1] > 0)


@pytest.mark.parametrize("amplitude", [0.0, 0.02])
def test_straddling_rows_geometry(amplitude: float) -> None:
    ifc = SineInterface(0.5, amplitude)
    h = 0.02
    rows = straddling_rows(ifc, 50, h, rows_per_side=3)
    assert rows.shape == (300, 2)
    # Every fixed node lies at one of the three hex offsets from the curve,
    # measured along the normal at its own foot point.
    foot_x = np.arange(50) / 50
    expected = np.array([0.5, 0.5 + math.sqrt(3) / 2, 0.5 + math.sqrt(3)]) * h
    for j, off in enumerate(expected):
        xs = foot_x + (j % 2) * 0.5 / 50
        base = np.stack([xs, ifc.height(xs)], axis=-1)
        nrm = ifc.normal(xs)
        below = rows[2 * j * 50 : (2 * j + 1) * 50]
        above = rows[(2 * j + 1) * 50 : (2 * j + 2) * 50]
        np.testing.assert_allclose(below, np.mod(base - off * nrm, 1.0), atol=1e-12)
        np.testing.assert_allclose(above, np.mod(base + off * nrm, 1.0), atol=1e-12)


@pytest.mark.parametrize("medium", [FLAT, CURVED], ids=["flat", "curved"])
def test_node_set_count_bounds_and_fixed_rows(medium: LayeredMedium2D) -> None:
    nodes = make_node_set(medium, 2500, repulsion_steps=30)
    assert nodes.n == 2500 and nodes.h == pytest.approx(0.02)
    assert np.all((nodes.xy >= 0) & (nodes.xy < 1))
    assert nodes.fixed.sum() == 2 * 6 * 50
    expected = np.concatenate(
        [straddling_rows(ifc, 50, nodes.h) for ifc in medium.interfaces]
    )
    np.testing.assert_allclose(nodes.xy[nodes.fixed], expected, atol=1e-12)


def test_node_set_is_reproducible_and_seed_dependent() -> None:
    a = make_node_set(FLAT, 900, repulsion_steps=20, seed=1)
    b = make_node_set(FLAT, 900, repulsion_steps=20, seed=1)
    c = make_node_set(FLAT, 900, repulsion_steps=20, seed=2)
    np.testing.assert_array_equal(a.xy, b.xy)
    assert not np.allclose(a.xy[~a.fixed], c.xy[~c.fixed])


def test_relaxed_nodes_are_quasi_uniform_and_stay_out_of_the_band() -> None:
    nodes = make_node_set(CURVED, 2500)
    d = nearest_spacing(nodes) / nodes.h
    assert d.min() > 0.8, d.min()
    assert abs(d[~nodes.fixed].mean() - 1.0) < 0.05
    band = (0.5 + math.sqrt(3)) * nodes.h
    dist = CURVED.distance_to_interfaces(nodes.x, nodes.y)
    assert np.all(dist[~nodes.fixed] > band)
    # Stencils around the interface rows draw on both sides of the curve.
    idx, _ = periodic_knn(nodes.xy, 30)
    offset = CURVED.lower.vertical_offset(nodes.x, nodes.y)
    side = np.sign(offset)
    lower_rows = np.flatnonzero(nodes.fixed & (np.abs(offset) < band + 1e-9))
    both_sides = [len(set(side[idx[i]])) == 2 for i in lower_rows]
    assert all(both_sides)


def test_rejects_non_square_and_too_small_n() -> None:
    with pytest.raises(ValueError):
        make_node_set(FLAT, 1000)
    with pytest.raises(ValueError):
        make_node_set(FLAT, 25)  # 5 x 5 grid cannot hold 60 fixed nodes


def test_plane_p_wave_amplitudes_follow_the_start_material() -> None:
    nodes = make_node_set(FLAT, 400, repulsion_steps=5)
    state = plane_p_wave(nodes, FLAT)
    u, v, f, g, h = state
    assert np.all(u == 0) and np.all(g == 0)
    np.testing.assert_allclose(h, math.sqrt(3) * v)
    np.testing.assert_allclose(f, v / math.sqrt(3))
    assert v.max() > 0.99 and v[np.argmin(np.abs(nodes.y - 0.25))] < 1e-30
    # A different start material changes the stress ratios accordingly.
    stiff = LayeredMedium2D(background=ElasticMaterial(lam=2.0, mu=1.0, rho=4.0))
    s2 = plane_p_wave(nodes, stiff)
    np.testing.assert_allclose(s2[4], 4 * math.sqrt(1.0) * s2[1])
    np.testing.assert_allclose(s2[2], 0.5 * s2[4])
    with pytest.raises(ValueError):
        plane_p_wave(nodes, FLAT, center=0.3)
    # Curved interfaces: the guard covers the whole sinusoid, not just x = 0.
    with pytest.raises(ValueError):
        plane_p_wave(nodes, CURVED, center=0.51)
    with pytest.raises(ValueError):
        plane_p_wave(nodes, CURVED, center=0.24)
    plane_p_wave(nodes, CURVED, center=0.53)  # clear of the crests
