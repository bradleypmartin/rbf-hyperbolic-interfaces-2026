"""Smooth flat edges in 2-D, the normal-incidence spectral reference (#36) and
the seed-aware operator (#39, #40)."""

import numpy as np
import pytest

from pdes_demo.wave1d import LayeredMedium
from pdes_demo.wave1d.spectral import reference_size
from pdes_demo.wave2d import (
    ElasticMaterial,
    LayeredMedium2D,
    SineInterface,
    build_operators,
    exact_plane_wave,
    make_node_set,
    minimal_image,
    periodic_knn,
    plane_p_wave,
    run,
    spectral_plane_wave,
)
from pdes_demo.wave2d.exact import _as_1d_medium, _MappedMedium1D

FLAT = LayeredMedium2D()
UNIFORM = LayeredMedium2D(layer=FLAT.background)
# lam / (lam + 2 mu) is 1/3 on both sides of the default contrast, which would
# hide a wrong stress recovery; this band has 2/3.
ODD_RATIO = ElasticMaterial(lam=2.0, mu=0.5, rho=1.5)


def _column(x: float, ys: np.ndarray) -> np.ndarray:
    return np.stack([np.full_like(ys, x), ys], axis=-1)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# --- the medium ------------------------------------------------------------------


def test_zero_edge_width_is_the_jump_medium_bit_for_bit() -> None:
    # The pre-#36 lookup was np.where(in_layer, layer, background); the
    # blend must reproduce it exactly, not just to a tolerance.
    nodes = make_node_set(FLAT, 900, repulsion_steps=10)
    medium = LayeredMedium2D(layer=ODD_RATIO, edge_width=0.0)
    assert not medium.is_smooth
    inside = medium.in_layer(nodes.x, nodes.y)
    for attr in ("lam", "mu", "rho"):
        expected = np.where(
            inside, getattr(ODD_RATIO, attr), getattr(medium.background, attr)
        )
        got = getattr(medium, attr + "_at")(nodes.x, nodes.y)
        assert np.array_equal(got, expected)
    np.testing.assert_array_equal(
        medium.layer_fraction(nodes.x, nodes.y), inside.astype(float)
    )


def test_material_at_agrees_with_the_three_accessors() -> None:
    rng = np.random.default_rng(3)
    x, y = rng.random(200), rng.random(200) * 1.5 - 0.25  # beyond [0, 1) too
    for medium in (FLAT, LayeredMedium2D(edge_width=0.01)):
        lam, mu, rho = medium.material_at(x, y)
        np.testing.assert_array_equal(lam, medium.lam_at(x, y))
        np.testing.assert_array_equal(mu, medium.mu_at(x, y))
        np.testing.assert_array_equal(rho, medium.rho_at(x, y))


def test_profile_is_periodic_symmetric_flat_and_bounded() -> None:
    medium = LayeredMedium2D(edge_width=0.03)
    x = np.linspace(0, 1, 7)[:, None]
    y = np.linspace(-0.5, 1.5, 2001)[None, :]
    s = medium.layer_fraction(x, y)
    assert s.shape == (7, 2001)
    np.testing.assert_allclose(s, medium.layer_fraction(x, y + 1.0), atol=1e-15)
    np.testing.assert_allclose(s, medium.layer_fraction(x, y - 2.0), atol=1e-15)
    np.testing.assert_array_equal(s, np.broadcast_to(s[:1], s.shape))  # flat: no x
    centre = 0.5 * (FLAT.lower.y0 + FLAT.upper.y0)
    d = np.linspace(0, 0.45, 300)
    np.testing.assert_allclose(
        medium.layer_fraction(0.2, centre + d),
        medium.layer_fraction(0.2, centre - d),
        atol=1e-15,
    )
    assert np.all(s >= 0) and np.all(s <= 1)


def test_edges_sit_at_the_midpoint_value_and_far_field_is_exact() -> None:
    medium = LayeredMedium2D(edge_width=0.01)
    for y_i in (medium.lower.y0, medium.upper.y0):
        assert medium.lam_at(0.3, y_i) == pytest.approx(2.5, abs=1e-12)
        assert medium.mu_at(0.3, y_i) == pytest.approx(2.5, abs=1e-12)
        assert medium.rho_at(0.3, y_i) == pytest.approx(1.5, abs=1e-12)
    # Beyond about 19 delta the tanh tails round away completely; the band's
    # centre is only 12.5 delta from its edges here and still carries 3e-11.
    assert medium.lam_at(0.3, 0.9) == 1.0 and medium.rho_at(0.3, 0.05) == 1.0
    assert medium.lam_at(0.3, 0.375) == pytest.approx(4.0, abs=1e-10)
    assert medium.lam_at(0.3, 0.375) != 4.0
    narrow = LayeredMedium2D(edge_width=0.005)
    assert narrow.lam_at(0.3, 0.375) == 4.0 and narrow.rho_at(0.3, 0.375) == 2.0
    # c_p is monotone in the blend weight, so c_max is still the layer's.
    ys = np.linspace(0, 1, 1001)
    zeros = np.zeros_like(ys)
    k = medium.lam_at(zeros, ys) + 2 * medium.mu_at(zeros, ys)
    c_p = np.sqrt(k / medium.rho_at(zeros, ys))
    assert medium.c_max == FLAT.c_max
    assert c_p.max() <= medium.c_max + 1e-12
    assert c_p.min() >= FLAT.background.c_p - 1e-12


def test_smooth_edge_converges_to_the_jump_on_the_straddling_rows() -> None:
    # The innermost fixed rows sit h/2 from each edge centre, where the tanh
    # has covered tanh(h / (2 delta)) of its swing (lam jumps by 3).
    nodes = make_node_set(FLAT, 2500, repulsion_steps=10)
    lam_jump = FLAT.lam_at(nodes.x, nodes.y)
    for width in (1e-2, 1e-3, 1e-4):
        smooth = LayeredMedium2D(edge_width=width)
        err = np.abs(smooth.lam_at(nodes.x, nodes.y) - lam_jump).max()
        expected = 3.0 * 0.5 * (1 - np.tanh(nodes.h / (2 * width)))
        assert err == pytest.approx(expected, rel=1e-6, abs=1e-15), width


def test_varies_over_is_straddling_for_a_jump_and_reaches_19_widths_smooth() -> None:
    nodes = make_node_set(FLAT, 900, repulsion_steps=10)
    idx, _ = periodic_knn(nodes.xy, 19)
    for i in range(0, nodes.n, 5):
        x, y = nodes.x[idx[i]], nodes.y[idx[i]]
        straddles = len(set(FLAT.in_layer(x, y))) == 2
        assert FLAT.varies_over(x, y) == straddles
    smooth = LayeredMedium2D(edge_width=0.01)
    two = np.zeros(2)
    assert smooth.varies_over(two, np.array([0.49, 0.51]))
    assert smooth.varies_over(two, np.array([0.5 - 0.18, 0.5 - 0.17]))
    assert not smooth.varies_over(two, np.array([0.5 + 0.30, 0.5 + 0.29]))
    assert not smooth.varies_over(np.zeros(3), np.array([0.8, 0.85, 0.9]))
    assert not smooth.varies_over(np.zeros(0), np.zeros(0))
    # A tolerance hides the tails.
    assert not smooth.varies_over(two, np.array([0.5 - 0.18, 0.5 - 0.17]), rtol=1e-6)


def test_rejections_and_guards() -> None:
    with pytest.raises(ValueError):
        LayeredMedium2D(edge_width=-0.1)
    # Edges wider than a quarter of the band would merge; 0.0625 is the limit
    # for the default band, and the curved band's narrowest width counts.
    with pytest.raises(ValueError, match="merge"):
        LayeredMedium2D(edge_width=0.07)
    LayeredMedium2D(edge_width=0.0625)
    with pytest.raises(ValueError, match="merge"):
        LayeredMedium2D(
            lower=SineInterface(0.25, 0.02),
            upper=SineInterface(0.5, 0.02),
            edge_width=0.06,
        )
    with pytest.raises(NotImplementedError, match="#42"):
        LayeredMedium2D(
            lower=SineInterface(0.25, 0.02),
            upper=SineInterface(0.5, 0.02),
            edge_width=0.01,
        )
    nodes = make_node_set(FLAT, 400, repulsion_steps=5)
    smooth = LayeredMedium2D(edge_width=0.01)
    build_operators(nodes, smooth, mode="naive")
    with pytest.raises(ValueError, match="spectral_plane_wave"):
        exact_plane_wave(nodes, 0.1, smooth)
    with pytest.raises(ValueError, match="exact_plane_wave"):
        spectral_plane_wave(nodes, 0.1, FLAT)
    # A pulse centred in the band is not the background plane wave the
    # mapping assumes; refuse like plane_p_wave and the ray sum do.
    # (The band is [lower, upper) as in plane_p_wave: the lower edge centre is
    # in, the upper one is out.)
    with pytest.raises(ValueError, match="background"):
        spectral_plane_wave(nodes, 0.1, smooth, center=0.35)
    with pytest.raises(ValueError, match="background"):
        spectral_plane_wave(nodes, 0.1, smooth, center=0.25)
    spectral_plane_wave(nodes, 0.0, smooth, center=0.2)


# --- the reference ---------------------------------------------------------------


def test_mapped_medium_is_the_pointwise_image_not_a_linear_speed_blend() -> None:
    x = np.linspace(-1, 1, 2001, endpoint=False) + 1e-4  # off the boundary points
    jump = _MappedMedium1D(FLAT)
    as_layered = _as_1d_medium(FLAT)
    np.testing.assert_allclose(jump.c_at(x), as_layered.c_at(x), atol=1e-14)
    np.testing.assert_allclose(jump.rho_at(x), as_layered.rho_at(x), atol=1e-14)
    assert jump.c_max == as_layered.c_max
    smooth = LayeredMedium2D(edge_width=0.02)
    mapped = _MappedMedium1D(smooth)
    y = (1 - x) / 2
    zeros = np.zeros_like(y)
    k = smooth.lam_at(zeros, y) + 2 * smooth.mu_at(zeros, y)
    np.testing.assert_allclose(
        mapped.rho_at(x) * mapped.c_at(x) ** 2, 2 * k, rtol=1e-14
    )
    np.testing.assert_allclose(
        mapped.rho_at(x), smooth.rho_at(zeros, y) / 2, rtol=1e-14
    )
    np.testing.assert_allclose(
        mapped.impedance_at(x), mapped.rho_at(x) * mapped.c_at(x)
    )
    # A LayeredMedium(edge_width = 2 delta) has the same rho and far field but
    # blends c linearly, which differs at the edge centre.
    linear = LayeredMedium(
        background=as_layered.background,
        layer=as_layered.layer,
        layer_start=as_layered.layer_start,
        layer_width=as_layered.layer_width,
        edge_width=2 * 0.02,
    )
    np.testing.assert_allclose(linear.rho_at(x), mapped.rho_at(x), atol=1e-14)
    np.testing.assert_allclose(linear.c_at(-0.9), mapped.c_at(-0.9), atol=1e-14)
    x_edge = 1 - 2 * smooth.upper.y0
    assert abs(linear.c_at(x_edge) - mapped.c_at(x_edge)) > 0.2
    with pytest.raises(ValueError):
        _MappedMedium1D(LayeredMedium2D(lower=SineInterface(0.25, 0.02)))


def test_spectral_reference_matches_the_initial_state() -> None:
    medium = LayeredMedium2D(layer=ODD_RATIO, edge_width=0.04)
    nodes = make_node_set(medium, 900, repulsion_steps=5)
    np.testing.assert_allclose(
        spectral_plane_wave(nodes, 0.0, medium), plane_p_wave(nodes, medium), atol=1e-12
    )


def test_spectral_reference_converges_to_the_ray_sum_as_the_edge_sharpens() -> None:
    pts = _column(0.3, np.linspace(0, 1, 801))
    t = 0.3
    exact = exact_plane_wave(pts, t, FLAT)
    errors = []
    for width in (0.008, 0.004, 0.002):
        ref = spectral_plane_wave(pts, t, LayeredMedium2D(edge_width=width))
        assert np.all(ref[0] == 0) and np.all(ref[3] == 0)
        errors.append(max(np.abs(ref[k] - exact[k]).max() for k in (1, 2, 4)))
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all(ratios > 1.7), (errors, ratios)  # first order in delta
    assert errors[-1] < 0.05, errors


def test_spectral_reference_is_self_convergent_at_delta_0_005() -> None:
    # By t = 0.4 the pulse has split at y = 0.5 and the transmitted part has
    # crossed the band and left through y = 0.25 (exit at t = 0.25).
    medium = LayeredMedium2D(layer=ODD_RATIO, edge_width=0.005)
    pts = _column(0.6, np.linspace(0, 1, 401))
    t = 0.4
    assert reference_size(2 * medium.edge_width) == 2048
    base = spectral_plane_wave(pts, t, medium, dt=5e-5)
    finer_grid = spectral_plane_wave(pts, t, medium, n_ref=4096, dt=5e-5)
    finer_step = spectral_plane_wave(pts, t, medium, dt=2.5e-5)
    for k in (1, 2, 4):
        assert np.abs(base[k] - finer_grid[k]).max() < 1e-9, k
        assert np.abs(base[k] - finer_step[k]).max() < 1e-9, k
    assert np.abs(base[1]).max() > 0.5  # the wave is there


def test_stress_is_exact_and_the_jump_shortcut_is_off_by_the_tail_overlap() -> None:
    # f = ratio h is exact only where h_0 vanishes or the material is the
    # background; the discrepancy |h_0| |ratio - ratio_bg| is set at t = 0
    # and frozen. It is negligible at delta = 0.01 and visible at 0.04.
    ys = np.linspace(0, 1, 401)
    pts = _column(0.3, ys)
    for width, low, high in ((0.01, 0.0, 1e-13), (0.04, 1e-6, 1e-4)):
        medium = LayeredMedium2D(layer=ODD_RATIO, edge_width=width)
        state0 = plane_p_wave(make_node_set(medium, 400, repulsion_steps=1), medium)
        assert np.abs(state0).max() > 0.99
        lam = medium.lam_at(pts[:, 0], pts[:, 1])
        mu = medium.mu_at(pts[:, 0], pts[:, 1])
        ratio = lam / (lam + 2 * mu)
        h0 = exact_plane_wave(pts, 0.0, FLAT)[4]  # same pulse, jump or smooth
        predicted = np.abs(h0 * (ratio - 1 / 3)).max()
        assert low <= predicted < high, (width, predicted)
        ref = spectral_plane_wave(pts, 0.3, medium)
        shortcut_error = np.abs(ref[2] - ratio * ref[4]).max()
        assert shortcut_error == pytest.approx(predicted, abs=1e-12), width


def test_matches_a_resolved_naive_2d_run_to_its_resolution_floor() -> None:
    # delta = 2h at 2500 nodes: the naive RBF-FD run resolves the edge, so its
    # error against the spectral reference is the pulse's own resolution error
    # (uniform medium vs the exact solution), while against the jump ray sum
    # it is off by the O(1) difference between smooth and jump reflections.
    n, t = 2500, 0.3
    medium = LayeredMedium2D(edge_width=0.04)
    nodes = make_node_set(medium, n, repulsion_steps=40)
    naive = run(nodes, medium, t_end=t, n_snapshots=1).state[-1]
    ref = spectral_plane_wave(nodes, t, medium)
    floor = run(nodes, UNIFORM, t_end=t, n_snapshots=1).state[-1]
    exact_uniform = exact_plane_wave(nodes, t, UNIFORM)
    # v, f and h alike: f = lam / (lam + 2 mu) h holds in the reference (1/3
    # here) and the run's f is as close to it as its v and h are to theirs.
    for k in (1, 2, 4):
        err, floor_err = _rel(naive[k], ref[k]), _rel(floor[k], exact_uniform[k])
        assert err < 1.5 * floor_err, (k, err, floor_err)
    np.testing.assert_allclose(ref[2], ref[4] / 3, atol=1e-13)
    jump_err = _rel(naive[1], exact_plane_wave(nodes, t, FLAT)[1])
    assert jump_err > 3 * _rel(naive[1], ref[1]), jump_err


# --- the seed-aware operator (#39) -----------------------------------------------


def test_seed_rows_are_the_stencils_that_see_the_edge_and_nothing_else_moves() -> None:
    nodes = make_node_set(FLAT, 900, repulsion_steps=10)
    smooth = LayeredMedium2D(edge_width=nodes.h / 8)
    naive = build_operators(nodes, smooth, mode="naive")
    aware = build_operators(nodes, smooth, mode="aware")
    idx, _ = periodic_knn(nodes.xy, 19)
    sees = np.array(
        [smooth.varies_over(nodes.x[idx[i]], nodes.y[idx[i]]) for i in range(nodes.n)]
    )
    np.testing.assert_array_equal(aware.interface_nodes, np.flatnonzero(sees))
    assert 0 < sees.sum() < nodes.n
    # 19 * delta + the stencil radius from an edge centre, and no further.
    dist = smooth.distance_to_interfaces(nodes.x, nodes.y)
    assert dist[sees].max() < 19 * smooth.edge_width + 3 * nodes.h
    assert dist[~sees].min() > 19 * smooth.edge_width - 3 * nodes.h
    for a, b in (
        (aware.elastic, naive.elastic),
        (aware.hyper_block, naive.hyper_block),
    ):
        changed = np.unique(abs(a - b).nonzero()[0] % nodes.n)
        assert set(changed) <= set(aware.interface_nodes)
        untouched = np.setdiff1d(np.arange(nodes.n), aware.interface_nodes)
        rows = np.concatenate([f * nodes.n + untouched for f in range(5)])
        assert abs(a[rows] - b[rows]).max() == 0.0
    # Seed rows are coupled: u_t draws on h data through the stress seeds.
    i = aware.interface_nodes[0]
    row = aware.elastic[[i], :].toarray().ravel()
    assert np.abs(row[4 * nodes.n : 5 * nodes.n]).max() > 0
    # Elastic rows on the 19-node interface stencil (3 stress fields), the
    # Delta^3 rows on the naive 30-node footprint (2 velocity fields).
    assert set(np.diff(aware.elastic.indptr)[aware.interface_nodes]) == {3 * 19}
    assert set(np.diff(aware.hyper_block.indptr)[aware.interface_nodes]) == {2 * 30}
    small = build_operators(nodes, smooth, mode="aware", seed_hyper_stencil=19)
    assert set(np.diff(small.hyper_block.indptr)[small.interface_nodes]) == {2 * 19}
    assert abs(small.elastic - aware.elastic).max() == 0.0
    # The 30-node Delta^3 rows carry the naive rows' damping; the 19-node
    # ones about half of it (the reason for the footprint, notes section 5.3).
    naive_norm = abs(naive.hyper_block).sum(axis=1)[aware.interface_nodes]
    big_norm = abs(aware.hyper_block).sum(axis=1)[aware.interface_nodes]
    small_norm = abs(small.hyper_block).sum(axis=1)[small.interface_nodes]
    assert 0.5 < np.median(big_norm / naive_norm) < 2.0
    assert np.median(small_norm / naive_norm) < 0.75
    # A tolerance trims the tails; the surviving rows all see the edge. Built
    # on a process pool: a seed row depends on its own stencil only, so the
    # rows must equal the serial build's bit for bit.
    trimmed = build_operators(nodes, smooth, mode="aware", seed_rtol=1e-3, workers=4)
    assert 0 < trimmed.interface_nodes.size < aware.interface_nodes.size
    assert set(trimmed.interface_nodes) <= set(aware.interface_nodes)
    rows = np.concatenate([f * nodes.n + trimmed.interface_nodes for f in range(5)])
    assert abs(trimmed.elastic[rows] - aware.elastic[rows]).max() == 0.0
    assert abs(trimmed.hyper_block[rows] - aware.hyper_block[rows]).max() == 0.0


def test_seed_aware_operator_is_exact_on_a_resolved_plane_wave() -> None:
    # Through a resolved edge (delta = 1.9 h) the rates of the plane pulse are
    # pointwise: h_t = (lam + 2 mu) v_y and v_t = h_y / rho with the local
    # material, u_t = g_t = 0. Both operators must reproduce them to the
    # resolution error, and the seed rows must not excite u.
    nodes = make_node_set(FLAT, 900, repulsion_steps=10)
    smooth = LayeredMedium2D(edge_width=0.0625)
    state = plane_p_wave(nodes, smooth, center=0.75, sharpness=8.0)
    lam, mu, rho = smooth.material_at(nodes.x, nodes.y)
    dy = minimal_image(nodes.y - 0.75)
    v_y = -2 * 64 * dy * np.exp(-64 * dy**2)
    exact_h_t = (lam + 2 * mu) * v_y
    exact_v_t = np.sqrt(3) * v_y / rho
    scale = np.abs(exact_h_t).max()
    for mode in ("naive", "aware"):
        ops = build_operators(nodes, smooth, mode=mode, seed_rtol=1e-3)
        rate = (ops.elastic @ state.ravel()).reshape(5, -1)
        np.testing.assert_allclose(rate[4], exact_h_t, atol=1e-2 * scale)
        np.testing.assert_allclose(rate[1], exact_v_t, atol=1e-2 * scale)
        assert np.abs(rate[0]).max() < 1e-2 * scale
        assert np.abs(rate[3]).max() < 1e-2 * scale
    assert ops.interface_nodes.size > nodes.n // 3


# --- the flat sweep (#40) --------------------------------------------------------


def test_seed_stencils_beat_naive_through_an_edge_between_the_rows() -> None:
    # The smallest configuration that shows the sweep's result: 900 nodes,
    # delta = h/8 (the edge centre sits between the straddling rows at h/2,
    # which see 98% of the contrast), the wide pulse of the sweep, t = 1.
    # The seeds cut the spurious u (exact: 0) 5x, to 1.5x the resolution
    # floor's, and bring v from 1.9x the floor to 1.1x; the numbers are those
    # of docs/stiff-features.md section 5.4.
    nodes = make_node_set(FLAT, 900, seed=0)
    medium = LayeredMedium2D(edge_width=nodes.h / 8)
    pulse = dict(pulse_center=0.875, pulse_sharpness=15.0)
    ref = spectral_plane_wave(nodes, 1.0, medium, center=0.875, sharpness=15.0)
    err_v, u_max = {}, {}
    for mode in ("naive", "aware"):
        ops = build_operators(nodes, medium, mode=mode, workers=4)
        state = run(nodes, medium, t_end=1.0, n_snapshots=1, operators=ops, **pulse)
        err_v[mode] = _rel(state.state[-1][1], ref[1])
        u_max[mode] = np.abs(state.state[-1][0]).max()
    floor = run(nodes, UNIFORM, t_end=1.0, n_snapshots=1, **pulse).state[-1]
    floor_v = _rel(floor[1], exact_plane_wave(nodes, 1.0, UNIFORM, 0.875, 15.0)[1])
    assert u_max["aware"] < u_max["naive"] / 3, u_max
    assert err_v["aware"] < err_v["naive"] / 1.4, err_v
    assert err_v["aware"] < 1.25 * floor_v, (err_v, floor_v)
    assert err_v["naive"] > 1.6 * floor_v, (err_v, floor_v)
