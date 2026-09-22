"""Smooth curved edges (#42): the signed-distance medium, the seed profile along
the true normal, route (a) seed operators, and the product-grid reference."""

import numpy as np
import pytest

from rbf_hyperbolic_interfaces.wave2d import (
    ElasticMaterial,
    LayeredMedium2D,
    SineInterface,
    build_operators,
    make_node_set,
    minimal_image,
    oblique_p_wave,
    periodic_knn,
    run_fourier,
    run_fourier_2d,
)
from rbf_hyperbolic_interfaces.wave2d.interface import closest_point
from rbf_hyperbolic_interfaces.wave2d.seeds import normal_profile, seed_basis

A = 0.02
CURVED = LayeredMedium2D(lower=SineInterface(0.25, A), upper=SineInterface(0.5, A))


def _curved(width: float, **kw) -> LayeredMedium2D:
    return LayeredMedium2D(
        lower=SineInterface(0.25, A),
        upper=SineInterface(0.5, A),
        edge_width=width,
        **kw,
    )


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# --- the medium ------------------------------------------------------------------


def test_signed_distance_is_orthogonal_and_the_vertical_offset_when_flat() -> None:
    ifc = SineInterface(0.5, A)
    x0 = np.array([0.05, 0.3, 0.62, 0.9])
    foot = np.stack([x0, ifc.height(x0)], -1)
    for d in (-0.3, -0.02, 0.0, 0.015, 0.4):
        p = foot + d * ifc.normal(x0)
        np.testing.assert_allclose(ifc.signed_distance(p[:, 0], p[:, 1]), d, atol=2e-15)
    flat = SineInterface(0.5)
    y = np.array([0.1, 0.55, 0.95])
    np.testing.assert_array_equal(flat.signed_distance(x0[:3], y), y - 0.5)
    # Curvature 0.79 at most: a signed distance is well defined to 1.27 and
    # the foot point is unique within half a period, where the blend is used.
    kappa = A * (2 * np.pi) ** 2
    assert kappa < 1.0


def test_curved_blend_pairs_the_edges_of_one_band_image() -> None:
    # Each edge's own nearest image would pair the lower edge of one band
    # with the upper edge of the next near y = 0.75 (weight -1); the blend
    # must stay in [0, 1], be periodic in both directions, and be exactly
    # 0 or 1 beyond 19 delta = 0.095 from both edges (the band is 0.25 wide).
    medium = _curved(0.005)
    rng = np.random.default_rng(3)
    x, y = rng.random(4000), rng.random(4000) * 3 - 1
    w = medium.layer_fraction(x, y)
    assert w.min() >= 0.0 and w.max() <= 1.0
    np.testing.assert_allclose(w, medium.layer_fraction(x, y + 1), atol=1e-15)
    np.testing.assert_allclose(w, medium.layer_fraction(x + 1, y), atol=1e-15)
    d_lower, d_upper = medium.normal_distances(x, y)
    far = (np.abs(d_lower) > 0.1) & (np.abs(d_upper) > 0.1)
    inside = far & (d_lower > 0) & (d_upper < 0)
    assert inside.sum() > 100 and (far & ~inside).sum() > 100
    np.testing.assert_array_equal(w[inside], 1.0)
    np.testing.assert_array_equal(w[far & ~inside], 0.0)
    # Distances refer to the same band: the pair is (d, d - gap) to within
    # the curvature and never straddles two images.
    assert np.all(d_upper < d_lower)
    assert np.abs((d_lower - d_upper) - 0.25).max() < 0.02


def test_curved_blend_reduces_to_the_flat_one_as_the_amplitude_vanishes() -> None:
    flat = LayeredMedium2D(edge_width=0.01)
    rng = np.random.default_rng(1)
    x, y = rng.random(2000), rng.random(2000)
    for amp, tol in ((1e-9, 1e-6), (1e-12, 1e-9)):
        tiny = LayeredMedium2D(
            lower=SineInterface(0.25, amp),
            upper=SineInterface(0.5, amp),
            edge_width=0.01,
        )
        assert not tiny.is_flat
        err = np.abs(tiny.layer_fraction(x, y) - flat.layer_fraction(x, y)).max()
        assert err < tol, (amp, err)
    # The flat branch itself is untouched: the smooth flat medium of #36
    # samples the same values as before to the bit (the cached operators of
    # #40 depend on it), spot-checked against the closed form it implements.
    d = 0.01
    y0 = np.mod(y, 1.0)
    expected = np.zeros_like(y)
    for m in (-1, 0, 1):
        expected += 0.5 * (np.tanh((y0 + m - 0.25) / d) - np.tanh((y0 + m - 0.5) / d))
    np.testing.assert_array_equal(flat.layer_fraction(x, y), expected)


def test_jump_curved_medium_is_untouched() -> None:
    nodes = make_node_set(CURVED, 400, repulsion_steps=5)
    inside = CURVED.in_layer(nodes.x, nodes.y)
    np.testing.assert_array_equal(CURVED.layer_fraction(nodes.x, nodes.y), inside)
    assert np.array_equal(CURVED.lam_at(nodes.x, nodes.y), np.where(inside, 4.0, 1.0))


# --- the seed profile along the normal --------------------------------------------


@pytest.mark.parametrize("which", ["lower", "upper"])
def test_normal_profile_is_the_edge_profile_in_the_normal_coordinate(which) -> None:
    # Along the normal through a foot point the medium's blend is exactly
    # the tanh step in y' for that edge, plus the other edge's tail.
    medium = _curved(0.005)
    ifc = getattr(medium, which)
    for x0 in (0.0, 0.13, 0.25, 0.7):
        profile = normal_profile(medium, ifc, x0)
        assert profile.theta == pytest.approx(float(ifc.angle(x0)))
        yp = np.linspace(-0.08, 0.08, 33)
        lam, _, _ = profile.material(yp)
        w = (lam - 1.0) / 3.0
        step = 0.5 * (1 + np.tanh(yp / 0.005))
        expected = step if which == "lower" else 1 - step
        np.testing.assert_allclose(w, expected, atol=1e-13)
        # The line's points have signed distance y' to this interface exactly.
        px, py = profile.points(yp)
        np.testing.assert_allclose(ifc.signed_distance(px, py), yp, atol=1e-15)
        # The crossings solve the line equation, and the own edge is at 0.
        for other in medium.interfaces:
            roots = profile.crossings(other)
            cx, cy = profile.points(roots)
            residual = cy - other.height(cx) - np.arange(-1.0, 2.0)
            np.testing.assert_allclose(residual, 0.0, atol=1e-14)
        assert profile.crossings(ifc)[1] == 0.0
    # A flat interface keeps the closed-form stops bit for bit.
    flat = normal_profile(
        LayeredMedium2D(edge_width=0.005), LayeredMedium2D().upper, 0.3
    )
    assert flat.theta == 0.0
    np.testing.assert_array_equal(
        flat.crossings(LayeredMedium2D().lower), [-1.25, -0.25, 0.75]
    )
    np.testing.assert_array_equal(
        flat.material(np.array([0.1])),
        LayeredMedium2D(edge_width=0.005).material_at(np.array([0.3]), np.array([0.6])),
    )


def test_curved_seeds_are_the_flat_seeds_of_the_same_local_stencil() -> None:
    # Route (a): the seeds see the material along the normal only, which
    # for the sine edge is the flat profile in y'. A stencil in the curved
    # frame must therefore get the same seeds as the flat medium would for
    # the same local coordinates, up to the other edge's tail.
    medium = _curved(0.005)
    nodes = make_node_set(CURVED, 900, repulsion_steps=10)
    fixed = np.flatnonzero(nodes.fixed)
    off = medium.lower.vertical_offset(nodes.x[fixed], nodes.y[fixed])
    cand = fixed[np.isclose(off, 0.5 * nodes.h, atol=0.05 * nodes.h)]
    centre = int(cand[np.argmin(np.abs(nodes.x[cand] - 0.13))])
    idx, _ = periodic_knn(nodes.xy, 19, query=nodes.xy[[centre]])
    x0, th = closest_point(medium.lower, nodes.x[[centre]], nodes.y[[centre]])
    origin = np.array([x0[0], medium.lower.height(x0[0])])
    disp = minimal_image(nodes.xy[idx[0]] - origin)
    c, s = np.cos(th[0]), np.sin(th[0])
    local = np.stack(
        [c * disp[:, 0] + s * disp[:, 1], -s * disp[:, 0] + c * disp[:, 1]], -1
    )
    assert abs(th[0]) > 0.05  # a stencil where the frame is really tilted
    curved = seed_basis(local, normal_profile(medium, medium.lower, x0[0]), 3)
    flat_medium = LayeredMedium2D(edge_width=0.005)
    flat = seed_basis(local, normal_profile(flat_medium, flat_medium.lower, x0[0]), 3)
    np.testing.assert_allclose(curved.uv, flat.uv, atol=1e-12)
    np.testing.assert_allclose(curved.fgh, flat.fgh, atol=1e-12)
    np.testing.assert_allclose(curved.uv_jet, flat.uv_jet, atol=1e-12)


# --- the operator ------------------------------------------------------------------


def test_seed_operator_on_a_curved_edge_reduces_to_naive_without_contrast() -> None:
    # Frames, profiles and rotations on the curved geometry: with a 1e-9
    # contrast the seeds are monomials to 1e-9 and every stress-rate seed
    # row must be the naive 19-node degree-3 row on the same footprint (the
    # velocity rows are constrained on the 27 stress seeds, not the full
    # degree-3 space, and differ from naive by design, flat or curved:
    # tests/test_wave2d_seeds.py checks those against the polynomial path).
    nodes = make_node_set(CURVED, 900, repulsion_steps=10)
    layer = ElasticMaterial(lam=1 + 1e-9, mu=1 + 1e-9, rho=1 + 1e-9)
    medium = _curved(nodes.h / 4, layer=layer)
    aware = build_operators(nodes, medium, mode="aware", workers=4)
    naive19 = build_operators(nodes, medium, stencil_size=19, poly_degree=3)
    rows = aware.interface_nodes
    assert 0 < rows.size < nodes.n
    stress_rows = np.concatenate([f * nodes.n + rows for f in (2, 3, 4)])
    diff = abs(aware.elastic[stress_rows] - naive19.elastic[stress_rows]).max()
    scale = abs(naive19.elastic[stress_rows]).max()
    assert diff < 1e-8 * scale, diff / scale


def test_seed_operator_on_a_curved_edge_selects_the_rows_that_see_it() -> None:
    nodes = make_node_set(CURVED, 900, repulsion_steps=10)
    medium = _curved(nodes.h / 8)
    aware = build_operators(nodes, medium, mode="aware", workers=4)
    idx, _ = periodic_knn(nodes.xy, 19)
    sees = np.array(
        [medium.varies_over(nodes.x[idx[i]], nodes.y[idx[i]]) for i in range(nodes.n)]
    )
    np.testing.assert_array_equal(aware.interface_nodes, np.flatnonzero(sees))
    assert set(np.diff(aware.elastic.indptr)[aware.interface_nodes]) == {3 * 19}
    assert set(np.diff(aware.hyper_block.indptr)[aware.interface_nodes]) == {2 * 30}
    # Rows on the tilted parts of the curve are rotated: u_t draws on f
    # data through the frame (zero for an axis-aligned seed stencil).
    tilted = aware.interface_nodes[
        np.abs(medium.lower.angle(nodes.x[aware.interface_nodes])) > 0.07
    ]
    row = aware.elastic[[tilted[0]], :].toarray().ravel()
    assert np.abs(row[2 * nodes.n : 3 * nodes.n]).max() > 1e-3 * np.abs(row).max()


# --- the product-grid reference ------------------------------------------------------


def test_grid_reference_matches_the_fourier_in_x_reference_on_a_flat_medium() -> None:
    # Same grid, same step: the two solvers differ only in representation
    # (modes vs physical space), so they must agree to rounding, at
    # oblique incidence where 13 modes couple through the edge.
    flat = LayeredMedium2D(edge_width=0.04)

    def initial(xy):
        return oblique_p_wave(xy, flat, (1, 2), 0.875, 15.0)

    modes = run_fourier(flat, initial, 0.1, n_y=256, n_x=64, dt=4e-4)[0]
    grid = run_fourier_2d(flat, initial, 0.1, n_x=64, n_y=256, dt=4e-4)[0]
    pts = np.random.default_rng(0).random((400, 2))
    a, b = modes.evaluate(pts), grid.evaluate(pts)
    assert np.abs(a - b).max() < 1e-11 * np.abs(a).max()
    assert grid.energy(flat) == pytest.approx(modes.energy(flat), rel=1e-12)
    np.testing.assert_allclose(grid.curl(pts), modes.curl(pts), atol=1e-10)


def test_grid_reference_interpolates_exactly_and_differentiates() -> None:
    curved = _curved(0.04)

    def initial(xy):
        return oblique_p_wave(xy, curved, (0, 1), 0.875, 15.0)

    grid = run_fourier_2d(curved, initial, 0.05, n_x=32, n_y=128)[0]
    gx, gy = np.meshgrid(grid.x, grid.y, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], -1)
    values = grid.evaluate(pts).reshape(5, 32, 128)
    np.testing.assert_allclose(values, grid.fields, atol=1e-13)
    # d/dy of v against a centred difference on a 4x finer trigonometric
    # sample (the interpolant is smooth, the difference is O(eps^2)).
    eps = 1e-4
    up = grid.evaluate(pts + [0, eps])[1]
    down = grid.evaluate(pts - [0, eps])[1]
    v_y = grid.evaluate(pts, dy=1)[1]
    assert np.abs(v_y - (up - down) / (2 * eps)).max() < 1e-5 * np.abs(v_y).max()


def test_grid_reference_conserves_energy_and_self_converges_on_a_curved_edge() -> None:
    # The pulse hits the curved upper edge by t = 0.25: energy is exact to
    # the time step, and doubling both grid directions moves the answer by
    # the tanh's spectral tail at delta = 0.04 on 64 x 128 (1e-11).
    curved = _curved(0.04)

    def initial(xy):
        return oblique_p_wave(xy, curved, (0, 1), 0.875, 15.0)

    coarse = run_fourier_2d(curved, initial, 0.3, n_x=32, n_y=128, dt=2e-4)
    fine = run_fourier_2d(curved, initial, 0.3, n_x=64, n_y=256, dt=2e-4)
    start = run_fourier_2d(curved, initial, 0.0, n_x=64, n_y=256)[0]
    assert fine[0].energy(curved) == pytest.approx(start.energy(curved), rel=1e-9)
    pts = np.random.default_rng(2).random((400, 2))
    a, b = coarse[0].evaluate(pts), fine[0].evaluate(pts)
    assert _rel(a[1], b[1]) < 1e-8
    # And the curved edge converts P to S: the curl is no longer zero.
    assert np.abs(fine[0].curl(pts)).max() > 1e-2 * np.abs(b[1]).max()
