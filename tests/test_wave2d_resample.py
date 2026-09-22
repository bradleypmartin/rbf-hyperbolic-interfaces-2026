import numpy as np
import pytest

from rbf_hyperbolic_interfaces.wave2d import (
    LayeredMedium2D,
    exact_plane_wave,
    make_node_set,
    pixel_grid,
    resample_matrix,
)

FLAT = LayeredMedium2D()


def test_pixel_grid_layout() -> None:
    m = 5
    grid = pixel_grid(m)
    assert grid.shape == (m * m, 2)
    g = (np.arange(m) + 0.5) / m
    # Row-major in y then x: reshaping to (m, m) gives [y index, x index].
    x = grid[:, 0].reshape(m, m)
    y = grid[:, 1].reshape(m, m)
    np.testing.assert_allclose(x[2], g)
    np.testing.assert_allclose(y[:, 2], g)


def test_identity_on_the_nodes_themselves() -> None:
    nodes = make_node_set(FLAT, 900, repulsion_steps=10)
    onto = resample_matrix(nodes, nodes.xy, FLAT)
    f = np.sin(2 * np.pi * nodes.x) * np.cos(4 * np.pi * nodes.y)
    np.testing.assert_allclose(onto @ f, f, atol=1e-12)


def test_reproduces_piecewise_degree_four_polynomials() -> None:
    # A different quartic in the band than outside it; targets away from the
    # periodic seams so no stencil wraps (polynomials are not periodic).
    def poly(xy: np.ndarray) -> np.ndarray:
        x, y = xy[:, 0], xy[:, 1]
        inside = FLAT.in_layer(x, y)
        return np.where(
            inside,
            1 + 2 * x - y + x**2 * y**2 - 3 * y**4,
            -2 + x * y**3 + 0.5 * x**4,
        )

    nodes = make_node_set(FLAT, 2500, repulsion_steps=20)
    targets = np.random.default_rng(1).random((500, 2)) * 0.7 + 0.15
    onto = resample_matrix(nodes, targets, FLAT)
    np.testing.assert_allclose(onto @ poly(nodes.xy), poly(targets), atol=1e-10)


def test_smooth_field_converges_with_source_resolution() -> None:
    targets = pixel_grid(40)

    def g(xy: np.ndarray) -> np.ndarray:
        return np.sin(2 * np.pi * xy[:, 0]) * np.cos(4 * np.pi * xy[:, 1])

    # Degree-4 augmentation gives 5th order for a smooth field. The max norm
    # is too noisy on scattered nodes for pairwise rates, so fit an rms slope.
    ns = [900, 1600, 2500, 3600]
    errs = []
    for n in ns:
        nodes = make_node_set(FLAT, n, repulsion_steps=40)
        onto = resample_matrix(nodes, targets, FLAT)
        errs.append(np.sqrt(np.mean((onto @ g(nodes.xy) - g(targets)) ** 2)))
    h = 1 / np.sqrt(np.array(ns, dtype=float))
    slope = np.polyfit(np.log(h), np.log(errs), 1)[0]
    assert slope > 4.0, (errs, slope)
    assert errs[-1] < 5e-6


def test_one_sided_stencils_keep_the_kink_at_the_interface() -> None:
    # While the pulse straddles y = 0.5 (t = 0.15) v has a kink there. A
    # stencil that mixes both sides smears it and the error concentrates in
    # the interface band; one-sided stencils are unaffected. Away from the
    # band the two interpolants agree.
    fine = make_node_set(FLAT, 10000, repulsion_steps=40)
    coarse = make_node_set(FLAT, 2500, repulsion_steps=40, seed=3)
    v_fine = exact_plane_wave(fine, 0.15, FLAT)[1]
    v_true = exact_plane_wave(coarse, 0.15, FLAT)[1]
    band = FLAT.distance_to_interfaces(coarse.x, coarse.y) < 3 * fine.h
    errs = {}
    for one_sided in (True, False):
        onto = resample_matrix(fine, coarse.xy, FLAT, one_sided=one_sided)
        errs[one_sided] = np.abs(onto @ v_fine - v_true)
    assert errs[True][band].max() < 3e-4
    assert errs[False][band].max() > 10 * errs[True][band].max()
    assert errs[False][~band].max() == pytest.approx(errs[True][~band].max(), rel=0.05)


def test_rejects_bad_targets_and_starved_materials() -> None:
    nodes = make_node_set(FLAT, 900, repulsion_steps=5)
    with pytest.raises(ValueError, match="shape"):
        resample_matrix(nodes, np.zeros((4, 3)), FLAT)
    # A stencil larger than one material's node count cannot be one-sided;
    # the 4 x 4 grid has a row of targets inside the band (y = 0.375).
    n_layer = int(FLAT.in_layer(nodes.x, nodes.y).sum())
    with pytest.raises(ValueError, match="too few"):
        resample_matrix(nodes, pixel_grid(4), FLAT, stencil_size=n_layer + 1)
