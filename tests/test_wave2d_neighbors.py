import numpy as np

from rbf_hyperbolic_interfaces.wave2d import (
    minimal_image,
    periodic_knn,
    stencil_offsets,
    wrap,
)


def _brute_force_knn(xy: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    disp = minimal_image(xy[:, None, :] - xy[None, :, :])
    dist = np.linalg.norm(disp, axis=-1)
    idx = np.argsort(dist, axis=1, kind="stable")[:, :k]
    return idx, np.take_along_axis(dist, idx, axis=1)


def test_minimal_image_measures_the_short_way_round() -> None:
    disp = np.array([[0.9, -0.9], [0.4, -0.45], [0.0, 1.0]])
    np.testing.assert_allclose(minimal_image(disp), [[-0.1, 0.1], [0.4, -0.45], [0, 0]])


def test_wrap_lands_in_unit_square() -> None:
    xy = wrap(np.array([[-0.25, 1.0], [1.75, -1e-9], [-1e-17, 0.3]]))
    assert np.all((xy >= 0) & (xy < 1))
    np.testing.assert_allclose(xy[0], [0.75, 0.0])
    # Sub-ULP negatives used to round to exactly 1.0 and crash the KD-tree.
    periodic_knn(np.array([[-1e-17, 0.5], [0.5, 0.5], [0.2, 0.2]]), 2)


def test_periodic_knn_matches_brute_force() -> None:
    rng = np.random.default_rng(3)
    xy = rng.random((300, 2))
    idx, dist = periodic_knn(xy, 7)
    idx_bf, dist_bf = _brute_force_knn(xy, 7)
    np.testing.assert_allclose(dist, dist_bf, atol=1e-12)
    # Index order can only differ for exactly tied distances, which random
    # points do not produce.
    np.testing.assert_array_equal(idx, idx_bf)
    assert np.all(idx[:, 0] == np.arange(300))


def test_periodic_knn_crosses_the_seam() -> None:
    xy = np.array([[0.01, 0.5], [0.98, 0.5], [0.5, 0.5], [0.5, 0.02], [0.5, 0.97]])
    idx, dist = periodic_knn(xy, 2)
    assert idx[0, 1] == 1 and abs(dist[0, 1] - 0.03) < 1e-12
    assert idx[3, 1] == 4 and abs(dist[3, 1] - 0.05) < 1e-12


def test_stencil_offsets_are_relative_to_centre_and_wrapped() -> None:
    xy = np.array([[0.02, 0.5], [0.97, 0.52], [0.05, 0.48]])
    idx = np.array([[0, 1, 2]])
    off = stencil_offsets(xy, idx)
    np.testing.assert_allclose(off[0], [[0, 0], [-0.05, 0.02], [0.03, -0.02]])
    # Explicit centres elsewhere than a node.
    off_c = stencil_offsets(xy, idx, centers=np.array([[0.99, 0.5]]))
    np.testing.assert_allclose(off_c[0, 0], [0.03, 0.0])
