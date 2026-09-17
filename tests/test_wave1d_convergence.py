"""End-to-end: reproduces dissertation Fig. 2-8 (naive first order, aware fourth)."""

import numpy as np

from pdes_demo.wave1d import LayeredMedium, exact_solution, periodic_grid, run
from pdes_demo.wave1d.operators import stencil_crossings


def _rel_l2_error(
    n: int, medium: LayeredMedium, mode: str, t_end: float = 1.0
) -> float:
    grid = periodic_grid(n)
    snaps = run(grid, medium, mode=mode, t_end=t_end, n_snapshots=1)
    _, f_exact = exact_solution(grid.x, snaps.t[-1], medium)
    return float(np.linalg.norm(snaps.f[-1] - f_exact) / np.linalg.norm(f_exact))


def _rates(ns: list[int], errors: list[float]) -> np.ndarray:
    e = np.array(errors)
    n = np.array(ns, dtype=float)
    return np.log(e[:-1] / e[1:]) / np.log(n[1:] / n[:-1])


def test_aware_is_fourth_order_naive_is_first_order() -> None:
    medium = LayeredMedium()  # c: 1 -> 2, layer [0, 0.5)
    ns = [200, 400, 800]
    aware = [_rel_l2_error(n, medium, "aware") for n in ns]
    naive = [_rel_l2_error(n, medium, "naive") for n in ns]
    assert np.all(_rates(ns, aware) > 3.5), _rates(ns, aware)
    assert np.all(_rates(ns, naive) < 1.5), _rates(ns, naive)
    assert aware[-1] < naive[-1] / 50


def test_thin_layer_double_cross_stays_fourth_order() -> None:
    # Layer width 0.01 is narrower than the 4th-order stencil (4h) on every
    # grid here, so some interface stencils straddle both edges at once.
    # (At n = 800 the layer equals 4h exactly and no stencil double-crosses,
    # which is why the sequence stops at 600.)
    medium = LayeredMedium(layer_width=0.01)
    ns = [200, 400, 600]
    for n in ns:
        crossings = stencil_crossings(periodic_grid(n), medium)
        assert (crossings == 2).sum() > 0, n
    aware = [_rel_l2_error(n, medium, "aware") for n in ns]
    naive = [_rel_l2_error(n, medium, "naive") for n in ns]
    assert np.all(_rates(ns, aware) > 3.5), _rates(ns, aware)
    assert aware[-1] < naive[-1] / 10


def test_uniform_medium_translation_error_is_small() -> None:
    from pdes_demo.wave1d import Material

    medium = LayeredMedium(layer=Material(c=1.0, rho=1.0))
    err = _rel_l2_error(400, medium, "naive", t_end=0.5)
    assert err < 5e-3
