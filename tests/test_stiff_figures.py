"""The figures of #54 drawn from a synthetic cache, and the 2-D seed sections.

The drawing tests render both layouts into a temporary directory (a broken
record set or layout raises). The section test is numerical: frozen
material returns the monomials and their stress rates, and the jump limit
of the seed ``u = Y`` has the slope ratio the shear traction dictates.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pytest

from pdes_demo.plotting import use_demo_style, use_print_style
from pdes_demo.results_cache import ResultsCache
from pdes_demo.stiff_figures import (
    SEED_MEMBERS_2D,
    convergence_1d,
    convergence_2d,
    seed_sections_2d,
)
from pdes_demo.wave2d import LayeredMedium2D

matplotlib.use("Agg")


def _cache_1d() -> ResultsCache:
    cache = ResultsCache.new("scripts/wave1d_stiff.py", {"t_end": 1.0})
    ns = [100, 200, 400]
    h = [2 / n for n in ns]
    for delta in (0.0, 0.0025, 0.01):
        cache.add_errors(
            ns,
            [7e-2, 3e-2, 1.6e-2],
            [1.2, 1.0],
            h=h,
            delta=delta,
            mode="naive",
            field="f",
        )
        cache.add_errors(
            ns,
            [4e-3, 2.5e-4, 1.6e-5],
            [4.0, 4.0],
            h=h,
            delta=delta,
            mode="aware",
            field="f",
        )
    return cache


def _cache_2d(u_field: str, amplitude: float = 0.0, direction=(0, 1)) -> ResultsCache:
    args = {"t_end": 1.0, "amplitude": amplitude, "direction": list(direction)}
    cache = ResultsCache.new("scripts/wave2d_stiff.py", args)
    ns = [2500, 4900, 10000]
    h = [n**-0.5 for n in ns]
    widths = [0.0025, 0.01] if direction != (0, 1) else [0.0, 0.0025, 0.01]
    for delta in widths:
        modes = ["naive", "aware", "sfloor"] if delta else ["naive", "aware"]
        for mode in modes:
            for field in ("v", "h", u_field):
                cache.add_errors(
                    ns,
                    [8e-2, 4e-2, 2e-2],
                    [2.0, 2.0],
                    h=h,
                    delta=delta,
                    mode=mode,
                    field=field,
                )
    for field in ("v", "h", u_field):
        cache.add_errors(
            ns,
            [5e-2, 1.8e-2, 5e-3],
            [3.0, 3.7],
            h=h,
            delta=None,
            mode="floor",
            field=field,
        )
    return cache


@pytest.mark.parametrize("print_mode", [False, True])
def test_convergence_figures_render_in_both_layouts(tmp_path: Path, print_mode) -> None:
    (use_print_style if print_mode else use_demo_style)()
    out = convergence_1d(_cache_1d(), tmp_path / "c1.pdf", print_mode=print_mode)
    assert out.stat().st_size > 1000
    for name, cache in [
        ("flat", _cache_2d("max_u")),
        ("oblique", _cache_2d("u", direction=(1, 2))),
        ("curved", _cache_2d("u", amplitude=0.02)),
    ]:
        out = convergence_2d(cache, tmp_path / f"{name}.pdf", print_mode=print_mode)
        assert out.stat().st_size > 1000


def test_print_pdfs_are_byte_identical_across_renders(tmp_path: Path) -> None:
    use_print_style()
    cache = _cache_1d()
    a = convergence_1d(cache, tmp_path / "a.pdf", print_mode=True)
    b = convergence_1d(cache, tmp_path / "b.pdf", print_mode=True)
    assert a.read_bytes() == b.read_bytes()


def test_seed_sections_frozen_material_gives_monomials_and_their_stresses() -> None:
    y = np.linspace(-1.0, 1.0, 41)
    h, delta = 0.02, 0.005
    mono = seed_sections_2d(delta, y, h=h, frozen=True)
    # The anchor is h/2 below the lower edge (y = 0.25), inside the tanh tail:
    # the frozen material is the anchor's, not exactly the background's.
    medium = LayeredMedium2D(edge_width=delta)
    _, mu, _ = medium.material_at(np.array([0.3]), np.array([0.25 - h / 2]))
    mu_e = float(mu[0])
    assert 1.0 < mu_e < 1.1
    np.testing.assert_allclose(mono[("u", (0, 1), "u", 0.0)], y, atol=1e-10)
    np.testing.assert_allclose(mono[("v", (0, 2), "v", 0.0)], y**2, atol=1e-10)
    np.testing.assert_allclose(mono[("u", (0, 2), "g", 0.0)], 2 * mu_e * y, atol=1e-10)
    np.testing.assert_allclose(mono[("v", (1, 1), "u", 0.5)], 0.0, atol=1e-10)


def test_seed_sections_jump_limit_has_the_traction_slope_ratio() -> None:
    """Through a near-jump the seed u = Y keeps mu u_Y continuous, so its
    slope above the edge (mu = 4) is a quarter of the slope below (mu = 1);
    the traction g of u = Y^2 stays continuous with slope ratio rho = 2."""
    y = np.linspace(-1.0, 1.0, 401)
    sec = seed_sections_2d(1e-6, y)
    edge = 0.2  # the anchor is h/2 below the edge, r_max = 2.5 h
    below = (y > -0.8) & (y < -0.2)
    above = (y > 0.4) & (y < 1.0)
    u = sec[("u", (0, 1), "u", 0.0)]
    s_below = np.polyfit(y[below], u[below], 1)[0]
    s_above = np.polyfit(y[above], u[above], 1)[0]
    assert s_below == pytest.approx(1.0, rel=1e-6)
    assert s_above == pytest.approx(0.25, rel=1e-4)
    g = sec[("u", (0, 2), "g", 0.0)]
    g_below = np.polyfit(y[below], g[below], 1)[0]
    g_above = np.polyfit(y[above], g[above], 1)[0]
    assert g_below == pytest.approx(2.0, rel=1e-6)
    assert g_above == pytest.approx(4.0, rel=1e-4)
    # continuity of u and g across the edge, to the resolution of the grid
    k = int(np.argmin(np.abs(y - edge)))
    assert abs(u[k + 1] - u[k - 1]) < 0.02 and abs(g[k + 1] - g[k - 1]) < 0.05
    # the induced u of v = XY vanishes below the edge and is non-zero above
    induced = sec[("v", (1, 1), "u", 0.5)]
    assert np.abs(induced[below]).max() < 1e-8
    assert np.abs(induced[above]).max() > 1e-3
    assert len(SEED_MEMBERS_2D) == 4
