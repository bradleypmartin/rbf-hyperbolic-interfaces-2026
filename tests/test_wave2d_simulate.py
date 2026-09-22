import numpy as np
import pytest

from pdes_demo.wave2d import (
    ElasticMaterial,
    LayeredMedium2D,
    build_operators,
    energy,
    exact_plane_wave,
    hyperviscosity_gamma,
    make_node_set,
    run,
    stable_dt,
)
from pdes_demo.wave2d.simulate import hyperviscosity_extreme

FLAT = LayeredMedium2D()
UNIFORM = LayeredMedium2D(layer=ElasticMaterial(lam=1.0, mu=1.0, rho=1.0))


def _rel_error_v(snaps, nodes, medium, sharpness: float) -> float:
    exact = exact_plane_wave(nodes, snaps.t[-1], medium, sharpness=sharpness)
    v = snaps.field("v")[-1]
    return float(np.linalg.norm(v - exact[1]) / np.linalg.norm(exact[1]))


def test_uniform_medium_plane_wave_converges() -> None:
    ns = [1600, 2500, 3600]
    errs = []
    for n in ns:
        nodes = make_node_set(UNIFORM, n, repulsion_steps=40)
        snaps = run(nodes, UNIFORM, t_end=0.2, n_snapshots=1, pulse_sharpness=10.0)
        errs.append(_rel_error_v(snaps, nodes, UNIFORM, 10.0))
    h = 1 / np.sqrt(np.array(ns, dtype=float))
    rates = np.log(np.array(errs[:-1]) / np.array(errs[1:])) / np.log(h[:-1] / h[1:])
    assert np.all(rates > 3.0), (errs, rates)
    assert errs[-1] < 2e-3


def test_layered_naive_is_stable_and_improves_with_resolution() -> None:
    # Naive stencils straight across the impedance jump: the error is large
    # and converges slowly (dissertation Fig. 3-5 "FD4"-like); this is the
    # baseline the interface-aware stencils must beat.
    errs = []
    for n in (2500, 4900):
        nodes = make_node_set(FLAT, n, repulsion_steps=40)
        snaps = run(nodes, FLAT, t_end=0.3, n_snapshots=1)
        errs.append(_rel_error_v(snaps, nodes, FLAT, 23.0))
        v = snaps.field("v")[-1]
        assert np.isfinite(v).all() and np.abs(v).max() < 1.5
    assert errs[0] < 0.2 and errs[1] < 0.75 * errs[0], errs


def test_snapshot_bookkeeping() -> None:
    nodes = make_node_set(FLAT, 400, repulsion_steps=5)
    snaps = run(nodes, FLAT, t_end=0.05, n_snapshots=4)
    assert snaps.state.shape[1:] == (5, 400)
    assert snaps.t[0] == 0.0 and snaps.t[-1] == pytest.approx(0.05)
    assert np.allclose(np.diff(snaps.t), snaps.t[1] - snaps.t[0])
    assert snaps.gamma == pytest.approx(hyperviscosity_gamma(nodes.h))
    np.testing.assert_array_equal(snaps.field("u"), snaps.state[:, 0])


def test_final_step_is_kept_when_the_stride_does_not_divide() -> None:
    # 900 nodes, t_end = 0.3 gives 45 steps; 11 snapshots means a stride of
    # 4, which does not divide 45. The last stored time must still be t_end.
    nodes = make_node_set(FLAT, 900, repulsion_steps=5)
    snaps = run(nodes, FLAT, t_end=0.3, n_snapshots=11)
    n_steps = round(0.3 / snaps.dt)
    assert n_steps % (n_steps // 11) != 0
    assert snaps.t[-1] == pytest.approx(0.3)
    assert np.allclose(np.diff(snaps.t[:-1]), (n_steps // 11) * snaps.dt)
    assert snaps.t[-1] - snaps.t[-2] < (n_steps // 11) * snaps.dt


def test_hyperviscosity_extreme_falls_back_to_gershgorin(monkeypatch) -> None:
    import scipy.sparse.linalg as sla

    nodes = make_node_set(FLAT, 400, repulsion_steps=5)
    ops = build_operators(nodes, FLAT)
    gersh = float(np.max(np.abs(ops.hyper).sum(axis=1)))
    arpack = hyperviscosity_extreme(ops.hyper)
    assert 0.3 * gersh < arpack < gersh

    def stall(*args, **kwargs):
        raise sla.ArpackNoConvergence("stalled", np.array([]), np.array([]))

    monkeypatch.setattr(sla, "eigs", stall)
    assert hyperviscosity_extreme(ops.hyper) == pytest.approx(gersh)
    # A complex or positive dominant eigenvalue is rejected the same way.
    monkeypatch.setattr(sla, "eigs", lambda *a, **k: np.array([1e9 + 1e9j]))
    assert hyperviscosity_extreme(ops.hyper) == pytest.approx(gersh)


def test_step_cap_keeps_rk4_amplification_bounded() -> None:
    # At 3x the MATLAB gamma the CFL step alone leaves the most damped modes
    # outside RK4's real-axis range; the hyperviscosity cap brings them back.
    nodes = make_node_set(FLAT, 400, repulsion_steps=20)
    ops = build_operators(nodes, FLAT)
    gamma = 3 * hyperviscosity_gamma(nodes.h)
    ev = np.linalg.eigvals((ops.elastic + gamma * ops.hyper_block).toarray())

    def amplification(dt: float) -> float:
        z = ev * dt
        return float(np.abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24).max())

    dt_cfl = stable_dt(nodes, FLAT)
    dt_capped = stable_dt(nodes, FLAT, hyper=ops.hyper, gamma=gamma)
    assert dt_capped < dt_cfl
    assert amplification(dt_cfl) > 1.05
    assert amplification(dt_capped) < 1.005
    # The ARPACK estimate matches a dense eigenvalue solve.
    dense = np.linalg.eigvals(ops.hyper.toarray())
    assert hyperviscosity_extreme(ops.hyper) == pytest.approx(
        abs(dense.real.min()), rel=1e-3
    )


def test_energy_tracks_the_exact_state_on_the_same_nodes() -> None:
    # The node quadrature itself drifts by a few percent as the pulse moves,
    # so compare numerical and exact energies on identical nodes instead.
    nodes = make_node_set(UNIFORM, 2500, repulsion_steps=40)
    snaps = run(nodes, UNIFORM, t_end=0.2, n_snapshots=2, pulse_sharpness=10.0)
    for t, state in zip(snaps.t, snaps.state, strict=True):
        e_num = energy(state, nodes, UNIFORM)
        e_ref = energy(
            exact_plane_wave(nodes, t, UNIFORM, sharpness=10.0), nodes, UNIFORM
        )
        assert abs(e_num / e_ref - 1) < 1e-2, (t, e_num, e_ref)


def test_aligned_snapshots_share_times_across_resolutions() -> None:
    # 400 and 900 nodes have different stable steps (about 30 and 45 steps to
    # t = 0.3); aligned to 7 snapshots both store exactly j * 0.3 / 7.
    times = []
    for n in (400, 900):
        nodes = make_node_set(FLAT, n, repulsion_steps=5)
        snaps = run(nodes, FLAT, t_end=0.3, n_snapshots=7, align_snapshots=True)
        assert len(snaps.t) == 8
        assert round(0.3 / snaps.dt) % 7 == 0
        times.append(snaps.t)
    np.testing.assert_allclose(times[0], times[1], atol=1e-14)
    np.testing.assert_allclose(times[0], np.arange(8) * 0.3 / 7, atol=1e-14)
