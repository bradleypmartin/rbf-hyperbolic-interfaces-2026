"""How much hyperviscosity, and does the h^(2k-1) scaling hold up?

Left: relative error of the naive RBF-FD solution (uniform medium, plane
P-wave, t = 0.2) against node count for several multiples of the MATLAB
amplitude gamma = 2.4e-11 (h / 0.02)^5. Right: how the largest real part of
the operator spectrum and the RK4 amplification factor respond to gamma on
a small node set. Brad's warning: too little and RK4 blows up on the modes
with positive real part, too much and accuracy or the time step suffers.

    uv run python scripts/wave2d_hyperviscosity.py
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from rbf_hyperbolic_interfaces.plotting import INK_MUTED, INK_SECONDARY, use_demo_style
from rbf_hyperbolic_interfaces.wave2d import (
    ElasticMaterial,
    LayeredMedium2D,
    build_operators,
    exact_plane_wave,
    hyperviscosity_gamma,
    make_node_set,
    run,
    stable_dt,
)

UNIFORM = LayeredMedium2D(layer=ElasticMaterial(lam=1.0, mu=1.0, rho=1.0))
SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
SCALE_COLORS = ("#c9b3e0", "#9b7fc4", "#2a78d6", "#1c4f8c", "#0b2a4d")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ns", type=int, nargs="+", default=[1600, 2500, 3600, 4900])
    parser.add_argument("--n-eig", type=int, default=900)
    parser.add_argument("--t-end", type=float, default=0.2)
    parser.add_argument("--sharpness", type=float, default=10.0)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/wave2d_hyperviscosity.png")
    )
    args = parser.parse_args()

    # --- error vs n for each gamma multiple -------------------------------
    t0 = time.perf_counter()
    errors = {s: [] for s in SCALES}
    steps = {s: [] for s in SCALES}
    for n in args.ns:
        nodes = make_node_set(UNIFORM, n, repulsion_steps=60)
        ops = build_operators(nodes, UNIFORM)
        exact = None
        for s in SCALES:
            snaps = run(
                nodes,
                UNIFORM,
                t_end=args.t_end,
                gamma_scale=s,
                n_snapshots=1,
                pulse_sharpness=args.sharpness,
                operators=ops,
            )
            if exact is None:
                exact = exact_plane_wave(
                    nodes, snaps.t[-1], UNIFORM, sharpness=args.sharpness
                )
            v = snaps.field("v")[-1]
            err = np.linalg.norm(v - exact[1]) / np.linalg.norm(exact[1])
            errors[s].append(float(err) if np.isfinite(err) else np.nan)
            steps[s].append(round(args.t_end / snaps.dt))
    print(f"runs: {time.perf_counter() - t0:.1f}s")
    print("gamma x   " + "".join(f"{n:>12d}" for n in args.ns))
    for s in SCALES:
        row = "".join(
            f"{e:>9.2e}({k:3d})" if np.isfinite(e) else f"{'blew up':>14s}"
            for e, k in zip(errors[s], steps[s], strict=True)
        )
        print(f"{s:<8.2f}  {row}   (error, RK4 steps)")

    # --- spectrum vs gamma on a small set -----------------------------------
    nodes = make_node_set(UNIFORM, args.n_eig, repulsion_steps=60)
    ops = build_operators(nodes, UNIFORM)
    dense = ops.elastic.toarray()
    hyper = ops.hyper_block.toarray()
    gamma0 = hyperviscosity_gamma(nodes.h)
    sweep = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    max_re, amp = [], []
    for s in sweep:
        ev = np.linalg.eigvals(dense + s * gamma0 * hyper)
        dt = stable_dt(nodes, UNIFORM, hyper=ops.hyper, gamma=s * gamma0)
        z = ev * dt
        max_re.append(ev.real.max())
        amp.append(np.abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24).max())
    print("gamma x   max Re(lambda)   RK4 max|R| at stable_dt")
    for s, m, a in zip(sweep, max_re, amp, strict=True):
        print(f"{s:<8.2f}  {m:+12.3e}   {a:.5f}")

    # --- figure ------------------------------------------------------------
    use_demo_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ns = np.array(args.ns)
    for s, color in zip(SCALES, SCALE_COLORS, strict=True):
        left.loglog(
            ns, errors[s], "o-", color=color, label=f"{s:g} x MATLAB gamma", ms=5
        )
    ref = errors[1.0][0] * (ns[0] / ns) ** 2
    left.loglog(ns, ref, "--", color=INK_MUTED, lw=1.2)
    left.annotate(
        "4th order",
        (ns[-1], ref[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=10,
        va="center",
    )
    left.set_xlabel("number of nodes")
    left.set_ylabel(f"relative error in v at t = {args.t_end:g}")
    left.set_xticks(ns, [str(n) for n in ns])
    left.xaxis.set_minor_formatter(NullFormatter())
    left.set_title("Accuracy is insensitive to gamma in this window")
    left.legend(fontsize=9)

    right.plot(sweep, max_re, "o-", color=SCALE_COLORS[2], ms=5)
    right.axhline(0, color=INK_MUTED, lw=1)
    right.set_yscale("symlog", linthresh=0.1)
    right.set_ylim(-0.03, 100)
    right.set_xlabel("gamma / MATLAB gamma")
    right.set_ylabel("max Re(lambda)", color=SCALE_COLORS[2])
    right.set_title(f"Spectrum vs gamma ({args.n_eig} nodes)")
    twin = right.twinx()
    twin.plot(sweep, amp, "s--", color="#eb6834", ms=5)
    twin.set_ylabel("RK4 max |R| at the chosen step", color="#eb6834")
    twin.grid(False)
    fig.suptitle(
        "Hyperviscosity: gamma = 2.4e-11 (h / 0.02)^5 keeps the naive RBF-FD "
        "solution stable without costing accuracy",
        fontsize=12,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
