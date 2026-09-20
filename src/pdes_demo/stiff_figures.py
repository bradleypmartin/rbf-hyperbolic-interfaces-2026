"""Part 3 figures drawn from the results cache (issue #54).

The stiff drivers used to draw their convergence figures inline from the
numbers they had just computed. The manuscript needs the same figures in
print style, regenerated from the committed cache without the sweeps, so the
drawing lives here and takes :class:`~pdes_demo.results_cache.ResultsCache`
records: a driver passes the cache it has just filled, and
``scripts/paper_figures.py`` passes one read from ``paper/data/``. The seed
basis figures (1-D, and the 2-D cross-sections the manuscript asks for)
compute their curves directly; they take seconds.

Every function draws in whatever rcParams are active
(:func:`~pdes_demo.plotting.use_demo_style` or
:func:`~pdes_demo.plotting.use_print_style`); ``print_mode`` picks the layout
and the labels. Print figures are laid out at :data:`TEXTWIDTH_IN` so the
fonts land at true size; demo figures keep the deck's wide layouts and their
suptitles (a print figure gets its caption in LaTeX).
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from .plotting import (
    AWARE,
    COLORS,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    LABELS,
    PRINT_LABELS,
    PRINT_LABELS_2D,
    TEXTWIDTH_IN,
)
from .results_cache import ResultsCache
from .wave1d import LayeredMedium
from .wave1d.stiff import seed_basis
from .wave2d import LayeredMedium2D
from .wave2d.seeds import SeedChain, frozen_profile, normal_profile

# Line styles per scheme, shared by the 2-D convergence figures.
STYLE_2D = {
    "naive": dict(color=COLORS["naive"], marker="o", ls="-"),
    "aware": dict(color=COLORS["aware"], marker="o", ls="-"),
    "ablate": dict(color=COLORS["aware"], marker="^", ls="--", mfc="none"),
    "floor": dict(color=INK_SECONDARY, marker="s", ls="--"),
    "sfloor": dict(color=AWARE, marker="s", ls=":", mfc="none"),
}
DEMO_LABELS_2D = {
    "naive": "standard RBF-FD (naive)",
    "aware": "seed stencils (interface-aware at delta = 0)",
    "ablate": "seeds for the normal monomials only (ablation)",
    "floor": "no interface (resolution floor)",
    "sfloor": "seed operator, no contrast (seed floor)",
}
PRINT_LABELS_2D_ALL = {
    "naive": PRINT_LABELS_2D["naive"],
    "aware": r"seeds (interface-aware at $\delta = 0$)",
    "ablate": "ablation: normal-only seeds",
    "floor": "no edge (resolution floor)",
    "sfloor": "seed floor (no contrast)",
}
DEMO_LABELS_1D = {"aware": "seed stencils (ODE-continued)", "naive": LABELS["naive"]}


def _width_title(width: float, print_mode: bool) -> str:
    if print_mode:
        return r"jump ($\delta = 0$)" if width == 0 else rf"$\delta = {width:g}$"
    return "jump edges (delta = 0)" if width == 0 else f"edge width delta = {width:g}"


def _small(print_mode: bool) -> float:
    return 6.5 if print_mode else 9


def _errors(cache: ResultsCache, **filters) -> tuple[np.ndarray, np.ndarray]:
    recs = cache.select("error", **filters)
    return np.array([r["n"] for r in recs], float), np.array([r["error"] for r in recs])


def _guide(ax, ns, anchor, order_in_n, label, print_mode, at=1):
    """A dotted reference slope anchored at the ``at``-th point."""
    ref = anchor * (ns[at] / ns) ** order_in_n
    ax.loglog(ns, ref, ":", color=INK_MUTED, lw=1.0 if print_mode else 1.1)
    ax.annotate(
        label,
        (ns[-1], ref[-1]),
        xytext=(3, 0),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=_small(print_mode),
        va="center",
    )


def _knee_line(ax, n_knee, ns, print_mode, label_axes=True):
    if not (ns[0] <= n_knee <= ns[-1]):
        return
    ax.axvline(n_knee, color=INK_MUTED, lw=0.8 if print_mode else 1.0, ls=":")
    if label_axes:
        ax.annotate(
            r"$h = \delta$" if print_mode else "h = delta",
            (n_knee, 0.97),
            xycoords=("data", "axes fraction"),
            xytext=(3, 0),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=_small(print_mode),
            rotation=90,
            va="top",
        )


# --- 1-D convergence ----------------------------------------------------------------


def convergence_1d(
    cache: ResultsCache,
    out: Path,
    *,
    print_mode: bool = False,
    widths: list[float] | None = None,
) -> Path:
    """Error vs resolution per edge width, naive FD4 against the seeds
    (notes §2, ``wave1d_stiff_convergence``)."""
    if widths is None:
        widths = [w for w in cache.values("delta", "error") if w is not None]
    labels = PRINT_LABELS if print_mode else DEMO_LABELS_1D
    t_end = cache.args.get("t_end", 1.0)
    n_w = len(widths)
    if print_mode:
        cols = min(2, n_w)
        rows = ceil(n_w / cols)
        fig, panels = plt.subplots(
            rows,
            cols,
            figsize=(TEXTWIDTH_IN, 1.85 * rows + 0.35),
            sharex=True,
            sharey=True,
            constrained_layout=True,
            squeeze=False,
        )
        axes = list(panels.ravel())
        for ax in axes[n_w:]:
            ax.set_visible(False)
        axes = axes[:n_w]
    else:
        fig, panels = plt.subplots(
            1,
            n_w,
            figsize=(3.6 * n_w, 4.4),
            sharey=True,
            constrained_layout=True,
            squeeze=False,
        )
        axes = list(panels.ravel())
        cols = n_w
    ns = None
    for ax, width in zip(axes, widths, strict=True):
        errors = {}
        for mode in ("naive", "aware"):
            ns, errors[mode] = _errors(cache, delta=width, mode=mode, field="f")
            ax.loglog(ns, errors[mode], label=labels[mode], **STYLE_2D[mode])
        _guide(ax, ns, errors["aware"][1], 4, "4th order", print_mode)
        if width == 0 or 2 / ns[-2] > width:
            _guide(ax, ns, errors["naive"][1], 1, "1st order", print_mode)
        if width:
            _knee_line(ax, 2 / width, ns, print_mode)
        ax.set_title(
            _width_title(width, print_mode), fontsize=None if print_mode else 11
        )
        ax.set_xticks(ns, [str(int(n)) for n in ns])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(ns[0] / 1.3, ns[-1] * 2.4)
    for i, ax in enumerate(axes):
        if print_mode and i // cols == ceil(n_w / cols) - 1 or not print_mode:
            ax.set_xlabel("grid nodes $n$" if print_mode else "number of grid nodes")
        if i % cols == 0:
            ax.set_ylabel(
                rf"rel. error in $f$ at $t = {t_end:g}$"
                if print_mode
                else f"relative error in stress at t = {t_end:g}"
            )
    axes[0].set_ylim(1e-8, 5e-1)
    axes[0].legend(loc="lower left", fontsize=_small(print_mode))
    if not print_mode:
        fig.suptitle(
            "Same equispaced grid, same time step: only the stencils that see the "
            "edge differ",
            fontsize=12,
        )
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# --- 2-D convergence ----------------------------------------------------------------


def _grid_2d(n_w: int, print_mode: bool):
    """Axes for the v (top) and u (bottom) panels of every width.

    Demo: one column per width, two rows. Print: two columns of widths at
    the text width, each width a v panel over a u panel, y axes shared
    within the v panels and within the u panels.
    """
    if not print_mode:
        fig, panels = plt.subplots(
            2,
            n_w,
            figsize=(3.6 * n_w, 7.6),
            sharey="row",
            sharex=True,
            constrained_layout=True,
            squeeze=False,
        )
        return fig, list(panels[0]), list(panels[1]), n_w
    cols = min(2, n_w)
    blocks = ceil(n_w / cols)
    fig, panels = plt.subplots(
        2 * blocks,
        cols,
        figsize=(TEXTWIDTH_IN, 1.65 * 2 * blocks + 0.7),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes_v = [panels[2 * (i // cols), i % cols] for i in range(n_w)]
    axes_u = [panels[2 * (i // cols) + 1, i % cols] for i in range(n_w)]
    for i in range(n_w, blocks * cols):
        panels[2 * (i // cols), i % cols].set_visible(False)
        panels[2 * (i // cols) + 1, i % cols].set_visible(False)
    for group in (axes_v, axes_u):
        for ax in group[1:]:
            ax.sharey(group[0])
    for i, (ax_v, ax_u) in enumerate(zip(axes_v, axes_u, strict=True)):
        if i % cols:
            ax_v.tick_params(labelleft=False)
            ax_u.tick_params(labelleft=False)
    return fig, axes_v, axes_u, cols


def convergence_2d(
    cache: ResultsCache,
    out: Path,
    *,
    print_mode: bool = False,
    widths: list[float] | None = None,
) -> Path:
    """Error in v (top) and the u column (bottom) vs nodes, per edge width,
    every scheme in the cache plus the floors (notes §5.4–5.6,
    ``wave2d_stiff{_d12,_a0.02}``)."""
    args = cache.args
    if widths is None:
        widths = [w for w in cache.values("delta", "error") if w is not None]
    fields = cache.values("field", "error")
    u_field = "u" if "u" in fields else "max_u"
    u_error = u_field == "u"
    curved = bool(args.get("amplitude", 0.0))
    oblique = tuple(args.get("direction", (0, 1))) != (0, 1)
    t_end = args.get("t_end", 1.0)
    labels = PRINT_LABELS_2D_ALL if print_mode else DEMO_LABELS_2D
    ms = 3.5 if print_mode else 5
    n_w = len(widths)
    fig, axes_v, axes_u, cols = _grid_2d(n_w, print_mode)
    ns = None
    for i, (ax, ax_u, width) in enumerate(zip(axes_v, axes_u, widths, strict=True)):
        modes = [m for m in cache.values("mode", "error", delta=width) if m != "sfloor"]
        floors = ["floor"] + (
            ["sfloor"] if "sfloor" in cache.values("mode", "error", delta=width) else []
        )
        errors = {}
        for mode in modes + floors:
            delta = None if mode == "floor" else width
            ns, ev = _errors(cache, delta=delta, mode=mode, field="v")
            _, eu = _errors(cache, delta=delta, mode=mode, field=u_field)
            errors[mode] = (ev, eu)
            kw = dict(STYLE_2D[mode], ms=ms)
            if mode in floors:
                kw["lw"] = 1.0 if print_mode else 1.4
            ax.loglog(ns, ev, label=labels[mode], **kw)
            # The curved floors' u is spurious, not an error against the
            # curved solution, so the u panel leaves them out there.
            if mode not in floors or not curved:
                ax_u.loglog(ns, eu, **kw)
        _guide(ax, ns, errors["naive"][0][0], 1, "2nd order", print_mode, at=0)
        if "aware" in errors:
            _guide(ax, ns, errors["aware"][0][0], 2, "4th order", print_mode, at=0)
        if width:
            _knee_line(ax, width**-2, ns, print_mode)
            _knee_line(ax_u, width**-2, ns, print_mode, label_axes=False)
        ax.set_title(
            _width_title(width, print_mode), fontsize=None if print_mode else 11
        )
        ax_u.set_xticks(ns, [str(int(n)) for n in ns])
        ax_u.xaxis.set_minor_formatter(NullFormatter())
        ax_u.set_xlim(ns[0] / 1.3, ns[-1] * 2.4)
        last_block = i // cols == ceil(n_w / cols) - 1
        if not print_mode or last_block:
            ax_u.set_xlabel("nodes $n$" if print_mode else "number of nodes")
        if i % cols == 0:
            ax.set_ylabel(
                rf"rel. error in $v$ at $t = {t_end:g}$"
                if print_mode
                else f"relative error in v at t = {t_end:g}"
            )
            if u_error:
                ax_u.set_ylabel(
                    rf"rel. error in $u$ at $t = {t_end:g}$"
                    if print_mode
                    else f"relative error in u at t = {t_end:g}"
                )
            else:
                ax_u.set_ylabel(
                    r"max $|u|$ (exact: 0)" if print_mode else "max |u| (exact: 0)"
                )
    # The jump panel has no seed floor, so collect the legend across panels.
    entries: dict[str, object] = {}
    for ax in axes_v:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, handle)
    handles, leg_labels = list(entries.values()), list(entries)
    if print_mode or len(leg_labels) > 3:
        fig.legend(
            handles,
            leg_labels,
            loc="outside lower center",
            ncol=2,
            fontsize=_small(print_mode),
        )
    else:
        axes_v[0].legend(loc="lower left", fontsize=9)
    if not print_mode:
        title = "Same nodes, same time step: the edge is only as sharp as delta"
        if oblique:
            d = tuple(args["direction"])
            angle = float(np.degrees(np.arctan2(*d)))
            title += f"\nP train at {angle:.1f} deg to the normal, direction {d}"
        if curved:
            title += (
                f"\nsine interfaces of amplitude {args['amplitude']:g} (curved case)"
            )
        fig.suptitle(title, fontsize=12)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# --- the seed bases ---------------------------------------------------------------


def seeds_1d(out: Path, *, print_mode: bool = False, h: float = 0.01) -> Path:
    """phi_1..phi_4 for the u field on one stencil whose centre is h/2 left of
    the edge at x = 0: a jump, an edge of width h/4, and one of width 2h,
    against the monomials (notes §2, ``wave1d_stiff_seeds``)."""
    xe = -0.5 * h
    x_plot = np.linspace(-3 * h, 3 * h, 301)
    nodes = xe + h * np.arange(-2, 3)
    cases = [
        (r"jump ($\delta \to 0$)" if print_mode else "jump (delta -> 0)", 1e-6),
        (r"$\delta = h/4$" if print_mode else "delta = h/4", h / 4),
        (r"$\delta = 2h$" if print_mode else "delta = 2h", 2 * h),
    ]
    styles = [(INK, "-"), (COLORS["aware"], "-"), (COLORS["naive"], "-")]
    if print_mode:
        fig, panels = plt.subplots(
            2, 2, figsize=(TEXTWIDTH_IN, 0.8 * TEXTWIDTH_IN), constrained_layout=True
        )
    else:
        fig, panels = plt.subplots(1, 4, figsize=(13, 3.6), constrained_layout=True)
    axes = list(panels.ravel())
    xi = (x_plot - xe) / h
    lw = 1.2 if print_mode else 1.8
    for k, ax in enumerate(axes, start=1):
        for node in nodes:
            ax.axvline((node - xe) / h, color=INK_MUTED, lw=0.5, ls=":")
        ax.axvline(-xe / h, color=INK_SECONDARY, lw=0.8)
        ax.plot(
            xi,
            xi**k,
            color=INK_MUTED,
            lw=0.9 if print_mode else 1.2,
            ls="--",
            label=f"monomial $\\xi^{k}$",
        )
        curves = [xi**k]
        for (label, width), (color, ls) in zip(cases, styles, strict=True):
            medium = LayeredMedium(edge_width=width)
            a_u, _, h_s = seed_basis(x_plot, xe, medium, 5)
            phi = a_u[k] * (h_s / h) ** k
            ax.plot(xi, phi, color=color, ls=ls, lw=lw, label=label)
            curves.append(phi)
        ax.set_title(
            rf"seed $\phi_{k}$ ($u$ field)"
            if print_mode
            else f"seed $\\phi_{k}$ (u field)",
            fontsize=None if print_mode else 11,
        )
        ax.set_xlim(-3, 3)
        # Frame the stencil's reach, |xi| <= 2.5, whatever the seeds do beyond it.
        inside = np.abs(xi) <= 2.5
        ys = np.concatenate([c[inside] for c in curves])
        span = ys.max() - ys.min()
        ax.set_ylim(ys.min() - 0.05 * span, ys.max() + 0.05 * span)
    for i, ax in enumerate(axes):
        if not print_mode or i >= 2:
            ax.set_xlabel(r"$\xi = (x - x_e) / h$")
    axes[0].legend(fontsize=_small(print_mode), loc="upper left")
    if not print_mode:
        fig.suptitle(
            "Seeds look like monomials at the evaluation point and are continued "
            "through the edge by the PDE (c: 1 -> 2, rho: 1 -> 1); edge at "
            r"$\xi = 0.5$",
            fontsize=11,
        )
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# (component the seed is a monomial of, its exponents (a, b) in (X, Y), the
# field drawn (u, v, f, g, h), the X at which the cross-section is taken, the
# title). The comparator of every member is the same seed in the anchor
# material frozen through the stencil: the monomial itself for a velocity
# field, the monomial's stress rate for a stress field.
SEED_MEMBERS_2D = (
    ("u", (0, 1), "u", 0.0, r"$u$ of the seed $u = Y$"),
    ("v", (0, 2), "v", 0.0, r"$v$ of the seed $v = Y^2$"),
    ("u", (0, 2), "g", 0.0, r"shear traction $g$ of the seed $u = Y^2$"),
    ("v", (1, 1), "u", 0.5, r"$u$ induced by the seed $v = XY$ ($X = 1/2$)"),
)
FIELDS_2D = ("u", "v", "f", "g", "h")


def seed_sections_2d(
    edge_width: float,
    y_grid: np.ndarray,
    *,
    h: float = 0.02,
    degree: int = 3,
    anchor_offset: float = -0.5,
    x0: float = 0.3,
    frozen: bool = False,
) -> dict[tuple[str, tuple[int, int], str, float], np.ndarray]:
    """Cross-sections of the elastic seeds of a flat edge in the normal
    coordinate ``Y = (y' - y'_e) / r_max``, for every member of
    :data:`SEED_MEMBERS_2D`: the anchor is ``anchor_offset * h`` from the
    lower edge (in the background for a negative offset), ``r_max = 2.5 h``
    the stencil radius of a 19-node stencil, and the section is taken at
    the member's ``X``. ``frozen`` freezes the material at the anchor, which
    turns the seeds into the monomials (and their stresses): the comparator."""
    medium = LayeredMedium2D(edge_width=edge_width)
    profile = normal_profile(medium, medium.lower, x0)
    y_e, scale = anchor_offset * h, 2.5 * h
    if frozen:
        profile = frozen_profile(profile, y_e)
    chain = SeedChain(profile, y_e, scale, degree)
    exps = [tuple(int(v) for v in e) for e in chain.exps]
    out = {}
    for comp, ab, draw, x_at, _ in SEED_MEMBERS_2D:
        e = ("u", "v").index(comp) * chain.m + exps.index(ab)
        fields = chain.evaluate(np.full_like(y_grid, x_at), y_grid)
        out[(comp, ab, draw, x_at)] = fields[FIELDS_2D.index(draw)][:, e]
    return out


def seeds_2d(out: Path, *, print_mode: bool = False, h: float = 0.02) -> Path:
    """Cross-sections of the elastic seeds of a straight edge in the normal
    coordinate: the jump limit, an edge of width h/4 and one of width 2h,
    against the monomials, for the members of :data:`SEED_MEMBERS_2D`
    (the manuscript's figure of the 2-D basis, #54)."""
    y_grid = np.linspace(-1.2, 1.2, 241)
    cases = [
        (r"jump ($\delta \to 0$)" if print_mode else "jump (delta -> 0)", 1e-6),
        (r"$\delta = h/4$" if print_mode else "delta = h/4", h / 4),
        (r"$\delta = 2h$" if print_mode else "delta = 2h", 2 * h),
    ]
    styles = [(INK, "-"), (COLORS["aware"], "-"), (COLORS["naive"], "-")]
    sections = {w: seed_sections_2d(w, y_grid, h=h) for _, w in cases}
    mono = seed_sections_2d(cases[0][1], y_grid, h=h, frozen=True)
    y_edge = 0.5 * h / (2.5 * h)  # the anchor is h/2 below the edge
    if print_mode:
        fig, panels = plt.subplots(
            2, 2, figsize=(TEXTWIDTH_IN, 0.8 * TEXTWIDTH_IN), constrained_layout=True
        )
    else:
        fig, panels = plt.subplots(1, 4, figsize=(13, 3.6), constrained_layout=True)
    axes = list(panels.ravel())
    lw = 1.2 if print_mode else 1.8
    for ax, member in zip(axes, SEED_MEMBERS_2D, strict=True):
        comp, (a, b), draw, x_at, title = member
        for x in (-1, 1):
            ax.axvline(x, color=INK_MUTED, lw=0.5, ls=":")
        ax.axvline(0, color=INK_MUTED, lw=0.5, ls=":")
        ax.axvline(y_edge, color=INK_SECONDARY, lw=0.8)
        mono_curve = mono[(comp, (a, b), draw, x_at)]
        ax.plot(
            y_grid,
            mono_curve,
            color=INK_MUTED,
            lw=0.9 if print_mode else 1.2,
            ls="--",
            label="monomial",
        )
        curves = [mono_curve]
        for (label, width), (color, ls) in zip(cases, styles, strict=True):
            phi = sections[width][(comp, (a, b), draw, x_at)]
            ax.plot(y_grid, phi, color=color, ls=ls, lw=lw, label=label)
            curves.append(phi)
        ax.set_title(title, fontsize=None if print_mode else 10)
        ax.set_xlim(-1.2, 1.2)
        inside = np.abs(y_grid) <= 1.05
        ys = np.concatenate([c[inside] for c in curves])
        span = max(ys.max() - ys.min(), 1e-3)
        ax.set_ylim(ys.min() - 0.08 * span, ys.max() + 0.08 * span)
    for i, ax in enumerate(axes):
        if not print_mode or i >= 2:
            ax.set_xlabel(r"$Y = (y' - y'_e) / r_{\max}$")
    axes[0].legend(fontsize=_small(print_mode), loc="upper left")
    if not print_mode:
        fig.suptitle(
            "Elastic seeds of a straight edge along the normal through the anchor "
            "(background below, band above; lam, mu: 1 -> 4, rho: 1 -> 2); "
            f"edge at Y = {y_edge:g}, stencil reach |Y| <= 1",
            fontsize=11,
        )
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
