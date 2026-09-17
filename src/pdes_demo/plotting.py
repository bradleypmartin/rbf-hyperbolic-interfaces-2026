"""Shared matplotlib styling for demo figures and animations.

Colours follow a validated two-series palette (blue for the interface-aware
method, orange for the naive one); reference curves and annotations use
neutral inks so colour only ever encodes "which method".
"""

import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"
LAYER_FILL = "#f0efec"
AWARE = "#2a78d6"
NAIVE = "#eb6834"

LABELS = {"aware": "interface-aware FD", "naive": "standard FD (naive)"}
COLORS = {"aware": AWARE, "naive": NAIVE}


def use_demo_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "text.color": INK,
            "font.size": 12,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
        }
    )
