"""Shared matplotlib styling for demo figures and animations.

Colours follow a validated two-series palette (blue for the interface-aware
method, orange for the naive one); reference curves and annotations use
neutral inks so colour only ever encodes "which method".

The 2-D colour maps use two single-hue sequential ramps from the same
palette, light to dark: blue for a field's magnitude and orange for its
error. The orange steps were derived from the blue ones by keeping each
step's OKLCH lightness and moving its hue to the palette's orange, so the
two ramps have the same lightness ladder and the eye reads magnitudes off
either the same way.
"""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

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

_BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]  # fmt: skip
_ORANGE_RAMP = [
    "#fbd7ca", "#f5c4b2", "#f2b098", "#eb9c7f", "#e68764", "#df7249", "#d95923",
    "#c94908", "#b44005", "#9f3600", "#8a2e00", "#742702", "#611e00",
]  # fmt: skip

# Both start at the surface colour so "zero" recedes into the background.
FIELD_CMAP = LinearSegmentedColormap.from_list("demo_field", [SURFACE, *_BLUE_RAMP])
ERROR_CMAP = LinearSegmentedColormap.from_list("demo_error", [SURFACE, *_ORANGE_RAMP])


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
