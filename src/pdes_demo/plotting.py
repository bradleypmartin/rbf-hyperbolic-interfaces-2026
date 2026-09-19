"""Shared matplotlib styling for demo figures and animations.

Colours follow a validated two-series palette (blue for the interface-aware
method, orange for the naive one); reference curves and annotations use
neutral inks so colour only ever encodes "which method".

The 2-D colour maps use two single-hue sequential ramps, light to dark:
aqua for a field's magnitude and violet for its error. Neither hue is used
anywhere else in the deck, so on a 2-D figure colour encodes the quantity
while blue and orange keep meaning "which method" on the line plots around
it. Both ramps were derived from the palette's blue ramp by keeping each
step's OKLCH lightness (and chroma, where sRGB allows) and rotating the hue
to the palette's aqua and violet, so all three share one lightness ladder
and the eye reads magnitudes off any of them the same way.
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

_AQUA_RAMP = [
    "#c9e9d7", "#afddc5", "#93d2b2", "#76c6a0", "#53bb8d", "#27af7b", "#059f6d",
    "#078f62", "#077f57", "#056f4b", "#016040", "#005136", "#02432c",
]  # fmt: skip
_VIOLET_RAMP = [
    "#dbddfb", "#cbccf6", "#babbf4", "#aaaaee", "#9a99eb", "#8c88e6", "#7d75e3",
    "#6f67d4", "#625bbd", "#564ea9", "#4a4393", "#3e377f", "#322d6a",
]  # fmt: skip

# Both start at the surface colour so "zero" recedes into the background.
FIELD_CMAP = LinearSegmentedColormap.from_list("demo_field", [SURFACE, *_AQUA_RAMP])
ERROR_CMAP = LinearSegmentedColormap.from_list("demo_error", [SURFACE, *_VIOLET_RAMP])


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
