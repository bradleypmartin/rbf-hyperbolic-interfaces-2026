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

import os

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


#: The manuscript's text width in inches (amsart 11pt: \textwidth = 30 pc). Print
#: figures are rendered at this width so their fonts land at true size and
#: ``\includegraphics`` in paper/main.tex uses them at natural width.
TEXTWIDTH_IN = 4.98

#: The scheme labels the manuscript uses (the deck's are in ``LABELS``).
PRINT_LABELS = {"aware": "seeds", "naive": "standard FD4"}
PRINT_LABELS_2D = {"aware": "seeds", "naive": "naive RBF-FD"}


def use_print_style() -> None:
    """Print chrome for the manuscript's figures (#54): white background,
    serif with Computer Modern mathtext, 8 pt, thin recessive axes. The
    palette (blue = seeds / interface-aware, orange = naive, the aqua and
    violet maps) is unchanged; only the chrome differs from the deck.

    Pins matplotlib's PDF ``CreationDate`` through ``SOURCE_DATE_EPOCH`` so a
    regenerated figure is byte-identical to the committed one, the figure
    analogue of ``main.tex``'s fixed ``\\date``. A hard assignment, not
    ``setdefault``: an inherited value would silently break the guarantee.
    matplotlib reads the variable when the PDF is written, so calling this
    any time before ``savefig`` is enough.
    """
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.size": 8.0,
            "mathtext.fontset": "cm",
            "axes.titlesize": 8.0,
            "axes.titleweight": "normal",
            "axes.titlecolor": INK,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "figure.dpi": 200,
        }
    )


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
