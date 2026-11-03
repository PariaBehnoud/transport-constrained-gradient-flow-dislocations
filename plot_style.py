"""Shared figure style for the Section 7 figures."""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots in fixed order (validated palette, light surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#6250d6", "#e34948"]
# Diverging pair for the sign of the applied stress, with a neutral gray.
STRESS_NEG = "#2a78d6"
STRESS_ZERO = "#52514e"
STRESS_POS_LIGHT = "#e87170"
STRESS_POS = "#a82525"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

LABEL_FS = 26
TICK_FS = 22
LEGEND_FS = 18


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": LABEL_FS,
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,
        "legend.fontsize": LEGEND_FS,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.4,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
