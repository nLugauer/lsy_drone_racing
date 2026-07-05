"""Shared matplotlib styling for publication-quality report figures.

Only matplotlib is used here (no seaborn). Import :func:`apply_style` once at
the top of any plotting script, and use :data:`COL_WIDTH` / :data:`DBL_WIDTH`
for IEEE single- and double-column figure widths.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# IEEE two-column layout: usable single-column ~3.5 in, full width ~7.16 in.
COL_WIDTH = 3.5
DBL_WIDTH = 7.16

# Colour-blind-friendly, print-safe palette (Okabe-Ito subset).
COLORS = {
    "pmm": "#0072B2",       # blue
    "baseline": "#D55E00",  # vermillion
    "accent": "#009E73",    # green
    "warn": "#CC79A7",      # purple
    "grey": "#666666",
}


def apply_style() -> None:
    """Set global rcParams for consistent, paper-ready figures."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
    })


def save(fig: "plt.Figure", out_dir: str, name: str) -> None:
    """Save a figure as both PDF (for LaTeX) and PNG (for quick preview)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
