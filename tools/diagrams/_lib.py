"""Shared helpers for tier-1 algorithm diagrams.

Each kernel gets its own script under `tools/diagrams/<kernel>.py` that
imports from here. Outputs land in `docs/diagrams/<kernel>.{svg,png}`.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared palette so all diagrams read as a set.
C_IN = "#cfe2ff"  # input tensor cells
C_HI = "#6ea8fe"  # highlighted region
C_TH = "#fff3cd"  # thread / per-thread work
C_RED = "#f1aeb5"  # reduction / sync barrier
C_OP = "#d1e7dd"  # compute op
C_OUT = "#c8b6e2"  # output
C_SMEM = "#ffe5d0"  # shared memory
C_REG = "#e6dffb"  # registers
C_EDGE = "#333"
C_DIM = "#888"


def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def title(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=13, fontweight="bold")


def grid(ax, x0, y0, w, h, cols, rows, face, highlight=None, ec_lw=0.5):
    """`highlight` may be a (r, c) tuple, a list of those, or None."""
    cw, ch = w / cols, h / rows
    hi = set()
    if highlight is not None:
        if isinstance(highlight, tuple) and isinstance(highlight[0], int):
            hi = {highlight}
        else:
            hi = set(highlight)
    for r in range(rows):
        for c in range(cols):
            fc = C_HI if (r, c) in hi else face
            ax.add_patch(
                mpatches.Rectangle(
                    (x0 + c * cw, y0 + (rows - 1 - r) * ch),
                    cw,
                    ch,
                    facecolor=fc,
                    edgecolor=C_EDGE,
                    linewidth=ec_lw,
                )
            )


def arrow(ax, p0, p1, **kw):
    kw.setdefault("arrowstyle", "->")
    kw.setdefault("mutation_scale", 14)
    kw.setdefault("color", C_EDGE)
    kw.setdefault("linewidth", 1.2)
    ax.add_patch(FancyArrowPatch(p0, p1, **kw))


def box(ax, x, y, w, h, label, face, fontsize=9):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=face,
            edgecolor=C_EDGE,
            linewidth=1.0,
        )
    )
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize)


def label(ax, x, y, text, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("fontsize", 9)
    kw.setdefault("color", "#333")
    ax.text(x, y, text, **kw)


def footer(ax, x, y, lines):
    for i, line in enumerate(lines):
        ax.text(x, y - i * 0.3, line, fontsize=9, color="#333")


def save(fig, name):
    svg = OUT_DIR / f"{name}.svg"
    png = OUT_DIR / f"{name}.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=160, bbox_inches="tight")
    print(f"wrote {svg} and {png}")
