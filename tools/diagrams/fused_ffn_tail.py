"""Tier-1 static diagram for `fused_ffn_tail`.

Generator lives under `tools/diagrams/`, rendered artifacts under
`docs/diagrams/`. Run from repo root:

    python tools/diagrams/fused_ffn_tail.py
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "diagrams"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = _OUT_DIR / "fused_ffn_tail.svg"

C_X = "#cfe2ff"
C_ROW = "#6ea8fe"
C_TH = "#fff3cd"
C_RED = "#f1aeb5"
C_OP = "#d1e7dd"
C_OUT = "#c8b6e2"
C_EDGE = "#333"
C_DIM = "#888"


def draw_grid(ax, x0, y0, w, h, cols, rows, face, label_row=None):
    cw, ch = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            fc = C_ROW if (label_row is not None and r == label_row) else face
            ax.add_patch(
                mpatches.Rectangle(
                    (x0 + c * cw, y0 + (rows - 1 - r) * ch),
                    cw, ch, facecolor=fc, edgecolor=C_EDGE, linewidth=0.5,
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
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=face, edgecolor=C_EDGE, linewidth=1.0,
        )
    )
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize)


def main():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        6, 6.65, "fused_ffn_tail — one block per row of x[M, H]",
        ha="center", va="center", fontsize=13, fontweight="bold",
    )

    # ---- Left: input tensor [M, H] with one row highlighted ----
    draw_grid(ax, 0.3, 3.0, 2.6, 2.8, cols=10, rows=8, face=C_X, label_row=3)
    ax.text(1.6, 5.95, "x : [M, H]", ha="center", fontsize=10)
    ax.text(1.6, 2.78, "row = blockIdx.x", ha="center", fontsize=8, color="#333")

    # Arrow row -> thread-stride view.
    arrow(ax, (2.95, 4.4), (3.55, 4.4))

    # ---- Middle: thread-stride view of the active row ----
    # 12 cells: first 3 explicitly labelled tid=0..2 ABOVE, last shows ellipsis.
    tcols = 12
    tx0, ty0, tw, th = 3.6, 4.2, 3.2, 0.4
    draw_grid(ax, tx0, ty0, tw, th, cols=tcols, rows=1, face=C_TH)
    # tid labels above the cells.
    cw = tw / tcols
    for c, lbl in [(0, "tid=0"), (1, "1"), (2, "2"), (tcols - 1, "...")]:
        ax.text(tx0 + (c + 0.5) * cw, ty0 + th + 0.05, lbl,
                ha="center", va="bottom", fontsize=7.5, color="#333")
    # Caption below the row (out of the way of the threads->pass1 arrow).
    ax.text(tx0 + tw / 2, ty0 - 0.25,
            "256 threads stride the row:  for (j = tid; j < H; j += 256)",
            ha="center", fontsize=9, color="#333")

    # ---- Pass 1: reduction (top-right) ----
    ax.text(9.4, 5.95, "PASS 1 — reduction", ha="center",
            fontsize=10.5, fontweight="bold")
    box(ax, 7.0, 5.25, 1.7, 0.5, "ss_t = Σ x²\n(per thread)", C_TH, fontsize=8.5)
    arrow(ax, (8.7, 5.5), (9.05, 5.5))
    box(ax, 9.05, 5.25, 2.05, 0.5,
        "block_sum:\nsmem-tree + warp-shfl", C_RED, fontsize=8.5)
    arrow(ax, (11.1, 5.5), (11.5, 5.5))
    ax.text(11.55, 5.5, "rrms", va="center", fontsize=10, fontweight="bold")

    # Threads -> pass 1: leave from the TOP-right of the threads row, into the
    # left edge of the ss_t box.
    arrow(ax, (tx0 + tw, ty0 + th + 0.25), (7.0, 5.4),
          connectionstyle="arc3,rad=-0.2")

    # ---- Pass 2: elementwise pipeline (bottom) ----
    ax.text(6, 3.0, "PASS 2 — elementwise", ha="center",
            fontsize=10.5, fontweight="bold")
    y2 = 1.95
    pipe = [
        ("x_j", C_X),
        ("× rrms", C_OP),
        ("× γ_j", C_OP),
        ("+ b_j", C_OP),
        ("act(·)", C_OP),
        ("+ r_j", C_OP),
        ("y_j", C_OUT),
    ]
    bw, gap = 1.15, 0.18
    total_w = len(pipe) * bw + (len(pipe) - 1) * gap
    x_start = (12 - total_w) / 2
    centers_x = []
    for i, (lbl, face) in enumerate(pipe):
        x = x_start + i * (bw + gap)
        centers_x.append(x + bw / 2)
        box(ax, x, y2, bw, 0.65, lbl, face, fontsize=9.5)
        if i > 0:
            arrow(ax, (x - gap, y2 + 0.325), (x, y2 + 0.325))

    # Optional-stage callouts under +b_j and +r_j.
    for ci, txt in [(3, "optional\n(bias=None → skipped)"),
                    (5, "optional\n(residual=None → skipped)")]:
        ax.annotate(
            txt,
            xy=(centers_x[ci], y2),
            xytext=(centers_x[ci], y2 - 0.95),
            ha="center", fontsize=7.5, color="#444",
            arrowprops=dict(arrowstyle="-", color="#888", linewidth=0.6),
        )

    # rrms feed-in: from the rrms text down into the "× rrms" box.
    arrow(ax, (11.55, 5.35), (11.55, 4.6),
          color=C_DIM, linewidth=0.9)
    arrow(ax, (11.55, 4.6), (centers_x[1], y2 + 0.65),
          connectionstyle="arc3,rad=0.25", color=C_DIM, linewidth=0.9)
    ax.text(11.7, 4.95, "rrms\nbroadcast\nvia smem[0]",
            fontsize=7.5, color="#555", va="center", ha="left")

    # ---- Footer ----
    ax.text(
        0.3, 0.4,
        "Block (256,)  ·  Grid (M,)  ·  Smem 256·float  ·  intermediates in registers, fp32 accum even on fp16 path",
        fontsize=9, color="#333",
    )
    ax.text(
        0.3, 0.1,
        "Two passes over H: pass 1 reads x to get rrms; pass 2 re-reads x and folds γ, bias, residual into y.",
        fontsize=9, color="#333",
    )

    fig.savefig(OUT, format="svg", bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), format="png", dpi=160, bbox_inches="tight")
    print(f"wrote {OUT} and {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
