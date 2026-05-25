"""Tier-1 diagram for `reduction_v2`."""

from _lib import (
    C_HI,
    C_IN,
    C_OUT,
    C_RED,
    C_TH,
    arrow,
    box,
    footer,
    grid,
    label,
    new_fig,
    save,
    title,
)


def main():
    fig, ax = new_fig(12, 7)
    title(ax, 6, 6.65, "reduction_v2 — ITEMS-per-thread fan-in + smem-tree + atomicAdd")

    # ---- Input N-length array, partitioned by grid ----
    grid(
        ax,
        0.3,
        4.7,
        6.0,
        0.5,
        cols=24,
        rows=1,
        face=C_IN,
        highlight=[(0, c) for c in range(8)],
    )
    label(
        ax,
        3.3,
        5.45,
        "input[N] — partitioned into grid blocks of THREADS·ITEMS elements",
        fontsize=9,
    )
    label(ax, 0.65, 4.55, "block 0", fontsize=8)
    label(ax, 2.65, 4.55, "block 1", fontsize=8)
    label(ax, 4.65, 4.55, "block 2", fontsize=8)
    label(ax, 6.05, 4.55, "...", fontsize=8)

    # Highlight one block, expand it: 8 threads × ITEMS=8 cells (illustrative)
    # We show 8 thread "stripes", each owning ITEMS=8 stridden elements (compressed: 4 shown).
    label(
        ax,
        6,
        3.85,
        "one block: THREADS threads each accumulate ITEMS elements (strided)",
        fontsize=9.5,
    )
    sx, sy, sw, sh = 0.8, 3.05, 10.4, 0.55
    cols = 32
    grid(ax, sx, sy, sw, sh, cols=cols, rows=1, face=C_IN)
    # Colour stripes by thread (0..7), each thread owns columns [t, t+8, t+16, t+24].
    cw = sw / cols
    palette = [C_TH, C_HI, C_IN, C_HI, C_TH, C_HI, C_IN, C_HI]
    for t in range(8):
        for k in range(4):
            c = t + k * 8
            ax.add_patch(
                __import__("matplotlib").patches.Rectangle(
                    (sx + c * cw, sy),
                    cw,
                    sh,
                    facecolor=palette[t],
                    edgecolor="#333",
                    linewidth=0.4,
                )
            )
    label(ax, sx + 0.5 * cw, sy - 0.2, "t=0", fontsize=7)
    label(ax, sx + 1.5 * cw, sy - 0.2, "1", fontsize=7)
    label(ax, sx + 7.5 * cw, sy - 0.2, "7", fontsize=7)
    label(ax, sx + 8.5 * cw, sy - 0.2, "t=0 (k=1)", fontsize=7)
    label(ax, sx + 16.5 * cw, sy - 0.2, "t=0 (k=2)", fontsize=7)
    label(ax, sx + 24.5 * cw, sy - 0.2, "t=0 (k=3)", fontsize=7)

    # ---- Per-thread partial sums ----
    py = 1.7
    pw = 1.05
    for i in range(8):
        x = 0.9 + i * (pw + 0.1)
        box(ax, x, py, pw, 0.5, f"partial[{i}]", C_TH, fontsize=8)
    arrow(ax, (sx + sw / 2, sy - 0.45), (sx + sw / 2, py + 0.55))
    label(
        ax,
        sx + sw / 2 + 1.5,
        sy - 0.7,
        "ITEMS strided loads -> per-thread sum",
        fontsize=8.5,
        color="#444",
    )

    # ---- Smem-tree reduction + warp shuffle ----
    box(
        ax, 0.9, 0.9, 4.2, 0.5, "smem-tree reduction (THREADS -> 32)", C_RED, fontsize=9
    )
    box(ax, 5.3, 0.9, 3.0, 0.5, "warp shuffle (32 -> 1)", C_RED, fontsize=9)
    box(ax, 8.5, 0.9, 2.7, 0.5, "atomicAdd → output", C_OUT, fontsize=9)
    arrow(ax, (5.1, 1.15), (5.3, 1.15))
    arrow(ax, (8.3, 1.15), (8.5, 1.15))
    # Partial-sums -> smem-tree
    arrow(ax, (3.0, py), (3.0, 1.4))

    footer(
        ax,
        0.3,
        0.45,
        [
            "Knobs swept in bench_reduction_v2.py: THREADS ∈ {128,256,512,1024}, ITEMS=8 (winner on Turing).",
            "ITEMS>1 amortises smem-tree cost over more bytes and shrinks the grid (less atomicAdd contention).",
        ],
    )

    save(fig, "reduction_v2")


if __name__ == "__main__":
    main()
