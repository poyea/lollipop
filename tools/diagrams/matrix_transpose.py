"""Tier-1 diagram for `matrix_transpose` (32x32 tiled, smem-padded)."""

from _lib import (
    C_IN,
    C_OUT,
    C_RED,
    C_SMEM,
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
    fig, ax = new_fig(13, 7)
    title(
        ax,
        6.5,
        6.65,
        "matrix_transpose — 32×32 tile via smem, +1 padding kills bank conflicts",
    )

    # ---- Input matrix ----
    grid(ax, 0.3, 3.2, 3.6, 3.0, cols=6, rows=5, face=C_IN, highlight=[(1, 2)])
    label(ax, 2.1, 6.35, "input [H, W]", fontsize=10)
    label(ax, 2.5, 4.2, "32×32 tile", fontsize=8, color="#222")
    label(ax, 2.1, 3.05, "blockIdx = (bx, by)", fontsize=8.5)

    # ---- Smem tile (the staging area) ----
    box(
        ax,
        4.6,
        4.0,
        3.8,
        1.7,
        "smem tile[32][32+1]\n+1 padding -> stride 33\n-> no bank conflicts",
        C_SMEM,
        fontsize=9.5,
    )

    # ---- Output matrix (transposed location) ----
    grid(ax, 9.1, 3.2, 3.6, 3.0, cols=5, rows=6, face=C_OUT, highlight=[(2, 1)])
    label(ax, 10.9, 6.35, "output [W, H]", fontsize=10)
    label(ax, 10.9, 3.05, "tile lands at (by, bx)", fontsize=8.5)

    # ---- Arrows: read coalesced, sync, write coalesced ----
    arrow(ax, (3.95, 4.85), (4.6, 4.85), linewidth=1.6)
    label(ax, 4.28, 5.15, "coalesced\nread", fontsize=8, color="#222")
    arrow(ax, (8.4, 4.85), (9.05, 4.85), linewidth=1.6)
    label(ax, 8.72, 5.15, "coalesced\nwrite", fontsize=8, color="#222")

    # ---- Sync barrier under the smem ----
    box(ax, 5.3, 3.0, 2.4, 0.45, "__syncthreads()", C_RED, fontsize=9)

    # ---- Per-thread mapping note ----
    label(
        ax,
        6.5,
        2.4,
        "Block (32, 8) — 32 threads × 8 rows; each thread handles 4 rows (32/8).",
        fontsize=9,
        color="#222",
    )
    label(
        ax,
        6.5,
        2.05,
        "Write path reads tile[threadIdx.x][threadIdx.y+j] (col-walk in smem).",
        fontsize=9,
        color="#222",
    )
    label(
        ax,
        6.5,
        1.7,
        "Without +1 pad, that col-walk collides 32-way on one bank; with +1, each thread hits a distinct bank.",
        fontsize=9,
        color="#222",
    )

    footer(
        ax,
        0.3,
        0.9,
        [
            "Naive transpose: coalesced read, scattered write (32-element gaps in DRAM) -> ~5× slower.",
            "Smem tile lets both directions be coalesced; the +1 pad is the load-bearing trick.",
            "matrix_transpose_naive and matrix_transpose_nopad are kept for parity / bank-conflict comparison.",
        ],
    )

    save(fig, "matrix_transpose")


if __name__ == "__main__":
    main()
