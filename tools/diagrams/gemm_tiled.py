"""Tier-1 diagram for `gemm_tiled` (128×128 macro × 8×8 register, double-buffered)."""

from _lib import (
    C_HI,
    C_IN,
    C_OP,
    C_OUT,
    C_REG,
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
    fig, ax = new_fig(13, 7.5)
    title(
        ax,
        6.5,
        7.15,
        "gemm_tiled — 128×128 macro-tile, 8×8 register micro-tile, double-buffered K-stream",
    )

    # ---- A (M×K), B (K×N), C (M×N) ----
    grid(
        ax,
        0.3,
        3.6,
        1.8,
        3.0,
        cols=4,
        rows=6,
        face=C_IN,
        highlight=[(2, 0), (2, 1), (2, 2), (2, 3)],
    )
    label(ax, 1.2, 6.75, "A [M, K]", fontsize=10)
    label(ax, 1.2, 3.45, "K-row strip", fontsize=8, color="#222")

    grid(ax, 4.3, 6.7, 3.0, 1.0, cols=6, rows=2, face=C_IN, highlight=[(0, 3), (1, 3)])
    label(ax, 5.8, 7.85, "B [K, N]", fontsize=10)

    grid(ax, 4.3, 3.6, 3.0, 3.0, cols=6, rows=6, face=C_OUT, highlight=[(2, 3)])
    label(ax, 5.8, 6.75, "C [M, N]   — block owns one 128×128 macro-tile", fontsize=10)
    label(ax, 5.8, 3.45, "blockIdx = (bx, by)", fontsize=8.5)

    # K direction labels
    label(ax, 1.2, 3.15, "K -> stream of K/BK tiles", fontsize=8.5, color="#333")
    label(
        ax,
        5.8,
        3.15,
        "macro-tile filled by accumulating across K",
        fontsize=8.5,
        color="#333",
    )

    # ---- Double-buffer smem schematic ----
    box(
        ax,
        8.2,
        5.55,
        4.5,
        1.1,
        "smem  As[2][BK][BM]  Bs[2][BK][BN]\ndouble-buffered: compute on buf,\nLDG next tile into buf^1",
        C_SMEM,
        fontsize=9,
    )

    # ---- Per-thread register micro-tile ----
    box(
        ax,
        8.2,
        4.05,
        4.5,
        1.1,
        "registers  Creg[8][8]\neach thread owns an 8×8 output sub-tile\n(16×16 thread grid → 128×128 macro)",
        C_REG,
        fontsize=9,
    )

    # ---- FFMA pipeline ----
    box(
        ax,
        8.2,
        2.55,
        4.5,
        1.1,
        "inner: for kk in BK:\n  load Areg[8], Breg[8] from smem\n  Creg[i][j] += Areg[i] * Breg[j]   (FFMA × 64)",
        C_OP,
        fontsize=9,
    )

    # ---- Sync ----
    box(ax, 9.0, 1.55, 2.9, 0.45, "__syncthreads() between buffers", C_RED, fontsize=9)

    # ---- Arrows: A/B into smem, smem -> regs, FFMA -> regs ----
    arrow(
        ax,
        (2.15, 5.1),
        (8.2, 6.1),
        color="#666",
        linewidth=1.0,
        connectionstyle="arc3,rad=-0.15",
    )
    arrow(
        ax,
        (7.3, 7.2),
        (8.2, 6.4),
        color="#666",
        linewidth=1.0,
        connectionstyle="arc3,rad=-0.1",
    )
    arrow(ax, (10.45, 5.55), (10.45, 5.15))
    arrow(ax, (10.45, 4.05), (10.45, 3.65))
    arrow(ax, (10.45, 2.55), (10.45, 2.0))

    # ---- Epilogue arrow back to C ----
    arrow(
        ax,
        (8.2, 4.6),
        (7.3, 5.0),
        color="#888",
        linewidth=1.0,
        connectionstyle="arc3,rad=0.2",
    )
    label(ax, 7.7, 4.5, "epilogue:\nCreg → C tile", fontsize=8, color="#444")

    footer(
        ax,
        0.3,
        1.0,
        [
            "BM=BN=128, BK=8, TM=TN=8. 256 threads/block, 16×16 grid of micro-tiles.",
            "All global loads are float4 (LDG.E.128). A transposed into smem so the kk-loop reads M-major contiguous.",
            "Double-buffer mirrors a cp.async pipeline without sm_80+ instructions — runs on Turing.",
        ],
    )

    save(fig, "gemm_tiled")


if __name__ == "__main__":
    main()
