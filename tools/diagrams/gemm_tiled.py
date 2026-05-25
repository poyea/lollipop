"""Tier-1 diagram for `gemm_tiled` (128×128 macro × 8×8 register, double-buffered)."""

from _lib import (
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
    H = 8.5
    fig, ax = new_fig(13, H)
    title(
        ax,
        6.5,
        H - 0.4,
        "gemm_tiled — 128×128 macro-tile, 8×8 register micro-tile, double-buffered K-stream",
    )

    # ---- A (top-left): tall, K-strip highlighted ----
    grid(
        ax,
        0.3,
        3.0,
        1.8,
        3.6,
        cols=4,
        rows=6,
        face=C_IN,
        highlight=[(2, 0), (2, 1), (2, 2), (2, 3)],
    )
    label(ax, 1.2, 6.8, "A [M, K]", fontsize=10)
    label(ax, 1.2, 2.8, "K-row strip", fontsize=8, color="#222")
    label(ax, 1.2, 2.55, "K → K/BK tiles", fontsize=8, color="#333")

    # ---- B (top-middle, sits above C) ----
    grid(ax, 2.6, 6.05, 3.0, 0.6, cols=6, rows=1, face=C_IN, highlight=[(0, 3)])
    label(ax, 4.1, 6.8, "B [K, N]", fontsize=10)

    # ---- C (middle): macro-tile + one highlighted output sub-tile ----
    grid(ax, 2.6, 3.0, 3.0, 3.0, cols=6, rows=6, face=C_OUT, highlight=[(2, 3)])
    label(ax, 4.1, 5.45, "C [M, N]", fontsize=10, color="#222")
    label(ax, 4.1, 2.8, "block owns one 128×128 macro-tile", fontsize=8.5, color="#222")
    label(ax, 4.1, 2.55, "blockIdx = (bx, by)", fontsize=8.5, color="#333")

    # ---- Right column: smem -> regs -> FFMA -> sync ----
    rx = 7.5
    rw = 5.2
    box(
        ax,
        rx,
        5.7,
        rw,
        1.0,
        "smem  As[2][BK][BM]   Bs[2][BK][BN]\n"
        "double-buffered: compute on buf, LDG next tile into buf^1",
        C_SMEM,
        fontsize=9.5,
    )
    box(
        ax,
        rx,
        4.25,
        rw,
        1.0,
        "registers  Creg[8][8]\n"
        "each thread owns an 8×8 output sub-tile (16×16 grid → 128×128)",
        C_REG,
        fontsize=9.5,
    )
    box(
        ax,
        rx,
        2.6,
        rw,
        1.2,
        "inner loop: for kk in BK:\n"
        "  load Areg[8], Breg[8] from smem\n"
        "  Creg[i][j] += Areg[i] * Breg[j]      (FFMA × 64)",
        C_OP,
        fontsize=9.5,
    )
    box(
        ax,
        rx + 0.8,
        1.6,
        rw - 1.6,
        0.5,
        "__syncthreads() between buffers",
        C_RED,
        fontsize=9.5,
    )

    # Arrows in the right column
    arrow(ax, (rx + rw / 2, 5.7), (rx + rw / 2, 5.25))
    arrow(ax, (rx + rw / 2, 4.25), (rx + rw / 2, 3.8))
    arrow(ax, (rx + rw / 2, 2.6), (rx + rw / 2, 2.1))

    # A and B into smem
    arrow(
        ax,
        (2.1, 4.5),
        (rx, 6.0),
        color="#666",
        linewidth=1.0,
        connectionstyle="arc3,rad=-0.15",
    )
    arrow(
        ax,
        (5.6, 6.35),
        (rx, 6.4),
        color="#666",
        linewidth=1.0,
        connectionstyle="arc3,rad=-0.1",
    )

    # Epilogue back to C
    arrow(
        ax,
        (rx, 4.8),
        (5.6, 4.5),
        color="#888",
        linewidth=1.0,
        connectionstyle="arc3,rad=0.2",
    )
    label(ax, 6.55, 4.95, "epilogue:\nCreg → C tile", fontsize=8, color="#444")

    footer(
        ax,
        0.3,
        1.05,
        [
            "BM=BN=128, BK=8, TM=TN=8. 256 threads/block, 16×16 grid of micro-tiles.",
            "All global loads are float4 (LDG.E.128). A transposed into smem so the kk-loop reads M-major contiguous.",
            "Double-buffer mirrors a cp.async pipeline without sm_80+ instructions — runs on Turing.",
        ],
    )

    save(fig, "gemm_tiled")


if __name__ == "__main__":
    main()
