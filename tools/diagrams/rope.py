"""Tier-1 diagram for `rope` (Llama half-rotation RoPE)."""

from _lib import (
    C_HI,
    C_IN,
    C_OP,
    C_OUT,
    C_REG,
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
    H = 7.5
    fig, ax = new_fig(13, H)
    title(
        ax,
        6.5,
        H - 0.4,
        "rope -- Llama half-rotation: pair (i, i+D/2) rotated by per-position (cos, sin)",
    )

    # ---- Input row of length D, split into x_lo / x_hi halves ----
    row_y, row_h = 5.2, 0.55
    half_w = 4.0
    # x_lo (left half)
    grid(ax, 0.5, row_y, half_w, row_h, cols=8, rows=1, face=C_IN, highlight=[(0, 3)])
    label(
        ax, 0.5 + half_w / 2, row_y + row_h + 0.25, "x_lo  =  x[..., :D/2]", fontsize=10
    )
    # x_hi (right half)
    grid(ax, 4.8, row_y, half_w, row_h, cols=8, rows=1, face=C_IN, highlight=[(0, 3)])
    label(
        ax, 4.8 + half_w / 2, row_y + row_h + 0.25, "x_hi  =  x[..., D/2:]", fontsize=10
    )
    # Pair annotation
    cell_w = half_w / 8
    p1x = 0.5 + 3.5 * cell_w
    p2x = 4.8 + 3.5 * cell_w
    arrow(ax, (p1x, row_y), (p1x, row_y - 0.35), color="#444", linewidth=1.0)
    arrow(ax, (p2x, row_y), (p2x, row_y - 0.35), color="#444", linewidth=1.0)
    label(ax, p1x, row_y - 0.55, "x_lo[i]", fontsize=8.5, color="#222")
    label(ax, p2x, row_y - 0.55, "x_hi[i]", fontsize=8.5, color="#222")
    label(
        ax,
        (p1x + p2x) / 2,
        row_y - 0.95,
        "one thread owns this PAIR (D/2 apart in memory)",
        fontsize=9,
        color="#222",
    )

    # ---- cos / sin per-position vector ----
    cy = 3.2
    box(
        ax,
        0.5,
        cy,
        4.0,
        0.6,
        "cos[row, :D/2]      (fp32, pre-gathered)",
        C_IN,
        fontsize=9.5,
    )
    box(
        ax,
        4.8,
        cy,
        4.0,
        0.6,
        "sin[row, :D/2]      (fp32, pre-gathered)",
        C_IN,
        fontsize=9.5,
    )
    label(
        ax,
        4.4,
        cy - 0.4,
        "v1 shortcut: pre-gathered.   v2: position_ids + cos_cache (gather inside the kernel).",
        fontsize=8.5,
        color="#555",
    )

    # ---- Per-thread compute box ----
    box(
        ax,
        9.5,
        4.4,
        3.2,
        1.6,
        "per thread, in registers:\n"
        "  xl = x_lo[i]\n"
        "  xh = x_hi[i]   <-- read BOTH\n"
        "       halves BEFORE either write\n"
        "       (in-place safe)\n"
        "  c  = cos[i],  s = sin[i]",
        C_REG,
        fontsize=9,
    )

    # ---- Output rotation ----
    out_y = 1.55
    box(ax, 0.5, out_y, 4.0, 0.7, "y_lo[i] = xl * c  -  xh * s", C_OP, fontsize=10)
    box(ax, 4.8, out_y, 4.0, 0.7, "y_hi[i] = xl * s  +  xh * c", C_OP, fontsize=10)
    box(ax, 9.5, out_y, 3.2, 0.7, "y written in place", C_OUT, fontsize=10)

    # Arrows: x halves + cos/sin -> compute box
    arrow(
        ax,
        (4.0, row_y + row_h / 2),
        (9.5, 5.3),
        color="#666",
        linewidth=1.0,
        connectionstyle="arc3,rad=-0.15",
    )
    arrow(
        ax,
        (8.4, row_y + row_h / 2),
        (9.5, 5.5),
        color="#666",
        linewidth=1.0,
        connectionstyle="arc3,rad=-0.05",
    )
    arrow(
        ax,
        (4.4, cy + 0.6),
        (9.5, 4.9),
        color="#666",
        linewidth=1.0,
        connectionstyle="arc3,rad=0.15",
    )
    arrow(
        ax,
        (8.7, cy + 0.6),
        (9.5, 4.7),
        color="#666",
        linewidth=1.0,
        connectionstyle="arc3,rad=0.1",
    )

    # Compute box -> output rows
    arrow(
        ax,
        (11.0, 4.4),
        (4.5, out_y + 0.35),
        color="#666",
        linewidth=1.0,
        connectionstyle="arc3,rad=0.2",
    )
    arrow(
        ax,
        (11.0, 4.4),
        (8.8, out_y + 0.35),
        color="#666",
        linewidth=1.0,
        connectionstyle="arc3,rad=0.25",
    )
    arrow(ax, (11.0, 4.4), (11.1, out_y + 0.7))

    footer(
        ax,
        0.3,
        0.85,
        [
            "Block (256,)  ·  Grid (N,) -- one block per row of x[N, D];  N typically packs B*S*H.",
            "D % 8 == 0 (kept open for v1.1 half2 vectorisation of adjacent pairs).",
            "Pair separation D/2 blocks half2-loading a single pair, but two adjacent x_lo can still half2-load.",
        ],
    )

    save(fig, "rope")


if __name__ == "__main__":
    main()
