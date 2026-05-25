"""Tier-1 diagram for `flash_attention` (FA-2 forward, online softmax)."""

from _lib import (
    C_IN, C_OP, C_OUT, C_RED, C_REG, C_SMEM, arrow, box, footer, grid, label,
    new_fig, save, title,
)


def main():
    H = 8.5
    fig, ax = new_fig(13, H)
    title(ax, 6.5, H - 0.4,
          "flash_attention (FA-2) — outer Q-tile × inner KV-tile, online softmax in registers")

    # ---- Q/K/V tensor cartoons (left column) ----
    top = 6.6
    grid_h = 2.4
    grid(ax, 0.3, top - grid_h, 1.4, grid_h, cols=2, rows=8, face=C_IN,
         highlight=[(2, 0), (2, 1), (3, 0), (3, 1)])
    label(ax, 1.0, top + 0.18, "Q [N, D]", fontsize=10)
    label(ax, 1.0, top - grid_h - 0.22, "outer: Q-tile (BR rows)", fontsize=8, color="#222")

    grid(ax, 2.1, top - grid_h, 1.4, grid_h, cols=2, rows=8, face=C_IN,
         highlight=[(c, c2) for c in range(2) for c2 in range(2)])
    label(ax, 2.8, top + 0.18, "K [N, D]", fontsize=10)
    label(ax, 2.8, top - grid_h - 0.22, "inner: KV-tile (BC rows)", fontsize=8, color="#222")

    grid(ax, 3.9, top - grid_h, 1.4, grid_h, cols=2, rows=8, face=C_IN,
         highlight=[(c, c2) for c in range(2) for c2 in range(2)])
    label(ax, 4.6, top + 0.18, "V [N, D]", fontsize=10)

    # ---- Smem (top right) ----
    box(ax, 6.0, 6.0, 5.0, 1.0,
        "smem  Ks[D][BC+1]  Vs[D][BC+1]\n+1 pad: cooperative store hits 32 distinct banks",
        C_SMEM, fontsize=9.5)

    # ---- Register state (middle right) ----
    box(ax, 6.0, 4.2, 5.0, 1.5,
        "registers (per thread = per Q row)\n"
        "q_reg[D]   o_reg[D]   m_i   l_i\n"
        "P = softmax(QKᵀ/√D) stays implicit — never materialised",
        C_REG, fontsize=9.5)

    # ---- Output (top-far-right) ----
    box(ax, 11.3, 6.0, 1.5, 1.0, "O [N, D]\nwrite once", C_OUT, fontsize=9.5)

    # ---- Online softmax recurrence ----
    box(ax, 5.6, 1.4, 7.2, 2.4,
        "for each score s = q · K[c] · 1/√D :\n"
        "    m_new = max(m_i, s)\n"
        "    α = exp(m_i − m_new)        # rescale old state\n"
        "    p = exp(s   − m_new)        # this column's weight\n"
        "    l_i ← l_i · α + p\n"
        "    o_i ← o_i · α + p · V[c]\n"
        "    m_i ← m_new\n"
        "final: O = o_i / l_i   (FA-2: normalise ONCE at the end)",
        C_OP, fontsize=9.5)

    # ---- Causal whole-tile skip annotation (bottom-left) ----
    box(ax, 0.3, 1.4, 5.0, 2.4,
        "causal: outer-loop tile cap\n\n"
        "n_tiles_causal =\n   ((q_tile+1)·BR − 1) / BC + 1\n\n"
        "→ KV tiles whose first column > q_tile's\n"
        "    last row are skipped entirely (no load,\n"
        "    no dot).\n"
        "On N=4096 this ~halves causal wall-clock.",
        C_RED, fontsize=9.5)

    # Arrows
    arrow(ax, (3.5, 5.5), (6.0, 6.5), color="#666", connectionstyle="arc3,rad=-0.15")
    arrow(ax, (5.3, 5.5), (6.0, 6.3), color="#666", connectionstyle="arc3,rad=-0.1")
    arrow(ax, (1.7, 5.5), (6.0, 4.95), color="#666", connectionstyle="arc3,rad=-0.25")
    arrow(ax, (8.5, 6.0), (8.5, 5.7))
    arrow(ax, (8.5, 4.2), (8.5, 3.8))
    arrow(ax, (11.0, 4.6), (11.5, 6.0), color="#666", connectionstyle="arc3,rad=0.25")
    label(ax, 11.7, 5.1, "after\nall KV tiles", fontsize=8, color="#444")

    footer(ax, 0.3, 0.85, [
        "BR=BC=D=64. Memory O(N·D) instead of O(N²) — never materialise P.",
        "Grid (ceil(N/BR), BH); block (BR,) = one thread per Q row.",
    ])

    save(fig, "flash_attention")


if __name__ == "__main__":
    main()
