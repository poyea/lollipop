"""Tier-1 diagram for `flash_attention` (FA-2 forward, online softmax)."""

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
        "flash_attention (FA-2) — outer Q-tile × inner KV-tile, online softmax in registers",
    )

    # ---- Q/K/V tensor cartoons ----
    grid(
        ax,
        0.3,
        4.2,
        1.6,
        2.6,
        cols=2,
        rows=8,
        face=C_IN,
        highlight=[(2, 0), (2, 1), (3, 0), (3, 1)],
    )
    label(ax, 1.1, 6.95, "Q [N, D]", fontsize=10)
    label(ax, 1.1, 4.05, "outer: Q-tile (BR rows)", fontsize=8, color="#222")

    grid(
        ax,
        2.3,
        4.2,
        1.6,
        2.6,
        cols=2,
        rows=8,
        face=C_IN,
        highlight=[(c, c2) for c in range(2) for c2 in range(2)],
    )
    label(ax, 3.1, 6.95, "K [N, D]", fontsize=10)
    label(ax, 3.1, 4.05, "inner: KV-tile (BC rows)", fontsize=8, color="#222")

    grid(
        ax,
        4.3,
        4.2,
        1.6,
        2.6,
        cols=2,
        rows=8,
        face=C_IN,
        highlight=[(c, c2) for c in range(2) for c2 in range(2)],
    )
    label(ax, 5.1, 6.95, "V [N, D]", fontsize=10)

    # ---- KV tile loaded into smem (transposed) ----
    box(
        ax,
        6.5,
        5.8,
        4.5,
        1.0,
        "smem  Ks[D][BC+1]  Vs[D][BC+1]\n+1 pad: cooperative store hits 32 distinct banks",
        C_SMEM,
        fontsize=9,
    )

    # ---- Per-thread register state (one Q row per thread) ----
    box(
        ax,
        6.5,
        3.9,
        4.5,
        1.6,
        "registers (per thread = per Q row)\n"
        "q_reg[D]   o_reg[D]   m_i   l_i\n"
        "all of P = softmax(QKᵀ/√D) stays implicit — never materialised",
        C_REG,
        fontsize=9,
    )

    # ---- Online softmax recurrence box ----
    box(
        ax,
        6.5,
        1.4,
        6.2,
        2.0,
        "for each score s = q · K[c] · 1/√D :\n"
        "    m_new = max(m_i, s)\n"
        "    α = exp(m_i - m_new)        # rescale old state\n"
        "    p = exp(s   - m_new)        # this column's weight\n"
        "    l_i ← l_i · α + p\n"
        "    o_i ← o_i · α + p · V[c]\n"
        "    m_i ← m_new\n"
        "final: O = o_i / l_i   (FA-2: normalise ONCE at the end)",
        C_OP,
        fontsize=9,
    )

    # ---- Output ----
    box(ax, 11.4, 5.8, 1.4, 1.0, "O [N, D]\nwrite once", C_OUT, fontsize=9)

    # Arrows
    arrow(ax, (3.9, 5.5), (6.5, 6.2), color="#666", connectionstyle="arc3,rad=-0.15")
    arrow(ax, (5.9, 5.5), (6.5, 6.0), color="#666", connectionstyle="arc3,rad=-0.1")
    arrow(ax, (1.9, 5.5), (6.5, 4.9), color="#666", connectionstyle="arc3,rad=-0.2")
    arrow(ax, (8.75, 5.8), (8.75, 5.5))
    arrow(ax, (8.75, 3.9), (8.75, 3.4))
    arrow(ax, (11.0, 4.0), (11.5, 5.8), color="#666", connectionstyle="arc3,rad=0.2")
    label(ax, 11.5, 4.4, "after\nall KV tiles", fontsize=8, color="#444")

    # ---- Causal whole-tile skip annotation ----
    box(
        ax,
        0.3,
        1.4,
        5.8,
        2.0,
        "causal: outer-loop tile cap\n"
        "n_tiles_causal = ((q_tile+1)·BR-1)/BC + 1\n"
        "→ KV tiles whose first column > q_tile's last row\n"
        "    are skipped entirely (no load, no dot).\n"
        "On N=4096 this ~halves wall-clock for the causal mask.",
        C_RED,
        fontsize=9,
    )

    footer(
        ax,
        0.3,
        0.8,
        [
            "BR=BC=D=64. Memory O(N·D) instead of O(N²) — never materialise P.",
            "Grid (ceil(N/BR), BH); block (BR,) = one thread per Q row.",
        ],
    )

    save(fig, "flash_attention")


if __name__ == "__main__":
    main()
