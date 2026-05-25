"""Tier-1 diagram for `gemm_int8` (W8A8) and `gemm_int4` (W4A16) — quant GEMM via wmma."""

from _lib import (
    C_IN,
    C_OP,
    C_OUT,
    C_REG,
    C_SMEM,
    arrow,
    box,
    footer,
    label,
    new_fig,
    save,
    title,
)


def main():
    H = 9.0
    fig, ax = new_fig(13, H)
    title(
        ax,
        6.5,
        H - 0.4,
        "gemm_int8 / gemm_int4 — quantised weights × wmma m16n16k16 fragments",
    )

    # ---- INT8 row ----
    label(
        ax,
        6.5,
        7.6,
        "INT8 W8A8  —  A_q, B_q both int8;  C = a_scale · b_scale · (A_q @ B_qᵀ)",
        fontsize=10.5,
        color="#222",
    )
    y8 = 6.2
    box(
        ax,
        0.3,
        y8,
        2.4,
        1.0,
        "A_q [M, K]\nint8 activations\n(per-row scale)",
        C_IN,
        fontsize=9,
    )
    box(
        ax,
        2.9,
        y8,
        2.4,
        1.0,
        "B_q [N, K]\nint8 weights\n(pre-transposed)",
        C_IN,
        fontsize=9,
    )
    box(
        ax,
        5.5,
        y8,
        2.4,
        1.0,
        "smem As[64][32]\n    Bs[64][32]\nint8 tiles, 2KB each",
        C_SMEM,
        fontsize=9,
    )
    box(
        ax,
        8.1,
        y8,
        2.4,
        1.0,
        "wmma m16n16k16 int8\n4 warps × 4 n-tiles\nint32 accumulator",
        C_OP,
        fontsize=9,
    )
    box(
        ax,
        10.7,
        y8,
        2.0,
        1.0,
        "epilogue\n× a_scale[m] × b_scale[n]\n→ fp32 C",
        C_OUT,
        fontsize=9,
    )
    for x in (2.7, 5.3, 7.9, 10.5):
        arrow(ax, (x, y8 + 0.5), (x + 0.2, y8 + 0.5))

    # ---- Divider ----
    ax.plot([0.3, 12.7], [5.55, 5.55], color="#bbb", linewidth=0.8)

    # ---- INT4 row ----
    label(
        ax,
        6.5,
        5.15,
        "INT4 W4A16 (AWQ/GPTQ-shaped)  —  fp16 activations, int4 weights packed 2-per-byte",
        fontsize=10.5,
        color="#222",
    )
    y4 = 3.6
    box(
        ax,
        0.3,
        y4,
        2.4,
        1.0,
        "A [M, K]\nfp16 activations\n(no quant)",
        C_IN,
        fontsize=9,
    )
    box(
        ax,
        2.9,
        y4,
        2.4,
        1.0,
        "Wq [N, K/2] uint8 packed\nscales / zeros [K/G, N] fp16\n(one per group G=64)",
        C_IN,
        fontsize=8.5,
    )
    box(
        ax,
        5.5,
        y4,
        2.4,
        1.0,
        "smem As[64][64] fp16\n    Ws[64][64] fp16\n(dequantised in-register)",
        C_SMEM,
        fontsize=8.5,
    )
    box(
        ax,
        8.1,
        y4,
        2.4,
        1.0,
        "wmma m16n16k16 fp16\n× fp16 → fp32 accum\nBK == G: scales\nconstant per K-tile",
        C_OP,
        fontsize=8.5,
    )
    box(ax, 10.7, y4, 2.0, 1.0, "epilogue\n→ fp16 C", C_OUT, fontsize=9)
    for x in (2.7, 5.3, 7.9, 10.5):
        arrow(ax, (x, y4 + 0.5), (x + 0.2, y4 + 0.5))

    # ---- Dequant detail callout (bottom-left) ----
    box(
        ax,
        0.3,
        1.4,
        6.0,
        1.4,
        "per K-iter, per thread (INT4 path):\n"
        "  load 16 packed bytes  (= 32 int4 weights)\n"
        "  dequant inline:  w = s · (q − z)        (in registers)\n"
        "  store 32 fp16 into Ws[t_row][..]    →  wmma reads it as fp16",
        C_REG,
        fontsize=9,
    )
    arrow(
        ax,
        (6.3, 2.1),
        (8.1, y4),
        color="#666",
        linewidth=1.0,
        connectionstyle="arc3,rad=-0.2",
    )

    footer(
        ax,
        0.3,
        0.9,
        [
            "Both kernels: 64×64 macro, 4 warps, wmma fragments in int32/fp32 accumulator.",
            "INT4 fp16 round-trip through smem is the cost of using wmma at all — sm_75 has no native int4-input fragment.",
            "Tested numerical parity vs fp32 dequant-then-matmul reference; see docs/profiles/gemm_int{4,8}.md.",
        ],
    )

    save(fig, "gemm_quant")


if __name__ == "__main__":
    main()
