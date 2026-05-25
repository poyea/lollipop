"""Tier-1 diagram for `gemm_int8` (W8A8) and `gemm_int4` (W4A16) — quant GEMM via wmma."""

from _lib import (
    C_IN,
    C_OP,
    C_OUT,
    C_RED,
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
    fig, ax = new_fig(13, 8)
    title(
        ax,
        6.5,
        7.65,
        "gemm_int8 / gemm_int4 — quantised weights × wmma m16n16k16 fragments",
    )

    # ---- INT8 row (top half) ----
    label(
        ax,
        6.5,
        7.15,
        "INT8 W8A8  —  A_q, B_q both int8;  C = a_scale · b_scale · (A_q @ B_qᵀ)",
        fontsize=10.5,
        color="#222",
    )
    box(
        ax,
        0.4,
        5.4,
        2.4,
        1.0,
        "A_q [M, K]\nint8 activations\n(per-row scale)",
        C_IN,
        fontsize=9,
    )
    box(
        ax,
        3.0,
        5.4,
        2.4,
        1.0,
        "B_q [N, K]\nint8 weights\n(pre-transposed)",
        C_IN,
        fontsize=9,
    )
    box(
        ax,
        5.6,
        5.4,
        2.4,
        1.0,
        "smem As[64][32]\n    Bs[64][32]\nint8 tiles, 2KB each",
        C_SMEM,
        fontsize=9,
    )
    box(
        ax,
        8.2,
        5.4,
        2.4,
        1.0,
        "wmma m16n16k16 int8\n4 warps × 4 n-tiles\nint32 accumulator",
        C_OP,
        fontsize=9,
    )
    box(
        ax,
        10.8,
        5.4,
        1.9,
        1.0,
        "epilogue\n× a_scale[m] × b_scale[n]\n→ fp32 C",
        C_OUT,
        fontsize=9,
    )
    for x in (2.8, 5.4, 8.0, 10.6):
        arrow(ax, (x, 5.9), (x + 0.2, 5.9))

    # ---- INT4 row (bottom half) ----
    label(
        ax,
        6.5,
        4.45,
        "INT4 W4A16 (AWQ/GPTQ-shaped)  —  fp16 activations, int4 weights packed 2-per-byte",
        fontsize=10.5,
        color="#222",
    )

    box(
        ax,
        0.4,
        2.6,
        2.4,
        1.0,
        "A [M, K]\nfp16 activations\n(no quant)",
        C_IN,
        fontsize=9,
    )
    box(
        ax,
        3.0,
        2.6,
        2.4,
        1.0,
        "Wq [N, K/2]\nuint8 packed:\nlow nibble = k-even\nhigh nibble = k-odd",
        C_IN,
        fontsize=8.5,
    )
    box(
        ax,
        5.6,
        3.7,
        2.4,
        0.5,
        "scales / zeros\n[K/G, N]  fp16, one per group of G=64",
        C_IN,
        fontsize=8,
    )
    box(
        ax,
        5.6,
        2.6,
        2.4,
        1.0,
        "smem As[64][64] fp16\n    Ws[64][64] fp16\n  (dequantised in-register)",
        C_SMEM,
        fontsize=9,
    )
    box(
        ax,
        8.2,
        2.6,
        2.4,
        1.0,
        "wmma m16n16k16 fp16\n× fp16 → fp32 accum\n(BK == G: scales\n constant per K-tile)",
        C_OP,
        fontsize=8.5,
    )
    box(ax, 10.8, 2.6, 1.9, 1.0, "epilogue\n→ fp16 C", C_OUT, fontsize=9)
    for x in (2.8, 5.4, 8.0, 10.6):
        arrow(ax, (x, 3.1), (x + 0.2, 3.1))

    # Dequant detail callout
    box(
        ax,
        0.4,
        1.2,
        5.0,
        1.0,
        "per K-iter, per thread:\n"
        "  load 16 packed bytes (= 32 int4 weights)\n"
        "  dequant inline:  w = s · (q − z)        (in registers)\n"
        "  store 32 fp16 into Ws[t_row][..]   →  wmma reads it as fp16",
        C_REG,
        fontsize=9,
    )
    arrow(
        ax,
        (5.5, 1.7),
        (6.5, 2.6),
        color="#666",
        linewidth=1.0,
        connectionstyle="arc3,rad=-0.2",
    )

    footer(
        ax,
        0.3,
        0.5,
        [
            "Both kernels: 64×64 macro tile, 4 warps (128 threads), wmma fragments stored in fp32/int32 accumulator.",
            "INT4 round-trip through smem in fp16 is the cost of using wmma at all — sm_75 has no native int4-input fragment.",
            "Tested numerical parity against an fp32 dequant-then-matmul reference; see docs/profiles/gemm_int{4,8}.md.",
        ],
    )

    save(fig, "gemm_quant")


if __name__ == "__main__":
    main()
