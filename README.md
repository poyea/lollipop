# lollipop

Sweet GPU compute kernels in CUDA, wrapped in Python via CuPy.

```bash
uv sync && uv pip install -e . && python examples/mandelbrot.py
```

You need CUDA Toolkit 11.8 (well, newer version *may not* work) and an NVIDIA GPU (sm_75+ for the HMMA kernels; Turing or anything newer). CuPy's bundled `nvrtc` compiles each kernel at first use, picking up `mma.h` and friends from `CUDA_PATH`.

## Example Kernels

| Kernel | What it does |
|---|---|
| [`reduction_v2`](lollipop/kernels/_sources/reduction_v2.cu) | sum-reduce a 1D float array |
| [`matrix_transpose`](lollipop/kernels/_sources/matrix_transpose.cu) | 2D fp32 transpose |
| [`flash_attention_hmma`](lollipop/kernels/_sources/flash_attention_hmma.cu) | causal self-attention, fp16 in / fp32 accum |
| [`gemm_tiled`](lollipop/kernels/_sources/gemm_tiled.cu) | dense fp32 GEMM |
| [`gemm_int8`](lollipop/kernels/_sources/gemm_int8.cu) | W8A8 INT8 GEMM |
| [`gemm_int4`](lollipop/kernels/_sources/gemm_int4.cu) | W4A16 weight-only GEMM (AWQ/GPTQ-shaped) |
| [`fused_ffn_tail`](lollipop/kernels/_sources/fused_ffn_tail.cu) | RMSNorm + bias + GELU/SiLU + residual, fused |
| [`rope`](lollipop/kernels/_sources/rope.cu) | rotary positional embedding (Llama half-rotation) |

## License
MIT