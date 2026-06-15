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
| [`reduction_cg`](lollipop/kernels/_sources/reduction_cg.cu) | same sum-reduce via Cooperative Groups `cg::reduce` |
| [`prefix_sum`](lollipop/kernels/_sources/prefix_sum.cu) | device-wide exclusive scan, hierarchical Blelloch |
| [`radix_sort`](lollipop/kernels/_sources/radix_sort.cu) | LSD radix sort of uint32 keys, multi-block |
| [`matrix_transpose`](lollipop/kernels/_sources/matrix_transpose.cu) | 2D fp32 transpose, 32×33 padded smem tile |
| [`softmax_vec4`](lollipop/kernels/_sources/softmax_vec4.cu) | row-wise softmax with `float4` loads |
| [`flash_attention_hmma`](lollipop/kernels/_sources/flash_attention_hmma.cu) | FA-2 forward, fp16 in / fp32 accum, `wmma` 16×16×16 |
| [`gemm_tiled`](lollipop/kernels/_sources/gemm_tiled.cu) | dense fp32 GEMM, 128×128 macro / 8×8 register micro, manual smem double-buffer |
| [`gemm_int8`](lollipop/kernels/_sources/gemm_int8.cu) | W8A8 INT8 GEMM, per-row act scale × per-channel weight scale |
| [`gemm_int4`](lollipop/kernels/_sources/gemm_int4.cu) | W4A16 weight-only (AWQ/GPTQ-shaped), G=64 asymmetric, dequant-fuse-matmul |
| [`fused_ffn_tail`](lollipop/kernels/_sources/fused_ffn_tail.cu) | RMSNorm → ×γ → +bias → GELU/SiLU → +residual, one kernel |
| [`rope`](lollipop/kernels/_sources/rope.cu) | rotary positional embedding, Llama half-rotation (pair separation D/2), in-place safe |
| [`rmsnorm`](lollipop/kernels/_sources/rmsnorm.cu) | RMSNorm forward + backward, per-row fused reductions, dgamma in fp32 accum |

## License
MIT