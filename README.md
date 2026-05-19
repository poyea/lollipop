# lollipop

Sweet GPU compute kernels in CUDA, wrapped in Python via CuPy.

```bash
uv sync && uv pip install -e . && python examples/mandelbrot.py
```

You need CUDA Toolkit 11.8 (well, newer version *may not* work) and an NVIDIA GPU (sm_75+ for the HMMA kernels; Turing or anything newer). CuPy's bundled `nvrtc` compiles each kernel at first use, picking up `mma.h` and friends from `CUDA_PATH`.

## License
MIT