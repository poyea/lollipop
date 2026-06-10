import lollipop._cuda_setup  # noqa: F401
import cupy as cp

from _timing import cuda_time
from lollipop import reduction, reduction_v2


def bench(fn, data, iters=50, warmup=5):
    return cuda_time(fn, data, iters=iters, warmup=warmup)  # seconds/call


n = 100_000_000
rng = cp.random.default_rng(0)
data = rng.standard_normal(n, dtype=cp.float32)
bytes_moved = n * 4

t = bench(reduction, data)
print(
    f"reduction (baseline t=256 i=2):  {t*1000:6.2f} ms   {bytes_moved/t/1e9:6.1f} GB/s"
)
for threads in (128, 256, 512, 1024):
    t = bench(lambda d: reduction_v2(d, threads=threads), data)
    print(
        f"reduction_v2  t={threads:4d} i=8:        "
        f"{t*1000:6.2f} ms   {bytes_moved/t/1e9:6.1f} GB/s"
    )
