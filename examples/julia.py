import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import time

from lollipop import julia


def main():
    print("=== Julia Set Fractal ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    width, height, max_iter = 2048, 2048, 500
    c_re, c_im = -0.7, 0.27015

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    result = julia(width, height, max_iter, c_re, c_im)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    print(f"  {width}x{height} @ {max_iter} iterations")
    print(f"  c = {c_re} + {c_im}i")
    print(f"  GPU: {gpu_time:.4f}s")
    print(f"  Pixels computed: {width * height:,}")

    try:
        from PIL import Image

        img = Image.fromarray(result.get(), mode="L")
        img.save("julia.png")
        print("  Saved: julia.png")
    except ImportError:
        cp.save("julia.npy", result)
        print("  Saved: julia.npy (install Pillow for PNG output)")


if __name__ == "__main__":
    main()
