import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import time

from lollipop import mandelbrot


def main():
    print("=== Mandelbrot Fractal (Custom CUDA Kernel) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    width, height, max_iter = 2048, 2048, 500

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    result = mandelbrot(width, height, max_iter)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    print(f"  {width}x{height} @ {max_iter} iterations")
    print(f"  GPU: {gpu_time:.4f}s")
    print(f"  Pixels computed: {width * height:,}")

    try:
        from PIL import Image

        img = Image.fromarray(result.get(), mode="L")
        img.save("mandelbrot.png")
        print("  Saved: mandelbrot.png")
    except ImportError:
        cp.save("mandelbrot.npy", result)
        print("  Saved: mandelbrot.npy (install Pillow for PNG output)")


if __name__ == "__main__":
    main()
