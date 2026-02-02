import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import time

from lollipop import voronoi


def main():
    print("=== Voronoi Diagram (Jump Flood Algorithm) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    width, height, num_seeds = 1024, 1024, 128

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    result = voronoi(width, height, num_seeds)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    print(f"  {width}x{height}, {num_seeds} seeds")
    print(f"  GPU: {gpu_time:.4f}s")
    print(f"  Output shape: {result.shape}")

    try:
        from PIL import Image

        img = Image.fromarray(result.get(), mode="RGB")
        img.save("voronoi.png")
        print("  Saved: voronoi.png")
    except ImportError:
        cp.save("voronoi.npy", result)
        print("  Saved: voronoi.npy (install Pillow for PNG output)")


if __name__ == "__main__":
    main()
