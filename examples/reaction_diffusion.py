import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import time

from lollipop import reaction_diffusion


def main():
    print("=== Gray-Scott Reaction Diffusion ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    width, height, steps = 512, 512, 10000

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    u, v = reaction_diffusion(width, height, steps)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    print(f"  {width}x{height} grid, {steps:,} steps")
    print(f"  GPU: {gpu_time:.4f}s")

    try:
        from PIL import Image
        import numpy as np

        img_data = (v.get() * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_data, mode="L")
        img.save("reaction_diffusion.png")
        print("  Saved: reaction_diffusion.png")
    except ImportError:
        cp.save("reaction_diffusion.npy", v)
        print("  Saved: reaction_diffusion.npy (install Pillow for PNG output)")


if __name__ == "__main__":
    main()
