import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import time

from lollipop import wave_equation


def main():
    print("=== 2D Wave Equation (GPU) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    width, height, steps = 512, 512, 2000

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    u = wave_equation(width, height, steps, c=1.0, dt=0.1)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    print(f"  {width}x{height} grid, {steps:,} time steps")
    print(f"  GPU: {gpu_time:.4f}s")

    try:
        from PIL import Image
        import numpy as np

        field = u.get()
        # Normalise to 0-255
        vmin, vmax = field.min(), field.max()
        if vmax - vmin > 1e-8:
            img_data = (
                ((field - vmin) / (vmax - vmin) * 255).clip(0, 255).astype(np.uint8)
            )
        else:
            img_data = np.zeros_like(field, dtype=np.uint8)
        img = Image.fromarray(img_data, mode="L")
        img.save("wave_equation.png")
        print("  Saved: wave_equation.png")
    except ImportError:
        cp.save("wave_equation.npy", u)
        print("  Saved: wave_equation.npy (install Pillow for PNG output)")


if __name__ == "__main__":
    main()
