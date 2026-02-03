import lollipop._cuda_setup  # noqa: F401
import cupy as cp
import time

from lollipop import lorenz


def main():
    print("=== Lorenz Attractor (GPU) ===\n")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB\n")

    cp.zeros(1)
    cp.cuda.Stream.null.synchronize()

    num_traj, num_steps = 1024, 50000

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    x, y, z = lorenz(num_traj, num_steps, dt=0.01)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    x_h = x.get()
    z_h = z.get()

    print(f"  {num_traj:,} trajectories, {num_steps:,} RK4 steps each")
    print(f"  GPU: {gpu_time:.4f}s")
    print(f"  Total integration steps: {num_traj * num_steps:,}\n")
    print(f"  Terminal x spread: [{x_h[:, -1].min():.2f}, {x_h[:, -1].max():.2f}]")
    print(f"  Terminal z range:  [{z_h[:, -1].min():.2f}, {z_h[:, -1].max():.2f}]")
    print(f"  (demonstrates sensitive dependence on initial conditions)")


if __name__ == "__main__":
    main()
