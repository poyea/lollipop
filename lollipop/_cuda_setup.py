import os
import sys


def _fix_cuda_path():
    """Add nvidia pip package DLL directories to PATH so CuPy can find them."""
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    nvidia_base = os.path.join(site_packages, "nvidia")
    if not os.path.isdir(nvidia_base):
        return
    for pkg in os.listdir(nvidia_base):
        bin_dir = os.path.join(nvidia_base, pkg, "bin")
        lib_dir = os.path.join(nvidia_base, pkg, "lib")
        for d in (bin_dir, lib_dir):
            if os.path.isdir(d):
                os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


_fix_cuda_path()
