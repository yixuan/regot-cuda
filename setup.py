import os
import sys
from pathlib import Path
from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import subprocess

__version__ = "0.1.0"

# The directory that contains setup.py
SETUP_DIRECTORY = Path(__file__).resolve().parent

# Try to import PyTorch's CUDA extension utilities
TORCH_BUILD = False
try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    TORCH_BUILD = True
    print("PyTorch detected, will use PyTorch's BuildExtension and CUDAExtension")
except ImportError:
    print("PyTorch not found, will use custom build extension")
    TORCH_BUILD = False

if TORCH_BUILD:
    # Use PyTorch's CUDAExtension (simpler configuration)
    ext_modules = [
        CUDAExtension(
            name="curegot._internal",
            sources=sorted(glob("src/*.cpp") + glob("src/*.cu")),
            include_dirs=[
                "include/cccl",
                os.path.join(sys.exec_prefix, "include")
            ],
            library_dirs=[
                os.path.join(sys.exec_prefix, "lib64"),
                os.path.join(sys.exec_prefix, "lib")
            ],
            libraries=["cudart", "cuda", "cudss"],
            define_macros=[("VERSION_INFO", __version__), ("TORCH_BUILD", None)],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"]
            }
        )
    ]

    setup(
        name="curegot",
        version=__version__,
        author="Yixuan Qiu",
        author_email="yixuanq@gmail.com",
        url="https://github.com/yixuan/regot-cuda",
        description="CUDA-Accelerated Regularized Optimal Transport",
        long_description="A CUDA-accelerated library for regularized optimal transport computation.",
        packages=["curegot"],
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
        python_requires=">=3.10"
    )

    # Exit with success
    sys.exit(0)



# Use Pybind11Extension with custom BuildExt
def find_cuda():
    """Find CUDA installation"""
    # Check common CUDA installation paths
    cuda_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/cuda"
    ]

    # Also check CUDA_HOME environment variable
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        cuda_paths.insert(0, cuda_home)

    for path in cuda_paths:
        if os.path.exists(path):
            print(f"CUDA path found: {path}")
            return path

    return None

def check_cuda_compiler():
    """Check if nvcc is available"""
    try:
        result = subprocess.run(["nvcc", "--version"],
                                capture_output=True, text=True, check=True)
        print(f"CUDA compiler found:\n{result.stdout}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# Check for CUDA
cuda_path = find_cuda()
if not cuda_path:
    print("Warning: CUDA installation not found. Please install CUDA toolkit and set CUDA_HOME.")
    sys.exit(1)

if not check_cuda_compiler():
    print("Warning: nvcc compiler not found. Please ensure CUDA toolkit is installed and in PATH.")
    sys.exit(1)

# Custom build_ext class that supports CUDA (used when PyTorch is not available)
class BuildExt(build_ext):
    def build_extension(self, ext):
        # Check if this extension has CUDA files
        cuda_sources = [src for src in ext.sources if src.endswith(".cu")]
        cpp_sources = [src for src in ext.sources if not src.endswith(".cu")]

        if cuda_sources:
            print(f"Building extension {ext.name} with CUDA support")

            # Store the original objects and add to extra_objects
            if not hasattr(ext, "extra_objects"):
                ext.extra_objects = []

            # Compile CUDA files first
            for cuda_file in cuda_sources:
                obj_file = self.compile_cuda_file(cuda_file, ext)
                ext.extra_objects.append(obj_file)

            # Remove CUDA files from sources, keep only C++ files
            ext.sources = cpp_sources

            # Ensure CUDA libraries are linked
            if not hasattr(ext, "libraries"):
                ext.libraries = []
            for lib in ["cudart", "cuda"]:
                if lib not in ext.libraries:
                    ext.libraries.append(lib)

        # Let the parent class handle the rest
        super().build_extension(ext)

    def compile_cuda_file(self, cuda_file, ext):
        """Compile a single .cu file to .o object file and return the object file path"""
        output_file = os.path.splitext(cuda_file)[0] + ".o"

        # Build nvcc command
        nvcc_cmd = ["nvcc", "-c", "-O3", "--use_fast_math", "-Xcompiler", "-fPIC"]

        # Add virtual environment include directory
        venv_inc_dir = os.path.join(sys.exec_prefix, "include")
        if venv_inc_dir not in self.compiler.include_dirs:
            nvcc_cmd.append("-I" + venv_inc_dir)

        # Add include directories specified in ext_modules
        if hasattr(ext, "include_dirs"):
            for inc_dir in ext.include_dirs:
                if inc_dir not in self.compiler.include_dirs:
                    nvcc_cmd.append("-I" + inc_dir)

        # Add Python include directories
        for inc_dir in self.compiler.include_dirs:
            nvcc_cmd.append("-I" + inc_dir)

        # Add source and output
        nvcc_cmd.extend([cuda_file, "-o", output_file])

        print(f"Compiling CUDA file: {cuda_file}")
        print(" ".join(nvcc_cmd))
        print()
        subprocess.check_call(nvcc_cmd)

        return output_file

ext_modules = [
    Pybind11Extension(
        name="curegot._internal",
        sources=sorted(glob("src/*.cpp") + glob("src/*.cu")),
        include_dirs=[
            "include/cccl",
            os.path.join(cuda_path, "include")
        ],
        library_dirs=[
            os.path.join(cuda_path, "lib64"),
            os.path.join(cuda_path, "lib")
        ],
        libraries=["cudart", "cuda", "cudss"],
        define_macros=[("VERSION_INFO", __version__)],
        extra_compile_args=["-O3"],
        language="c++"
    )
]

setup(
    name="curegot",
    version=__version__,
    author="Yixuan Qiu",
    author_email="yixuanq@gmail.com",
    url="https://github.com/yixuan/regot-cuda",
    description="CUDA-Accelerated Regularized Optimal Transport",
    long_description="A CUDA-accelerated library for regularized optimal transport computation.",
    packages=["curegot"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    python_requires=">=3.10"
)
