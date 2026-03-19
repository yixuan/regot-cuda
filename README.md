# RegOT-CUDA

CUDA-Accelerated Regularized Optimal Transport Library.

## Overview

RegOT-CUDA is a CUDA-accelerated library for optimal transport computation, providing high-performance implementations of regularized optimal transport algorithms.

## Work in Progress

The RegOT-CUDA package is a work in progress. Currently we have implemented the block coordinate descent algorithm (BCD) and the sparse-plus-low-rank quasi-Newton method (SPLR) for entropic-regularized optimal transport. More state-of-the-art solvers are under development, and a list of candidate algorithms can be found in the [RegOT-Python](https://github.com/yixuan/regot-python) package.

## Requirements

- Python >= 3.11
- NumPy >= 1.23.0
- PyTorch >= 2.10 (optional, for PyTorch interface)
- CUDA Toolkit >= 13.0
- Compatible NVIDIA GPU
- C++ compiler (C++11 or higher, for building from source)

## Installation

### Environment Setup

The CUDA development environment is required to build this package from source.
You can install the C++ compiler and CUDA libraries using Conda:

```bash
# Create CUDA development environment
conda create -n nvdev
conda activate nvdev
conda install python=3.12 numpy gxx_linux-64
conda install -c nvidia cuda-toolkit=13.1 libcudss-dev cuda-nvtx-dev
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install requests pybind11
```

You also need to set the `CUDA_HOME` environment variable, for example:

```bash
export CUDA_HOME=/usr/local/cuda
```

If you use the Conda installation method above, you can run the following command to set the environment variable for the virtual environment:

```bash
conda activate nvdev
conda env config vars set CUDA_HOME="<path_to_conda>/envs/nvdev/"
```

Some additional packages are needed to run the numerical experiments:

```bash
pip install scipy matplotlib pot cupy-cuda13x "jax[cuda13]" ott-jax
```

### Build and Install

```bash
cd regot-cuda
pip install --no-build-isolation .
```

### Verify Installation

```bash
python -c "import curegot; print('RegOT-CUDA imported successfully')"
```

## Usage

```python
import numpy as np
import curegot

# Create data
np.random.seed(123)
n, m = 100, 80
M = np.random.rand(n, m)  # Cost matrix
a = np.random.rand(n)     # Source distribution
a = a / np.sum(a)         # Normalize
b = np.random.rand(m)     # Target distribution
b = b / np.sum(b)         # Normalize
reg = 0.1                 # Regularization parameter

# Call algorithm
result1 = curegot.numpy.sinkhorn_bcd(M, a, b, reg, tol=1e-6, max_iter=1000, verbose=1)
plan1 = result1["plan"]
print(plan1[:3, :3])

result2 = curegot.numpy.sinkhorn_splr(M, a, b, reg, tol=1e-6, max_iter=1000, verbose=1)
plan2 = result2["plan"]
print(plan2[:3, :3])
```

## Tests

```bash
cd regot-cuda/test
pip install regot
python test_sinkhorn_bcd.py
python test_sinkhorn_splr.py
```
