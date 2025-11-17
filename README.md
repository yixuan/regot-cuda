# RegOT-CUDA

CUDA-Accelerated Regularized Optimal Transport Library.

## Overview

RegOT-CUDA is a CUDA-accelerated library for optimal transport computation, providing high-performance implementations of regularized optimal transport algorithms.

## Work in Progress

The RegOT-CUDA package is a work in progress. Currently we have implemented the block coordinate descent (BCD) algorithm for entropic-regularized optimal transport, which is equivalent to the well-known Sinkhorn algorithm in the logarithmic scale. More state-of-the-art solvers are under development, and a list of candidate algorithms can be found in the [RegOT-Python](https://github.com/yixuan/regot-python) package.

## Requirements

- Python >= 3.10
- NumPy >= 1.23.0
- CUDA Toolkit >= 11.0
- Compatible NVIDIA GPU
- C++ compiler (C++11 or higher, for building from source)

## Installation

### Environment Setup

It is recommended to use Conda to create a virtual environment and install necessary packages (using Linux as an example):

```bash
# Create CUDA development environment
conda create -n nvdev
conda activate nvdev
conda install python=3.12 numpy scipy matplotlib notebook ipywidgets gxx_linux-64
conda install -c nvidia cuda-toolkit libcudss-dev
```

To compile RegOT-CUDA, you need to set the `CUDA_HOME` environment variable, for example:

```bash
export CUDA_HOME=/usr/local/cuda
```

If you use the Conda installation method above, you can run the following command to set the environment variable for the virtual environment:

```bash
conda activate nvdev
conda env config vars set CUDA_HOME="<path_to_conda>/envs/<nvdev>/targets/<x86_64-linux>/"
```

Please replace `<path_to_conda>` with your Conda installation directory, `<nvdev>` with your virtual environment name (which is `nvdev` according to the installation commands above), and `<x86_64-linux>` with the corresponding installation directory for your operating system (which is `x86_64-linux` on Linux).

### Build and Install

```bash
cd regot-cuda
pip install pybind11
pip install .
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
result = curegot.sinkhorn_bcd(M, a, b, reg, tol=1e-6, max_iter=1000, verbose=1)
plan = result["plan"]
```

## Tests

```bash
cd regot-cuda/test
pip install regot
python test_sinkhorn_bcd.py
```
