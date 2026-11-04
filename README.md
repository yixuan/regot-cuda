

# RegOT-CUDA

**RegOT-CUDA (cuRegOT)** is a CUDA-accelerated library for optimal transport computation, providing high-performance implementations of regularized optimal transport algorithms.

As a complement to this library, the [RegOT-Python](https://github.com/yixuan/regot-python) repository provides efficient CPU-based solvers for regularized optimal transport.

## Problem Formulation

Currently RegOT-CUDA solves the entropic-regularized optimal transport problem:

```math
\begin{align*}
\min_{T\in\mathbb{R}^{n\times m}}\quad & \langle T,M\rangle-\eta\cdot h(T),\\
\text{subject to}\quad & T\mathbf{1}_{m}=a,T^{T}\mathbf{1}_{n}=b,T\ge0,
\end{align*}
```

where $a\in\mathbb{R}^n$ and $b\in\mathbb{R}^m$ are two given
probability vectors with $a_i>0$, $b_j>0$, $\sum_{i=1}^n a_i=\sum_{j=1}^m b_j=1$,
and $M\in\mathbb{R}^{n\times m}$ is a given cost matrix.
The function $h(T)=\sum_{i=1}^{n}\sum_{j=1}^{m}T_{ij}(1-\log T_{ij})$ is the entropy term,
and $\eta>0$ is a regularization parameter.

## Work in Progress

RegOT-CUDA is a work in progress. Currently we have implemented the block coordinate descent algorithm (BCD, equivalent to the well-known Sinkhorn algorithm) and the sparse-plus-low-rank quasi-Newton method (SPLR) for entropic-regularized optimal transport. More state-of-the-art solvers are under development, and a list of candidate algorithms can be found in the [RegOT-Python](https://github.com/yixuan/regot-python) package.

## Requirements

- Python >= 3.11
- NumPy >= 1.23.0
- PyTorch >= 2.10 (optional, for PyTorch interface)
- CUDA Toolkit >= 13.0
- The [cuDSS](https://developer.nvidia.com/cudss) library
- Compatible NVIDIA GPU
- C++ compiler (C++11 or higher, for building from source)

## Installation

### Using `pip`

You can simply install **cuRegOT** using the `pip` command, and dependent CUDA runtime libraries will be automatically installed:

```bash
pip install curegot
```

### Building from source

#### Environment Setup

The CUDA development environment is required to build this package from source.
You can install the C++ compiler and CUDA libraries using Conda:

```bash
# Create CUDA development environment
conda create -n nvdev
conda activate nvdev
conda install python=3.12 gxx_linux-64
conda install -c nvidia cuda-toolkit=13.2 libcudss-dev cuda-nvtx-dev
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install numpy requests pybind11
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

#### Build and Install

```bash
cd regot-cuda
pip install --no-build-isolation .
```

### Verify Installation

```bash
python -c "import curegot; print('RegOT-CUDA imported successfully')"
```

## Usage

### NumPy interface

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
result1 = curegot.numpy.sinkhorn_bcd(M, a, b, reg, tol=1e-6, max_iter=1000, verbose=0)
plan1 = result1["plan"]
print(plan1[:3, :3])

result2 = curegot.numpy.sinkhorn_splr(M, a, b, reg, tol=1e-6, max_iter=1000, verbose=0)
plan2 = result2["plan"]
print(plan2[:3, :3])
```

### PyTorch interface

```python
import torch
import curegot

# Create data
torch.manual_seed(123)
n, m = 100, 80
device = "cuda" if torch.cuda.is_available() else "cpu"
M = torch.rand(n, m, device=device)  # Cost matrix
a = torch.rand(n, device=device)     # Source distribution
a = a / torch.sum(a)                 # Normalize
b = torch.rand(m, device=device)     # Target distribution
b = b / torch.sum(b)                 # Normalize
reg = 0.1                            # Regularization parameter

# Call algorithm
result1 = curegot.torch.sinkhorn_bcd(M, a, b, reg, tol=1e-6, max_iter=1000, verbose=0)
plan1 = result1["plan"]
print(plan1[:3, :3])

result2 = curegot.torch.sinkhorn_splr(M, a, b, reg, tol=1e-6, max_iter=1000, verbose=0)
plan2 = result2["plan"]
print(plan2[:3, :3])
```

#### Differentiable PyTorch Sinkhorn

Use `curegot.torch.sinkhorn_splr` for forward-only SPLR solves and
`curegot.torch.sinkhorn_autograd` when the cost matrix or marginals need gradients.
`sinkhorn_autograd` runs the SPLR forward pass while retaining the context
required by PyTorch autograd; `loss.backward()` later invokes the implicit
backward solver.

Inputs must be CUDA float64 tensors on the same device. The cost matrix must
have shape `(n, m)` with `n >= 1` and `m >= 2`.

```python
M = M.to(device="cuda", dtype=torch.float64).requires_grad_(True)
a = a.to(device=M.device, dtype=M.dtype)
b = b.to(device=M.device, dtype=M.dtype)
T = curegot.torch.sinkhorn_autograd(
    M, a, b, reg,
    tol=1e-6,
    max_iter=1000,
    verbose=0,
    backward_preconditioner="diag",
    backward_tol=1e-6,
    backward_max_iter=2000,
)
loss = T.square().sum()
loss.backward()  # M.grad
```

`tol`, `max_iter` and `verbose` control the forward solve. `backward_tol` and
`backward_max_iter` separately control convergence of the implicit backward PCG
solve.

`backward_preconditioner` accepts `none` (plain CG), `diag` (default, inverse
diagonal of the context sparse matrix A), or `splr` (reused forward sparse
factorization plus its rank-2 correction when active). The diagonal is not
the exact Q(T) diagonal. Native C++ PCG always applies the matrix-free CUDA Q(T) operator.
Performance depends on workload and requested accuracy; SPLR is not always
faster than diagonal preconditioning.

First derivatives of M, a and b are supported, including when only a or only b
requires gradients. All three use the same native adjoint solve with the true
Q(T) operator. reg remains a constant scalar. Forward or backward
non-convergence raises; AMP and double backward are unsupported.

Marginals must be strictly positive, finite and have equal total mass. The
solver does not normalize them automatically. Marginal derivatives act on
feasible perturbations with `sum(da) = sum(db)`. We return the reduced-gauge
representative `grad_a = lambda_alpha`, `grad_b = [lambda_beta, 0]`; there is
no extra reg factor. The last zero fixes the gauge: it does **not** mean the
last b entry has no effect. Adding `(c, -c)` to the two marginal gradients
leaves every feasible directional derivative unchanged.

For trainable probability marginals, use separate softmax logits:

```python
u = torch.zeros(M.shape[0], device=M.device, dtype=M.dtype, requires_grad=True)
v = torch.zeros(M.shape[1], device=M.device, dtype=M.dtype, requires_grad=True)
a, b = u.softmax(0), v.softmax(0)
T = curegot.torch.sinkhorn_autograd(M, a, b, reg)
T.square().sum().backward()  # M.grad, u.grad and v.grad
```

PyTorch applies the softmax chain rule and removes the arbitrary additive
constants in marginal gradients. A shared positive mass can multiply both
softmax outputs if total mass is also trainable. For finite differences,
perturb marginals along equal-mass directions (or perturb the logits), rather
than changing a single marginal coordinate while holding the other fixed.

See [the example](examples/differentiable_splr_ot.py). After installation, run
`python -B test/test_cost_grad.py` for the dense oracle, all three autograd
preconditioners, context lifetime, fallback and active rank-2 checks. Run
`python -B test/test_marginal_grad.py` for
marginal gradients, feasible marginal/joint-mass finite differences and shared
autograd input-contract checks.

## Tests

```bash
cd regot-cuda/test
pip install regot
python test_sinkhorn_bcd.py
python test_sinkhorn_splr.py
```

## Benchmark

The plot below shows the benchmark result for some popular GPU-based solvers. The horizontal axis represents the elapsed wall time, and the vertical axis is the optimization error on a logarithmic scale. Lower curves indicate better performance, achieving lower errors in less time.

![](figs/benchmark.png)

More details on the benchmark can be found in [the paper](https://arxiv.org/pdf/2605.08793).

## Bibliography

Please consider to cite our work if you find our algorithms or
software useful in your research and applications.

```bibtex
@inproceedings{qiu2026curegot,
  title={{{cuRegOT}}: A {{GPU}}-Accelerated Solver for Entropic-Regularized Optimal Transport},
  author={Qiu, Yixuan},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```
