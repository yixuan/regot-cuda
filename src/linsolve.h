#pragma once

#include <iostream>
#include <cmath>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// cuDSS header
#include <cudss.h>

// Sparse Cholesky solver using cuDSS
// Solves the linear system H * x = rhs where H is a symmetric positive definite matrix
// H is given in CSR format: values, colind, rowptr
// Only the lower triangular part is used
//
// In: d_values  [nnz]  -- non-zero values of H in CSR format
// In: d_colind  [nnz]  -- column indices
// In: d_rowptr  [n+1]  -- row pointers
// In: d_rhs     [n]    -- right-hand side vector
// Out: d_x      [n]    -- solution vector
// In: n         [int]  -- matrix dimension
// In: nnz       [int]  -- number of non-zero elements
void sparse_cholesky_solve(
    double* d_values,
    int* d_colind,
    int* d_rowptr,
    double* d_rhs,
    double* d_x,
    int n,
    int nnz
);

// Host function, mainly to test sparse Cholesky solver
void sparse_cholesky_solve_host(
    const double* values,
    const int* colind,
    const int* rowptr,
    const double* rhs,
    double* x,
    int n,
    int nnz
);
