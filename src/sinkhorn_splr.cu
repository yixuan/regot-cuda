#include <iostream>
#include <cmath>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Utility functions
#include "utils.h"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Define block dimensions (16x16 = 256 threads)
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define BLOCK_DIM 256

// Kernel function to compute optimal alpha given beta
// From sinkhorn_bcd_kernel.cu
__global__ void optimal_alpha_kernel(
    const double* __restrict__ M,
    const double* __restrict__ beta,
    const double* __restrict__ loga,
    double* __restrict__ alpha,
    double reg,
    int n, int m
);

// CUDA kernel for computing final transport plan P
// From sinkhorn_bcd_kernel.cu
__global__ void compute_transport_plan_kernel(
    const double* __restrict__ M,
    const double* __restrict__ alpha,
    const double* __restrict__ beta,
    double* __restrict__ P,
    double reg,
    int n, int m
);

// Helper function to compute objective function value objfn,
// gradient grad, and sparsified Hessian in CSR form
// From sinkhorn_splr_kernel.cu
void launch_objfn_grad_sphess(
    const double* d_gamma,
    const double* d_M,
    const double* d_ab,
    double reg,
    int n, int m, int K,
    double* d_objfn, double* d_grad,
    double* d_Hvalues, int* d_Hcolind, int* d_Hrowptr,
    double* d_work, int* d_iwork
);

// CUDA implementation of SPLR algorithm for entropic-regularized OT
void cuda_sinkhorn_splr(
    const double* M, const double* a, const double* b, double* P,
    double reg, int max_iter, double tol, int n, int m, int* niter,
    double density_max,
    const double* x0 = nullptr, double* dual = nullptr
)
{
    // Dimensions
    size_t Me = n * m;
    size_t Te = n * (m - 1);
    size_t Hsize = n + m - 1;
    density_max = std::min(density_max, 1.0);
    density_max = std::max(density_max, 0.0);
    size_t Kmax = static_cast<size_t>(density_max * Te);
    Kmax = std::max(Kmax, size_t(1));

    // Allocate device memory
    double *d_M, *d_ab, *d_loga;
    double *d_gamma, *d_gamma_prev, *d_direc;
    double *d_objfn, *d_grad;
    double *d_Hvalues, *d_work;
    int *d_Hcolind, *d_Hrowptr, *d_iwork;
    CUDA_CHECK(cudaMalloc(&d_M, Me * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ab, (n + m) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_loga, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gamma, (n + m) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gamma_prev, (n + m) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_direc, Hsize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_objfn, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad, Hsize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Hvalues, (Te + Hsize) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_work, (n + m + 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Hcolind, (Kmax + Hsize) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Hrowptr, (Hsize + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_iwork, (Te + 2 * Hsize) * sizeof(int)));

    // Pointer aliases
    double* d_alpha = d_gamma;
    double* d_beta = d_gamma + n;

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_M, M, n * m * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ab, a, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ab + n, b, m * sizeof(double), cudaMemcpyHostToDevice));

    // Compute log(a) on device
    compute_log_vector_cuda(d_ab, d_loga, n);

    // Initialize alpha and beta
    if (x0 != nullptr)
    {
        // Use provided initial values: x0 contains [alpha (n elements), beta (m elements)]
        // But note that we force beta[m-1]=0, so we do a shifting
        // alpha += beta[m-1], beta -= beta[m-1]
        double* gamma0 = new double[n + m];
        const double bm1 = x0[n + m - 1];
        for (int i = 0; i < n; i++)
        {
            gamma0[i] = x0[i] + bm1;
        }
        for (int i = 0; i < m; i++)
        {
            gamma0[n + i] = x0[n + i] - bm1;
        }
        CUDA_CHECK(cudaMemcpy(d_gamma, gamma0, (n + m) * sizeof(double), cudaMemcpyHostToDevice));
        delete [] gamma0;
    }
    else
    {
        // If no initial values are provided, first set beta to zero,
        // and then compute alpha using BCD iteration
        CUDA_CHECK(cudaMemset(d_gamma + n, 0, m * sizeof(double)));
        // Optimal alpha given beta
        // Configure kernel launch parameters
        dim3 threadsPerBlock(BLOCK_DIM);
        // Now each block handles one row, so we need n blocks for alpha
        dim3 numBlocks_alpha(n);
        // Calculate shared memory size
        size_t sharedMemory_alpha = threadsPerBlock.x * sizeof(double);
        // Optimal alpha given beta
        // d_alpha = d_gamma
        // d_beta = d_gamma + n
        optimal_alpha_kernel<<<numBlocks_alpha, threadsPerBlock, sharedMemory_alpha>>>(
            d_M, d_beta, d_loga, d_alpha, reg, n, m
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaMemset(d_gamma_prev, 0, (n + m) * sizeof(double)));

    // Main iteration
    for (int iter = 0; iter < max_iter; iter++)
    {
        *niter = iter + 1;
    }

    // Compute final transport plan
    // d_Hvalues is no longer used, and it has at least n*m elements
    // So we use d_Hvalues to hold transport plan
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim;
    gridDim.x = (m + blockDim.x - 1) / blockDim.x;
    gridDim.y = (n + blockDim.y - 1) / blockDim.y;
    double* d_P = d_Hvalues;
    compute_transport_plan_kernel<<<gridDim, blockDim>>>(
        d_M, d_alpha, d_beta, d_P, reg, n, m
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(P, d_P, n * m * sizeof(double), cudaMemcpyDeviceToHost));

    // Copy dual variables back to host if requested
    if (dual != nullptr)
    {
        CUDA_CHECK(cudaMemcpy(dual, d_gamma, (n + m) * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_ab));
    CUDA_CHECK(cudaFree(d_loga));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_gamma_prev));
    CUDA_CHECK(cudaFree(d_direc));
    CUDA_CHECK(cudaFree(d_objfn));
    CUDA_CHECK(cudaFree(d_grad));
    CUDA_CHECK(cudaFree(d_Hvalues));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_Hcolind));
    CUDA_CHECK(cudaFree(d_Hrowptr));
    CUDA_CHECK(cudaFree(d_iwork));
}
