#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>

// Host function to compute log of vectors
// From sinkhorn_bcd_kernel.cu
void compute_log_vector(const double* x, double* log_x, int size);

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

// CUDA implementation of SPLR algorithm for entropic-regularized OT
void cuda_sinkhorn_splr(
    const double* M, const double* a, const double* b, double* P,
    double reg, int max_iter, double tol, int n, int m, int* niter,
    const double* x0 = nullptr, double* dual = nullptr
)
{
    // Allocate device memory
    double *d_M, *d_ab, *d_loga, *d_P;
    double *d_gamma, *d_gamma_prev, *d_direc;
    cudaMalloc(&d_M, n * m * sizeof(double));
    cudaMalloc(&d_ab, (n + m) * sizeof(double));
    cudaMalloc(&d_loga, n * sizeof(double));
    cudaMalloc(&d_P, n * m * sizeof(double));
    cudaMalloc(&d_gamma, (n + m) * sizeof(double));
    cudaMalloc(&d_gamma_prev, (n + m) * sizeof(double));
    cudaMalloc(&d_direc, (n + m - 1) * sizeof(double));

    // Pointer aliases
    double* d_alpha = d_gamma;
    double* d_beta = d_gamma + n;

    // Copy input to device
    cudaMemcpy(d_M, M, n * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ab, a, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ab + n, b, m * sizeof(double), cudaMemcpyHostToDevice);

    // Compute log vectors on host and copy to device
    double* loga = new double[n];
    compute_log_vector(a, loga, n);
    cudaMemcpy(d_loga, loga, n * sizeof(double), cudaMemcpyHostToDevice);
    delete[] loga;

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
        cudaMemcpy(d_gamma, gamma0, (n + m) * sizeof(double), cudaMemcpyHostToDevice);
        delete [] gamma0;
    }
    else
    {
        // If no initial values are provided, first set beta to zero,
        // and then compute alpha using BCD iteration
        cudaMemset(d_gamma + n, 0, m * sizeof(double));
        // Optimal alpha given beta
        // Configure kernel launch parameters
        dim3 threadsPerBlock(256);
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
        cudaDeviceSynchronize();
    }
    cudaMemset(d_gamma_prev, 0, (n + m) * sizeof(double));

    // Main iteration
    for (int iter = 0; iter < max_iter; iter++)
    {
        *niter = iter + 1;
    }

    // Compute final transport plan
    dim3 threadsPerBlock_2d(16, 16);
    dim3 numBlocks_2d((m + threadsPerBlock_2d.x - 1) / threadsPerBlock_2d.x,
                      (n + threadsPerBlock_2d.y - 1) / threadsPerBlock_2d.y);
    compute_transport_plan_kernel<<<numBlocks_2d, threadsPerBlock_2d>>>(
        d_M, d_alpha, d_beta, d_P, reg, n, m
    );
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(P, d_P, n * m * sizeof(double), cudaMemcpyDeviceToHost);

    // Copy dual variables back to host if requested
    if (dual != nullptr)
    {
        cudaMemcpy(dual, d_gamma, (n + m) * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // Free device memory
    cudaFree(d_M);
    cudaFree(d_ab);
    cudaFree(d_loga);
    cudaFree(d_P);
    cudaFree(d_gamma);
    cudaFree(d_gamma_prev);
    cudaFree(d_direc);
}
