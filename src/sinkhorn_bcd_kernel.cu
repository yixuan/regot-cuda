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

// CUDA kernel for computing optimal beta given alpha
__global__ void optimal_beta_kernel(
    const double* __restrict__ M,
    const double* __restrict__ alpha,
    const double* __restrict__ logb,
    double* __restrict__ beta,
    double reg,
    int n, int m
)
{
    extern __shared__ double shared_data[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    // Grid stride (number of blocks)
    int stride = gridDim.x;

    // Grid-stride loop instead of `if (j < m)`
    // Each block processes multiple columns
    for (int j = blockIdx.x; j < m; j += stride)
    {
        // First pass: find maximum in column j
        double local_max = -INFINITY;

        // Each thread processes multiple elements of the column
        for (int i = tid; i < n; i += block_size)
        {
            double D_ij = (alpha[i] - M[i * m + j]) / reg;
            local_max = max(local_max, D_ij);
        }

        // Find global maximum across all threads in the block
        shared_data[tid] = local_max;
        __syncthreads();

        for (int s = block_size / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                shared_data[tid] = max(shared_data[tid], shared_data[tid + s]);
            }
            __syncthreads();
        }

        double global_max = shared_data[0];
        __syncthreads();

        // Second pass: compute sum of exp(D_ij - global_max)
        double local_sum = 0.0;

        // Each thread processes multiple elements again
        for (int i = tid; i < n; i += block_size)
        {
            double D_ij = (alpha[i] - M[i * m + j]) / reg;
            local_sum += exp(D_ij - global_max);
        }

        // Reduce sum across all threads in the block
        shared_data[tid] = local_sum;
        __syncthreads();

        for (int s = block_size / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                shared_data[tid] += shared_data[tid + s];
            }
            __syncthreads();
        }

        double global_sum = shared_data[0];

        // First thread writes the result
        if (tid == 0)
        {
            double log_sum = global_max + log(global_sum);
            beta[j] = reg * (logb[j] - log_sum);
        }
    }
}

// CUDA kernel for computing optimal alpha given beta
__global__ void optimal_alpha_kernel(
    const double* __restrict__ M,
    const double* __restrict__ beta,
    const double* __restrict__ loga,
    double* __restrict__ alpha,
    double reg,
    int n, int m
)
{
    extern __shared__ double shared_data[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    // Grid stride (number of blocks)
    int stride = gridDim.x;

    // Grid-stride loop instead of `if (i < n)`
    // Each block processes multiple rows
    for (int i = blockIdx.x; i < n; i += stride)
    {
        // First pass: find maximum in row i
        double local_max = -INFINITY;

        // Each thread processes multiple elements of the row
        for (int j = tid; j < m; j += block_size)
        {
            double D_ij = (beta[j] - M[i * m + j]) / reg;
            local_max = max(local_max, D_ij);
        }

        // Find global maximum across all threads in the block
        shared_data[tid] = local_max;
        __syncthreads();

        for (int s = block_size / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                shared_data[tid] = max(shared_data[tid], shared_data[tid + s]);
            }
            __syncthreads();
        }

        double global_max = shared_data[0];
        __syncthreads();

        // Second pass: compute sum of exp(D_ij - global_max)
        double local_sum = 0.0;

        // Each thread processes multiple elements again
        for (int j = tid; j < m; j += block_size)
        {
            double D_ij = (beta[j] - M[i * m + j]) / reg;
            local_sum += exp(D_ij - global_max);
        }

        // Reduce sum across all threads in the block
        shared_data[tid] = local_sum;
        __syncthreads();

        for (int s = block_size / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                shared_data[tid] += shared_data[tid + s];
            }
            __syncthreads();
        }

        double global_sum = shared_data[0];

        // First thread writes the result
        if (tid == 0)
        {
            double log_sum = global_max + log(global_sum);
            alpha[i] = reg * (loga[i] - log_sum);
        }
    }
}

// CUDA kernel for computing marginal a (row sums of transport plan)
__global__ void compute_marginal_a_kernel(
    const double* __restrict__ M,
    const double* __restrict__ alpha,
    const double* __restrict__ beta,
    double* __restrict__ marginal_a,
    double reg,
    int n, int m
)
{
    // Grid-stride loop instead of `if (i < n)`
    // Each thread processes multiple rows
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
        double sum = 0.0;
        const double alpha_i = alpha[i];
        const double* M_i = M + i * m;
        for (int j = 0; j < m; j++)
        {
            // exp((alpha_i + beta_j - M_ij) / reg)
            sum += exp((alpha_i + beta[j] - M_i[j]) / reg);
        }
        marginal_a[i] = sum;
    }
}

// CUDA kernel for computing marginal b (column sums of transport plan)
__global__ void compute_marginal_b_kernel(
    const double* __restrict__ M,
    const double* __restrict__ alpha,
    const double* __restrict__ beta,
    double* __restrict__ marginal_b,
    double reg,
    int n, int m
)
{
    // Grid-stride loop instead of `if (j < m)`
    // Each thread processes multiple columns
    int stride = gridDim.x * blockDim.x;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < m; j += stride)
    {
        double sum = 0.0;
        const double beta_j = beta[j];
        const double* Mj = M + j;
        for (int i = 0; i < n; i++, Mj += m)
        {
            // exp((alpha_i + beta_j - M_ij) / reg)
            sum += exp((alpha[i] + beta_j - *Mj) / reg);
        }
        marginal_b[j] = sum;
    }
}

// CUDA kernel for computing final transport plan P
__global__ void compute_transport_plan_kernel(
    const double* __restrict__ M,
    const double* __restrict__ alpha,
    const double* __restrict__ beta,
    double* __restrict__ P,
    double reg,
    int n, int m
)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < m)
    {
        P[i * m + j] = exp((alpha[i] + beta[j] - M[i * m + j]) / reg);
    }
}

// CUDA implementation of BCD algorithm for entropic-regularized OT
void cuda_sinkhorn_bcd(
    const double* M, const double* a, const double* b, double* P,
    double reg, int max_iter, double tol, int n, int m, int* niter,
    const double* x0 = nullptr, double* dual = nullptr
)
{
    // Allocate device memory
    double *d_M, *d_a, *d_b, *d_alpha, *d_beta, *d_loga, *d_logb, *d_P;
    double *d_alpha_prev, *d_beta_prev, *d_marginal;
    CUDA_CHECK(cudaMalloc(&d_M, n * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_alpha, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beta, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_loga, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_logb, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_P, n * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_alpha_prev, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beta_prev, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_marginal, max(n, m) * sizeof(double)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_M, M, n * m * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, m * sizeof(double), cudaMemcpyHostToDevice));

    // Compute log vectors on device
    compute_log_vector_cuda(d_a, d_loga, n);
    compute_log_vector_cuda(d_b, d_logb, m);

    // Initialize alpha and beta
    if (x0 != nullptr)
    {
        // Use provided initial values: x0 contains [alpha (n elements), beta (m elements)]
        CUDA_CHECK(cudaMemcpy(d_alpha, x0, n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_beta, x0 + n, m * sizeof(double), cudaMemcpyHostToDevice));
    }
    else
    {
        // Initialize to zero if no initial values provided
        CUDA_CHECK(cudaMemset(d_alpha, 0, n * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_beta, 0, m * sizeof(double)));
    }
    CUDA_CHECK(cudaMemset(d_alpha_prev, 0, n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_beta_prev, 0, m * sizeof(double)));

    // Configure kernel launch parameters
    dim3 threadsPerBlock(BLOCK_DIM);
    // Limit number of blocks to 1024 to avoid excessive kernel launch overhead
    // Each block will process multiple rows/columns via the
    // grid-stride loop in optimal_alpha_kernel()/optimal_beta_kernel()
    dim3 numBlocks_alpha(std::min(n, 1024));
    dim3 numBlocks_beta(std::min(m, 1024));

    // Calculate shared memory size for each kernel (for reduction operations)
    size_t sharedMemory_alpha = threadsPerBlock.x * sizeof(double);  // For reduction in optimal_alpha_kernel
    size_t sharedMemory_beta = threadsPerBlock.x * sizeof(double);   // For reduction in optimal_beta_kernel

    // BCD iterations with convergence checking
    // First use iterates difference error, and then switch to marginal error
    bool use_marginal_error = false;
    for (int iter = 0; iter < max_iter; iter++)
    {
        // Store previous values for error checking
        CUDA_CHECK(cudaMemcpy(d_alpha_prev, d_alpha, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_beta_prev, d_beta, m * sizeof(double), cudaMemcpyDeviceToDevice));

        // Optimal alpha given beta
        optimal_alpha_kernel<<<numBlocks_alpha, threadsPerBlock, sharedMemory_alpha>>>(
            d_M, d_beta, d_loga, d_alpha, reg, n, m
        );

        // Optimal beta given alpha
        optimal_beta_kernel<<<numBlocks_beta, threadsPerBlock, sharedMemory_beta>>>(
            d_M, d_alpha, d_logb, d_beta, reg, n, m
        );

        // Check convergence using dual variable differences -- easier to compute
        if (!use_marginal_error)
        {
            double alpha_diff = compute_l2_distance_cuda(d_alpha, d_alpha_prev, n);
            double beta_diff = compute_l2_distance_cuda(d_beta, d_beta_prev, m);
            double diff_error = std::hypot(alpha_diff, beta_diff);

            if (diff_error < tol)
            {
                use_marginal_error = true;
            }
        }
        else
        {
            // Use marginal error
            // After computing optimal beta, b_tilde=colsum(P) is always equal to b
            // So we use ||a_tilde - a|| as the convergence criterion,
            // where a_tilde=rowsum(P)
            // Compute a_tilde based on current alpha and beta
            int numBlocks_marginal_a = (n + threadsPerBlock.x - 1) / threadsPerBlock.x;
            // Limit number of blocks to 256
            // The grid-stride loop in compute_marginal_a_kernel() will handle larger sizes
            numBlocks_marginal_a = std::min(numBlocks_marginal_a, 256);
            compute_marginal_a_kernel<<<numBlocks_marginal_a, threadsPerBlock>>>(
                d_M, d_alpha, d_beta, d_marginal, reg, n, m
            );

            double marginal_error = compute_l2_distance_cuda(d_marginal, d_a, n);

            if (marginal_error < tol)
            {
                *niter = iter + 1;
                break;
            }
        }

        *niter = iter + 1;
    }

    // Compute final transport plan
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim;
    gridDim.x = (m + blockDim.x - 1) / blockDim.x;
    gridDim.y = (n + blockDim.y - 1) / blockDim.y;
    compute_transport_plan_kernel<<<gridDim, blockDim>>>(
        d_M, d_alpha, d_beta, d_P, reg, n, m
    );

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(P, d_P, n * m * sizeof(double), cudaMemcpyDeviceToHost));

    // Copy dual variables back to host if requested
    if (dual != nullptr)
    {
        CUDA_CHECK(cudaMemcpy(dual, d_alpha, n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dual + n, d_beta, m * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_alpha));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_loga));
    CUDA_CHECK(cudaFree(d_logb));
    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_alpha_prev));
    CUDA_CHECK(cudaFree(d_beta_prev));
    CUDA_CHECK(cudaFree(d_marginal));
}
