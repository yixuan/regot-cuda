#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>

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
    // Each block handles one column, with fixed block size (e.g., 256)
    int j = blockIdx.x;

    if (j < m)
    {
        extern __shared__ double shared_data[];
        int tid = threadIdx.x;
        int block_size = blockDim.x;

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
    // Each block handles one row, with fixed block size (e.g., 256)
    int i = blockIdx.x;  // Each block handles one row

    if (i < n)
    {
        extern __shared__ double shared_data[];
        int tid = threadIdx.x;
        int block_size = blockDim.x;

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
    // Row index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
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
    // Column index
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < m)
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

// CUDA kernel for computing sum of squares of vector difference
__global__ void compute_squared_l2_kernel(
    const double* __restrict__ vec1,
    const double* __restrict__ vec2,
    double* __restrict__ error,
    int size
)
{
    extern __shared__ double shared_data[];

    // Indices
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load squared difference into shared memory
    double squared_diff = 0.0;
    if (idx < size)
    {
        double diff = vec1[idx] - vec2[idx];
        squared_diff = diff * diff;
    }
    shared_data[tid] = squared_diff;

    __syncthreads();

    // Reduction in shared memory -- Compute partial sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write result to global memory
    if (tid == 0)
    {
        atomicAdd(error, shared_data[0]);
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

// Host function to compute log of vectors
void compute_log_vector(const double* x, double* log_x, int size)
{
    for (int i = 0; i < size; i++)
    {
        log_x[i] = std::log(x[i]);
    }
}

// Helper function to compute L2 norm on device
double compute_l2_norm_cuda(double* d_vec1, double* d_vec2, int size)
{
    double* d_error;
    cudaMalloc(&d_error, sizeof(double));
    cudaMemset(d_error, 0, sizeof(double));

    dim3 threadsPerBlock(BLOCK_DIM);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    size_t sharedMemory = threadsPerBlock.x * sizeof(double);

    compute_squared_l2_kernel<<<numBlocks, threadsPerBlock, sharedMemory>>>(
        d_vec1, d_vec2, d_error, size
    );
    cudaDeviceSynchronize();

    double error;
    cudaMemcpy(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_error);

    return sqrt(error);
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
    cudaMalloc(&d_M, n * m * sizeof(double));
    cudaMalloc(&d_a, n * sizeof(double));
    cudaMalloc(&d_b, m * sizeof(double));
    cudaMalloc(&d_alpha, n * sizeof(double));
    cudaMalloc(&d_beta, m * sizeof(double));
    cudaMalloc(&d_loga, n * sizeof(double));
    cudaMalloc(&d_logb, m * sizeof(double));
    cudaMalloc(&d_P, n * m * sizeof(double));
    cudaMalloc(&d_alpha_prev, n * sizeof(double));
    cudaMalloc(&d_beta_prev, m * sizeof(double));
    cudaMalloc(&d_marginal, max(n, m) * sizeof(double));

    // Copy input to device
    cudaMemcpy(d_M, M, n * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, a, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, m * sizeof(double), cudaMemcpyHostToDevice);

    // Compute log vectors on host and copy to device
    double* loga = new double[n];
    double* logb = new double[m];
    compute_log_vector(a, loga, n);
    compute_log_vector(b, logb, m);
    cudaMemcpy(d_loga, loga, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_logb, logb, m * sizeof(double), cudaMemcpyHostToDevice);

    // Initialize alpha and beta
    if (x0 != nullptr)
    {
        // Use provided initial values: x0 contains [alpha (n elements), beta (m elements)]
        cudaMemcpy(d_alpha, x0, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, x0 + n, m * sizeof(double), cudaMemcpyHostToDevice);
    }
    else
    {
        // Initialize to zero if no initial values provided
        cudaMemset(d_alpha, 0, n * sizeof(double));
        cudaMemset(d_beta, 0, m * sizeof(double));
    }
    cudaMemset(d_alpha_prev, 0, n * sizeof(double));
    cudaMemset(d_beta_prev, 0, m * sizeof(double));

    // Configure kernel launch parameters
    dim3 threadsPerBlock(BLOCK_DIM);
    // Now each block handles one row/column, so we need n blocks for alpha and m blocks for beta
    // One block per row
    dim3 numBlocks_alpha(n);
    // One block per column
    dim3 numBlocks_beta(m);

    // Calculate shared memory size for each kernel (for reduction operations)
    size_t sharedMemory_alpha = threadsPerBlock.x * sizeof(double);  // For reduction in optimal_alpha_kernel
    size_t sharedMemory_beta = threadsPerBlock.x * sizeof(double);   // For reduction in optimal_beta_kernel

    // BCD iterations with convergence checking
    // First use iterates difference error, and then switch to marginal error
    bool use_marginal_error = false;
    for (int iter = 0; iter < max_iter; iter++)
    {
        // Store previous values for error checking
        cudaMemcpy(d_alpha_prev, d_alpha, n * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_beta_prev, d_beta, m * sizeof(double), cudaMemcpyDeviceToDevice);

        // Optimal alpha given beta
        optimal_alpha_kernel<<<numBlocks_alpha, threadsPerBlock, sharedMemory_alpha>>>(
            d_M, d_beta, d_loga, d_alpha, reg, n, m
        );
        cudaDeviceSynchronize();

        // Optimal beta given alpha
        optimal_beta_kernel<<<numBlocks_beta, threadsPerBlock, sharedMemory_beta>>>(
            d_M, d_alpha, d_logb, d_beta, reg, n, m
        );
        cudaDeviceSynchronize();

        // Check convergence using dual variable differences -- easier to compute
        if (!use_marginal_error)
        {
            double alpha_diff = compute_l2_norm_cuda(d_alpha, d_alpha_prev, n);
            double beta_diff = compute_l2_norm_cuda(d_beta, d_beta_prev, m);
            double diff_error = hypot(alpha_diff, beta_diff);

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
            compute_marginal_a_kernel<<<numBlocks_alpha, threadsPerBlock>>>(
                d_M, d_alpha, d_beta, d_marginal, reg, n, m
            );
            cudaDeviceSynchronize();

            double marginal_error = compute_l2_norm_cuda(d_marginal, d_a, n);

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
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(P, d_P, n * m * sizeof(double), cudaMemcpyDeviceToHost);

    // Copy dual variables back to host if requested
    if (dual != nullptr)
    {
        cudaMemcpy(dual, d_alpha, n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(dual + n, d_beta, m * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // Free device memory
    cudaFree(d_M);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_loga);
    cudaFree(d_logb);
    cudaFree(d_P);
    cudaFree(d_alpha_prev);
    cudaFree(d_beta_prev);
    cudaFree(d_marginal);

    // Free host memory
    delete[] loga;
    delete[] logb;
}
