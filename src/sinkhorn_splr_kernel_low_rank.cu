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

// Define block dimension
#define BLOCK_DIM 256
#define MAX_NUM_BLOCK_1D 256

// Helper function: intra-warp reduction
__device__ __forceinline__ double warp_reduce_sum(double val)
{
    const unsigned int mask = 0xffffffff;
    for (int s = warpSize / 2; s > 0; s >>= 1)
    {
        val += __shfl_down_sync(mask, val, s);
    }
    return val;
}

// Helper function: intra-block reduction
// Input: local sum in the current thread
// Output: when threadIdx.x == 0, return the block sum
// Do not call this function for other threads
__device__ __forceinline__ double block_reduce_sum(double val)
{
    // Reduction within the warp
    double sum = warp_reduce_sum(val);

    // Use shared memory for intra-block reduction
    // Assuming max blockDim.x is 1024 and warpSize is 32, max 32 slots are needed
    static __shared__ double shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // These threads (lane == 0) contain the warp sums
    if (lane == 0)
    {
        // shared[0]: warp 0 sum
        // shared[1]: warp 1 sum
        // ...
        shared[warp_id] = sum;
    }
    __syncthreads();

    // Let the first warp aggregate results from all warps
    // Only warp 0 does this
    // This time results are stored in shared memory
    if (warp_id == 0)
    {
        // (blockDim.x + warpSize - 1) / warpSize means how many warps are actually used
        sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0;
        sum = warp_reduce_sum(sum);
    }

    return sum;
}

// CUDA kernel to compute low-rank vectors y and s
// y = grad - grad_prev
// s = gamma - gamma_prev
// ys = y's
// yy = y'y
//
// In: grad        [Hsize=n+m-1]
// In: grad_prev   [Hsize]
// In: gamma       [Hsize]
// In: gamma_prev  [Hsize]
// Out: y          [Hsize]
// Out: s          [Hsize]
// Out: ys         [1]
// Out: yy         [1]
__global__ void low_rank_fused_kernel(
    const double* __restrict__ grad,
    const double* __restrict__ grad_prev,
    const double* __restrict__ gamma,
    const double* __restrict__ gamma_prev,
    double* __restrict__ y,
    double* __restrict__ s,
    double* __restrict__ ys,
    double* __restrict__ yy,
    int size
)
{
    // Indices
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Local accumulators
    double local_ys = 0.0;
    double local_yy = 0.0;

    // Grid-stride loop
    for (int i = idx; i < size; i += stride)
    {
        // Load
        double g = grad[i];
        double gp = grad_prev[i];
        double gam = gamma[i];
        double gamp = gamma_prev[i];

        // Compute
        double yval = g - gp;
        double sval = gam - gamp;

        // Store
        y[i] = yval;
        s[i] = sval;

        // Accumulate
        local_ys += yval * sval;
        local_yy += yval * yval;
    }

    // Parallel reduction
    local_ys = block_reduce_sum(local_ys);
    // block_reduce_sum contains a __syncthreads(),
    // so it is safe to call sequentially
    local_yy = block_reduce_sum(local_yy);

    // Write to global memory
    // Only one thread needs to do this once
    if (tid == 0)
    {
        atomicAdd(ys, local_ys);
        atomicAdd(yy, local_yy);
    } 
}

// Helper function to compute low-rank vectors y and s
//
// In: d_grad        [Hsize=n+m-1]
// In: d_grad_prev   [Hsize]
// In: d_gamma       [Hsize]
// In: d_gamma_prev  [Hsize]
// Out: d_y          [Hsize]
// Out: d_s          [Hsize]
// Out: ys (host)    [1]
// Out: yy (host)    [1]
void launch_low_rank(
    const double* d_grad,
    const double* d_grad_prev,
    const double* d_gamma,
    const double* d_gamma_prev,
    double* d_y,
    double* d_s,
    double& ys,
    double& yy,
    int size,
    cudaStream_t stream = cudaStreamPerThread
)
{
    // Allocate device memory for scalars
    double* d_ys;
    double* d_yy;
    CUDA_CHECK(cudaMalloc(&d_ys, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_yy, sizeof(double)));

    // Initialize to zero
    CUDA_CHECK(cudaMemsetAsync(d_ys, 0, sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(d_yy, 0, sizeof(double), stream));

    // Call kernel function
    dim3 threadsPerBlock(BLOCK_DIM);
    int numBlocks = (size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    // Limit number of blocks to 256
    // The grid-stride loop in low_rank_fused_kernel() will handle larger sizes
    numBlocks = std::min(numBlocks, 256);
    low_rank_fused_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_grad, d_grad_prev, d_gamma, d_gamma_prev, d_y, d_s, d_ys, d_yy, size
    );

    // Copy results to host
    CUDA_CHECK(cudaMemcpy(&ys, d_ys, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&yy, d_yy, sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_yy));
}

// CUDA kernels to compute search direction with low-rank terms
__global__ void search_direc_dot_kernel(
    const double* __restrict__ s,
    const double* __restrict__ g,
    const double* __restrict__ y,
    const double* __restrict__ invA_y,
    const double* __restrict__ invA_g,
    double* __restrict__ d_scalars,  // sg, yinvAy, yinvAg
    int size
)
{
    // Indices
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Local accumulators
    double local_sg = 0.0;
    double local_yinvAy = 0.0;
    double local_yinvAg = 0.0;

    // Grid-stride loop
    for (int i = idx; i < size; i += stride)
    {
        double yval = y[i];
        local_sg += s[i] * g[i];
        local_yinvAy += yval * invA_y[i];
        local_yinvAg += yval * invA_g[i];
    }

    // Parallel reduction
    local_sg = block_reduce_sum(local_sg);
    local_yinvAy = block_reduce_sum(local_yinvAy);
    local_yinvAg = block_reduce_sum(local_yinvAg);

    // Write to global memory
    // Only one thread needs to do this once
    if (tid == 0)
    {
        atomicAdd(&d_scalars[0], local_sg);
        atomicAdd(&d_scalars[1], local_yinvAy);
        atomicAdd(&d_scalars[2], local_yinvAg);
    }
}

__global__ void update_direc_kernel(
    double* __restrict__ direc,
    const double* __restrict__ invA_y,
    const double* __restrict__ s,
    const double* __restrict__ d_scalars,  // sg, yinvAy, yinvAg
    double ys,
    double reg,
    int size
)
{
    // Indices
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Only the first thread computes common values
    // Use shared memory broadcast to avoid reading global memory by every thread
    __shared__ double common_term1;
    __shared__ double common_term2;

    if (tid == 0)
    {
        double sg = d_scalars[0];
        double yinvAy = d_scalars[1];
        double yinvAg = d_scalars[2];
        double sg_ys = sg / ys;

        // direc += term1 * s - term2 * invA_y
        // term1 = (1 / reg + yinvAy / ys) * sg_ys - yinvAg / ys
        // term2 = sg_ys
        double term1 = (1.0 / reg + yinvAy / ys) * sg_ys - yinvAg / ys;
        
        common_term1 = term1;
        common_term2 = sg_ys;
    }
    // Wait for computing common values
    __syncthreads();

    // Grid-stride Loop
    double t1 = common_term1;
    double t2 = common_term2;
    // direc += t1 * s - t2 * invA_y
    for (int i = idx; i < size; i += stride)
    {
        direc[i] += t1 * s[i] - t2 * invA_y[i];
    }
}

// Helper function to compute search direction with low-rank terms
//
// 1. sg = sum(s * g)
// 2. yinvAy = sum(y * invA_y)
// 3. yinvAg = sum(y * invA_g), invA_g is an alias of direc
// 4. sg_ys = sg / ys
// 5. direc += ((1 / reg + yinvAy / ys) * sg_ys - yinvAg / ys) * s - sg_ys * invA_y
//
// In/Out: d_direc  [size]
// In: d_invA_y     [size]
// In: d_g          [size]
// In: d_y          [size]
// In: d_s          [size]
void launch_low_rank_search_direc(
    double* d_direc,
    const double* d_invA_y,
    const double* d_g,
    const double* d_y,
    const double* d_s,
    const double ys,
    const double reg,
    int size,
    cudaStream_t stream = cudaStreamPerThread
)
{
    // Allocate workspace
    double* d_work;
    CUDA_CHECK(cudaMalloc(&d_work, 3 * sizeof(double)));

    // Initialize to zero
    CUDA_CHECK(cudaMemsetAsync(d_work, 0, 3 * sizeof(double), stream));

    // Call first kernel function
    dim3 threadsPerBlock(BLOCK_DIM);
    int numBlocks = (size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    // Limit number of blocks to 256
    // The grid-stride loop in search_direc_dot_kernel() will handle larger sizes
    numBlocks = std::min(numBlocks, 256);
    search_direc_dot_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_s, d_g, d_y, d_invA_y, d_direc, 
        d_work, size
    );

    // Call second kernel function
    update_direc_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_direc, d_invA_y, d_s, 
        d_work, ys, reg, size
    );

    // Free workspace
    CUDA_CHECK(cudaFree(d_work));
}
