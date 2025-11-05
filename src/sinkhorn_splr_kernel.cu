#include <iostream>
#include <cmath>

// CUDA Headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Thrust Headers
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

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

// Fused CUDA kernel for computation on T
// 1. Compute T[i,j] = exp(...)
// 2. Perform parallel reduction for row sums, column sums, and total sum using shared memory
// 3. Write modified T (last column = 0) to T_out
// 4. Fill flat_indices (0...n*m-1) for the subsequent Top-K selection
// @param T_out         [n*m]  Corresponds to Tvalues in the helper function
// @param row_sums      [n]    Corresponds to Trowsums
// @param col_sums      [m]    Corresponds to Tcolsums
// @param total_sum     [1]    Corresponds to Tsum
// @param flat_indices  [n*m]  Corresponds to indices
__global__ void T_fused_kernel(
    const double* __restrict__ alpha, 
    const double* __restrict__ beta, 
    const double* __restrict__ M, 
    double reg,
    int n, 
    int m,
    double* __restrict__ row_sums,
    double* __restrict__ col_sums,
    double* __restrict__ total_sum,
    double* __restrict__ T_out,
    int* __restrict__ flat_indices
)
{
    // Shared memory for partial reductions
    // One element per thread in the x-dimension of the block
    __shared__ double s_col[BLOCK_DIM_X];
    // One element per thread in the y-dimension of the block
    __shared__ double s_row[BLOCK_DIM_Y];
    // Block's partial sum
    __shared__ double s_block_sum;

    // Indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Global row index
    int i = blockIdx.y * BLOCK_DIM_Y + ty;
    // Global column index
    int j = blockIdx.x * BLOCK_DIM_X + tx;
    int flat_idx = i * m + j;

    // Initialize shared memory
    // Only one thread per row/column needs to do this
    if (tx == 0)
    {
        s_row[ty] = 0.0;
    }
    if (ty == 0)
    {
        s_col[tx] = 0.0;
    }
    if (tx == 0 && ty == 0)
    {
        s_block_sum = 0.0;
    }
    __syncthreads();

    // Boundary check and computation
    if (i < n && j < m) {
        // 1. Compute T[i, j]
        double T_ij = exp((alpha[i] + beta[j] - M[flat_idx]) / reg);

        // 2. Fill flat index
        flat_indices[flat_idx] = flat_idx;

        // 3. Accumulate to shared memory (should be fast)
        atomicAdd(&s_row[ty], T_ij);
        atomicAdd(&s_col[tx], T_ij);
        atomicAdd(&s_block_sum, T_ij);

        // 4. Write to T_out and modify the last column
        //    (The sums include the original value of the last column)
        T_out[flat_idx] = (j == m - 1) ? 0.0 : T_ij;
    }

    // Synchronize to ensure all shared memory writes are complete
    __syncthreads();

    // 5. Write partial sums back to global memory
    // The first column of threads (tx=0) writes back the row sums
    if (tx == 0 && i < n)
    {
        atomicAdd(&row_sums[i], s_row[ty]);
    }
    // The first row of threads (ty=0) writes back the column sums
    if (ty == 0 && j < m)
    {
        atomicAdd(&col_sums[j], s_col[tx]);
    }
    // Write back the block's partial sum
    // Only one thread per block does this
    if (tx == 0 && ty == 0)
    {
        atomicAdd(total_sum, s_block_sum);
    }
}

// Helper function to launch CUDA kernel on device
//
// Given alpha, beta, M (all on device), and reg:
// 1. Compute T matrix
// 2. Compute row/column/total sums of T
// 3. The largest K elements of T are stored in the first K elements in d_Tvalues
// 4. The corresponding (flattened) indices are stored in d_indices
// 5. The first K elements in d_indices are in ascending order
void launch_T_computation(
    const double* d_alpha,
    const double* d_beta,
    const double* d_M,
    double reg,
    int n, int m, int K,
    double* d_Trowsums, double* d_Tcolsums, double* d_Tsum,
    double* d_Tvalues, int* d_indices
)
{
    // Total number of elements
    size_t N_total = n * m;

    // Bound check
    size_t Ks = max(K, 1);
    Ks = min(Ks, N_total);

    // Step 1: Zero out the reduction arrays
    CUDA_CHECK(cudaMemset(d_Trowsums, 0, n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_Tcolsums, 0, m * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_Tsum, 0, sizeof(double)));

    // Step 2: Launch the fused kernel
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim;
    gridDim.x = (m + blockDim.x - 1) / blockDim.x;
    gridDim.y = (n + blockDim.y - 1) / blockDim.y;

    T_fused_kernel<<<gridDim, blockDim>>>(
        d_alpha, d_beta, d_M, reg, n, m,
        d_Trowsums, d_Tcolsums, d_Tsum, d_Tvalues, d_indices
    );
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    // Currently we do have a good Top-K implementation,
    // so we directly sort the T values
    // Step 3: Call thrust::sort_by_key to sort T values

    // Wrap raw device pointers with thrust::device_ptr
    thrust::device_ptr<double> d_Tvalues_ptr = thrust::device_pointer_cast(d_Tvalues);
    thrust::device_ptr<int> d_indices_ptr = thrust::device_pointer_cast(d_indices);

    // Perform the in-place sorting according to T values (descending order)
    thrust::sort_by_key(
        d_Tvalues_ptr,
        d_Tvalues_ptr + N_total,
        d_indices_ptr,
        ::cuda::std::greater<double>()
    );
    cudaDeviceSynchronize();

    // Step 4: Call thrust::sort_by_key on the first K elements to sort indices (ascending order)
    thrust::sort_by_key(
        d_indices_ptr,
        d_indices_ptr + Ks,
        d_Tvalues_ptr
    );
}

// Host function, mainly to test launch_T_computation()
extern "C" void T_computation(
    int nrun,
    const double* alpha,
    const double* beta,
    const double* M,
    double reg,
    int n, int m, int K,
    double* Trowsums, double* Tcolsums, double* Tsum,
    double* Tvalues, int* indices
)
{
    // Total number of elements
    size_t N_total = n * m;

    // Allocate device memory
    double *d_alpha, *d_beta, *d_M;
    double *d_Trowsums, *d_Tcolsums, *d_Tsum, *d_Tvalues;
    int *d_indices;

    CUDA_CHECK(cudaMalloc(&d_alpha, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beta, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_M, N_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Trowsums, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Tcolsums, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Tsum, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Tvalues, N_total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_indices, N_total * sizeof(int)));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_alpha, alpha, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta, m * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_M, M, N_total * sizeof(double), cudaMemcpyHostToDevice));

    // Launch computation
    // Multiple runs for benchmarking
    for (int i = 0; i < nrun; i++)
    {
        launch_T_computation(
            d_alpha, d_beta, d_M,
            reg, n, m, K,
            d_Trowsums, d_Tcolsums, d_Tsum,
            d_Tvalues, d_indices
        );
        cudaDeviceSynchronize();
    }

    // Step 5: Copy results back to host
    CUDA_CHECK(cudaMemcpy(Trowsums, d_Trowsums, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Tcolsums, d_Tcolsums, m * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Tsum, d_Tsum, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Tvalues, d_Tvalues, N_total * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(indices, d_indices, N_total * sizeof(int), cudaMemcpyDeviceToHost));

    // Step 5: Synchronize
    // Ensure all ops (kernel + Thrust) are complete before returning
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 6: Free device memory
    CUDA_CHECK(cudaFree(d_alpha));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_Trowsums));
    CUDA_CHECK(cudaFree(d_Tcolsums));
    CUDA_CHECK(cudaFree(d_Tsum));
    CUDA_CHECK(cudaFree(d_Tvalues));
    CUDA_CHECK(cudaFree(d_indices));
}
