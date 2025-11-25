#include <iostream>
#include <cmath>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUB headers
#include <cub/cub.cuh>

// Thrust headers
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

// cuDSS error checking macro
#define CUDSS_CHECK(call) \
    do { \
        cudssStatus_t err = call; \
        if (err != CUDSS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDSS Error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Define block dimensions (32x8 = 256 threads)
// BLOCK_DIM_X is the warp size 32, facilitating fast
// intra-warp reduction using __shfl_down_sync()
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8
#define BLOCK_DIM 256



// Background: we want to solve the dual problem of regularized optimal transport
//
// The dual variables are x = (alpha, beta_t), where beta_t means beta[:-1],
// and we always set beta = (beta_t, 0)
// alpha [n], beta [m], beta_t [m-1]
//
// The objective function (f), gradient (g), and Hessian (H) have closed-form expressions
// Let T[i, j] = exp((alpha[i] + beta[j] - M[i, j]) / reg), T_t = T[:, :-1]
// f = reg * sum(T) - <alpha, a> - <beta, b>
// g = (rowsum(T) - a, colsum(T_t) - b_t) = (Trowsums - a, Tcolsums_t - b_t)
// H = [diag(Trowsums)  T_t             ] / reg
//     [(T_t)'          diag(Tcolsums_t)]
//
// In the SPLR algorithm, we will sparsify T into a sparse matrix Tsp, so
// Hsp = [diag(Trowsums)  Tsp_t           ] / reg
//       [(Tsp_t)'        diag(Tcolsums_t)]
// Then we need to compute the CSR representation of the Hsp matrix
// For sparse Cholesky decomposition, we only need the lower triangular part, so
// it suffices to find the scaled and lower-triangular sparse matrix
// Hsl = [diag(Trowsums)  0               ]
//       [(Tsp_t)'        diag(Tcolsums_t)]
//
// To get the CSR representation of Hsl, we need to know the values of the
// nonzero elements and their locations in the matrix
// One way to do this is flattening the matrix and obtain two pointers:
//     Hvalues = [h0, h1, ..., hs],
//     Hindices = [i0, i1, ..., is],
// where Hindices contains the flattened indices of Hvalues, and we assume that
// Hindices is in ascending order
// Then the CSR representation of Hsl is
//     val = [h0, h1, ..., hs],
//     colind = [i0 % N, i1 % N, ..., is % N],
//     rowptr = [0, p[1], p[2], ..., p[N]], p[1] = #{i / N == 0},
//                                          p[2] - p[1] = #{i / N == 1},
//                                          p[3] - p[2] = #{i / N == 2}, ...,
// where N = n+m-1 is the size of Hsl
//
// Moreover, we know that Hvalues can be divided into three parts:
// 1. Trowsums = r = [r0, r1, ..., r[n-1]]
// 2. Tcolsums_t = c_t = [c0, c1, ..., c[m-2]]
// 3. Values of (Tsp_t)', Tvalues = [t0, t1, ..., t[K-1]]
// The corresponding indices are:
// 1. r indices: [i * N + i] = [i * (N + 1)]], i = 0, ..., n-1
// 2. c_t indices: [(n + j) * N + n + j] = [(n + j) * (N + 1)], j = 0, ..., m-2
// 3. Indices of (Tsp_t)', with the mapping
//        (i, j) in T -> (j, i) in T'/(T_t)' -> (n+j, i) in Hsl -> (n+j)*N+i for flattened indices

// To finish this task, we construct three core functions that are mainly run on GPU
// 1. A fused CUDA kernel that focuses on the computation on T:
//    (a) Tsum, Trowsums, and Tcolsums
//    (b) Flattened T' values and indices, which are used for downstream sparsification
//        We need to exclude the last row of T' in the output T_out
// 2. A function preparing the data for (Tsp_t)':
//    (a) Find the top-K elements in the output T_out array and the corresponding indices
//    (b) Put these K (Tvalues, Tindices)-pairs at the front of pointers
//    (c) Put (r, r indices) and (c_t, c_t indices) after the (Tvalues, T indices)-pairs
//    (d) Sort these (N+K) (val, ind)-pairs according to ind, so that the reordered val pointer
//        is exactly the CSR value pointer of Hsl
// 3. A function to compute the CSR pointers from flattened values and indices



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

// Fused CUDA kernel for computation on T
// 1. Compute T[i, j] = exp(...)
// 2. Perform parallel reduction for row sums, column sums, and total sum using shared memory
// 3. Write (T_t)' (T' excluding the last row) to T_out
// 4. Compute the flat indices in the Hsl matrix coordinates for the subsequent Top-K selection
//
// The option write_values_and_indices controls whether do steps 3 and 4
//
// In: alpha          [n]
// In: beta           [m]
// In: M              [n*m]
// Out: row_sums      [n]        Corresponds to Trowsums
// Out: col_sums      [m]        Corresponds to Tcolsums
// Out: total_sum     [1]        Corresponds to Tsum
// Out: T_out         [n*(m-1)]  Corresponds to values
// Out: flat_indices  [n*(m-1)]  Corresponds to indices
__global__ void T_fused_kernel(
    const double* __restrict__ alpha,
    const double* __restrict__ beta,
    const double* __restrict__ M,
    double reg,
    int n,
    int m,
    bool write_values_and_indices,
    double* __restrict__ row_sums,
    double* __restrict__ col_sums,
    double* __restrict__ total_sum,
    double* __restrict__ T_out,
    int* __restrict__ flat_indices
)
{
    // Shared memory for partial column sums
    // Each block contains an array of size BLOCK_DIM_X
    // s_col_sum[tx] collects the sum of the tx-th
    // column in this block
    // Each block column is of length BLOCK_DIM_Y
    __shared__ double s_col_sum[BLOCK_DIM_X];
    // Collects the sum of the entire block
    __shared__ double s_block_sum;

    // Indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int lane_id = tx % warpSize;
    // Global row index
    int i = blockIdx.y * blockDim.y + ty;
    // Global column index
    int j = blockIdx.x * blockDim.x + tx;

    // Initialize shared memory
    // Only one thread per column needs to do this
    // Let the first row handle this
    if (ty == 0)
    {
        s_col_sum[tx] = 0.0;
    }
    if (tx == 0 && ty == 0)
    {
        s_block_sum = 0.0;
    }
    __syncthreads();

    // Boundary check and computation
    double T_ij = 0.0;
    if (i < n && j < m)
    {
        // Flattened index reading M[i, j]
        int flat_idx_M = i * m + j;
        // We read T_t by row, and write to T_out [n*(m-1)]
        // T_t[i, j] -> T_out[i * (m-1) + j]
        int flat_idx_T_out = flat_idx_M - i;  // == i * (m - 1) + j;
        // Index of T_t[i, j] in flattened Hsl matrix
        int flat_idx_Hsl = (n + j) * (n + m - 1) + i;

        // 1. Compute T[i, j]
        T_ij = exp((alpha[i] + beta[j] - M[flat_idx_M]) / reg);

        // 2. Fill flat_indices and T_out only when j < m-1,
        //    excluding the last row of T' (last column of T)
        if (write_values_and_indices && j < m - 1)
        {
            flat_indices[flat_idx_T_out] = flat_idx_Hsl;
            T_out[flat_idx_T_out] = T_ij;
        }

        // 3. Accumulate to shared memory (should be fast)
        //    (The sums include all the original elements)
        atomicAdd(&s_col_sum[tx], T_ij);
    }

    // Fast reduction within the warp, which is exactly
    // one row in the block
    double row_partial_sum = warp_reduce_sum(T_ij);

    // 4. Write partial sums back to global memory
    // The warp leader (lane_id=0) writes back the row sums
    // to the global memory
    // At the same time, accumulate block row sums to block total sum
    if (lane_id == 0 && i < n)
    {
        atomicAdd(&row_sums[i], row_partial_sum);
        atomicAdd(&s_block_sum, row_partial_sum);
    }
    // Make sure s_col_sum and s_block_sum have finished
    __syncthreads();

    // The first row of threads (ty=0) writes back the column sums
    // to the global memory
    if (ty == 0 && j < m)
    {
        atomicAdd(&col_sums[j], s_col_sum[tx]);
    }
    // Write back the block's partial sum
    // Only one thread per block does this
    if (tx == 0 && ty == 0)
    {
        atomicAdd(total_sum, s_block_sum);
    }
}

// CUDA kernel to compute objective function value (f) and gradient (g)
//
// f = reg * sum(T) - <alpha, a> - <beta, b>
// g = (Trowsums - a, Tcolsums_t - b_t)
//
// In: ab        [n+m]       ab = (a, b)
// In: gamma     [n+m]       gamma = (alpha, beta)
// In: Trowsums  [n]
// In: Tcolsums  [m]
// In: Tsum      [1]
// Out: objfn    [1]
// Out: grad     [n+m-1]
__global__ void obj_grad_kernel(
    const double* __restrict__ ab,
    const double* __restrict__ gamma,
    const double* __restrict__ Trowsums,
    const double* __restrict__ Tcolsums,
    const double* __restrict__ Tsum,
    double reg,
    int n,
    int m,
    double* __restrict__ objfn,
    double* __restrict__ grad
)
{
    // Indices
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    int total_len = n + m;

    // Local variable to accumulate the dot product <gamma, ab> = <alpha, a> + <beta, b>
    double local_dotprod = 0.0;

    // Grid-stride loop instead of `if (idx < total_len)`
    for (int i = idx; i < total_len; i += stride)
    {
        double val_ab = ab[i];
        double val_gamma = gamma[i];

        // Task 1: Accumulate dot product
        local_dotprod += val_ab * val_gamma;

        // Task 2: Calculate gradient
        // Note that grad is only [n + m - 1]
        if (i < n)
        {
            // First n elements: Trowsums - a
            grad[i] = Trowsums[i] - val_ab;
        } 
        else if (i < n + m - 1)
        {
            // Next m-1 elements: Tcolsums_t - b_t
            // Current i corresponds to index (i - n) in b
            grad[i] = Tcolsums[i - n] - val_ab;
        }
        // Note that when i == n + m - 1 (i.e., the last element of b),
        // we do not write to grad, but still compute <gamma, ab>
    }

    // Parallel reduction to calculate the dot product
    local_dotprod = block_reduce_sum(local_dotprod);

    // Atomically accumulate block results to global objfn
    if (tid == 0)
    {
        // Let objfn hold -<gamma, ab>
        atomicAdd(objfn, -local_dotprod);
    }

    // Handle constant term reg * Tsum
    // Only one thread needs to do this once
    if (idx == 0)
    {
        atomicAdd(objfn, reg * (*Tsum));
    }
}

// CUDA kernel to write diagonal elements of Hsl to flattened value and index pointers
//
// In: Trowsums  [n]
// In: Tcolsums  [m]
// Out: values   [n+m-1]
// OUt: indices  [n+m-1]
__global__ void write_diagonal_kernel(
    const double* __restrict__ Trowsums, 
    const double* __restrict__ Tcolsums,
    double shift,
    int n, 
    int m,
    double* __restrict__ values,
    int* __restrict__ indices
)
{
    // Indices
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Write Trowsums[0...(n-1)] + shift to values[0...(n-1)]
    // Write i * (N + 1), i=0, ..., n-1 to indices[0...(n-1)]
    //
    // Write Tcolsums[0...(m-2)] + shift to (values+n)[0...(m-2)]
    // Write (n + j) * (N + 1), j=0, ..., m-2 to (indices+n)[0...(m-2)]
    int Hsize = n + m - 1;
    if (idx < Hsize)
    {
        values[idx] = (idx < n) ? (Trowsums[idx] + shift) : (Tcolsums[idx - n] + shift);
        indices[idx] = idx * (Hsize + 1);
    }
}

// Helper function to launch CUDA computations on T
//
// Given gamma = (alpha, beta), M, ab = (a, b) (all on device), and reg:
// 1. Compute T matrix
// 2. Compute row/column/total sums of T
// 3. Compute objective function value objfn and gradient grad
//
// In: d_gamma      [n+m]        d_gamma = (d_alpha, d_beta)
// In: d_M          [n*m]
// In: d_ab         [n+m]
// Out: d_Trowsums  [n]
// Out: d_Tcolsums  [m]
// Out: d_Tsum      [1]
// Out: d_objfn     [1]
// Out: d_grad      [n+m-1]
// Out: d_values    [n*(m-1)]
// Out: d_indices   [n*(m-1)]
void launch_T_computation(
    const double* d_gamma,
    const double* d_M,
    const double* d_ab,
    double reg,
    int n, int m,
    bool write_values_and_indices,
    double* d_Trowsums, double* d_Tcolsums, double* d_Tsum,
    double* d_objfn, double* d_grad,
    double* d_values, int* d_indices
)
{
    // Pointer aliases
    const double* d_alpha = d_gamma;
    const double* d_beta = d_gamma + n;

    // Zero out the reduction arrays
    CUDA_CHECK(cudaMemset(d_Trowsums, 0, n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_Tcolsums, 0, m * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_Tsum, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_objfn, 0, sizeof(double)));

    // Launch the fused kernel
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim;
    gridDim.x = (m + blockDim.x - 1) / blockDim.x;
    gridDim.y = (n + blockDim.y - 1) / blockDim.y;

    T_fused_kernel<<<gridDim, blockDim>>>(
        d_alpha, d_beta, d_M, reg, n, m, write_values_and_indices,
        d_Trowsums, d_Tcolsums, d_Tsum, d_values, d_indices
    );

    // Compute objfn and grad
    dim3 threadsPerBlock(BLOCK_DIM);
    dim3 numBlocks_obj_grad((n + m + threadsPerBlock.x - 1) / threadsPerBlock.x);
    obj_grad_kernel<<<numBlocks_obj_grad, threadsPerBlock>>>(
        d_ab, d_gamma, d_Trowsums, d_Tcolsums, d_Tsum,
        reg, n, m,
        d_objfn, d_grad
    );
}

// Helper function to compute sparse representation of H
//
// (continued from launch_T_computation())
// 4. The largest K elements of (T_t)' are written into the first K elements of d_Hvalues
// 5. The corresponding (flattened) indices are written into d_Hflatind
// 6. Add diagonal elements of Hsl matrix (plus a shift) to d_Hvalues and corresponding indices to d_Hflatind
//    (overwrite d_Hvalues[K:] and d_Hflatind[K:])
// 7. The first (K+Hsize) elements in d_Hflatind are in ascending order, Hsize = n+m-1
//
// In: d_values     [n*(m-1)]
// In: d_indices    [n*(m-1)]
// Out: d_Hvalues   [nnz] = [K+Hsize], reserve [Kmax+Hsize]
// Out: d_Hflatind  [nnz], reserve [Kmax+Hsize]
void launch_H_sparsification(
    const double* d_values,
    const int* d_indices,
    const double* d_Trowsums,
    const double* d_Tcolsums,
    int n, int m, int K,
    double shift,
    double* d_Hvalues,
    int* d_Hflatind
)
{
    // Total number of T elements
    size_t Te = n * (m - 1);
    size_t Hsize = n + m - 1;

    // Bound check
    size_t Ks = max(K, 1);
    Ks = min(Ks, Te);

    // We use cub::DeviceTopK::MaxPairs to find the largest K values in d_values,
    // and get the corresponding elements in d_indices

    // Set required environment
    auto requirements = cuda::execution::require(
        cuda::execution::determinism::not_guaranteed,
        cuda::execution::output_ordering::unsorted
    );
    auto env = cuda::std::execution::env{requirements};

    // Get memory size for DeviceTopK
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceTopK::MaxPairs(
        d_temp_storage, temp_storage_bytes,
        d_values, d_Hvalues,
        d_indices, d_Hflatind,
        Te, Ks,
        env
    ));
    // Allocate memory
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run DeviceTopK
    CUDA_CHECK(cub::DeviceTopK::MaxPairs(
        d_temp_storage, temp_storage_bytes,
        d_values, d_Hvalues,
        d_indices, d_Hflatind,
        Te, Ks,
        env
    ));
    // Free memory
    CUDA_CHECK(cudaFree(d_temp_storage));

    // Add diagonal elements of Hsl to (d_Hvalues, d_Hflatind)
    dim3 threadsPerBlock(BLOCK_DIM);
    dim3 numBlocks_write_diagonal((Hsize + threadsPerBlock.x - 1) / threadsPerBlock.x);
    write_diagonal_kernel<<<numBlocks_write_diagonal, threadsPerBlock>>>(
        d_Trowsums, d_Tcolsums, shift, n, m, d_Hvalues + Ks, d_Hflatind + Ks
    );

    // Wrap raw device pointers with thrust::device_ptr
    thrust::device_ptr<double> d_Hvalues_ptr = thrust::device_pointer_cast(d_Hvalues);
    thrust::device_ptr<int> d_Hflatind_ptr = thrust::device_pointer_cast(d_Hflatind);

    // Call thrust::sort_by_key to sort d_Hflatind in ascending order
    thrust::sort_by_key(
        d_Hflatind_ptr,
        d_Hflatind_ptr + (Ks + Hsize),
        d_Hvalues_ptr
    );
}

// CUDA kernel to recompute nonzero elements according to existing sparsity pattern
//
// In: flatind  [K+Hsize]
// Out: values  [K+Hsize]
__global__ void recompute_nonzero_values_kernel(
    const int* __restrict__ flatind,
    const double* __restrict__ Trowsums, 
    const double* __restrict__ Tcolsums,
    const double* __restrict__ alpha,
    const double* __restrict__ beta,
    const double* __restrict__ M,
    double reg,
    double shift,
    int n,
    int m,
    int K,
    double* __restrict__ values
)
{
    // Indices
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    int Hsize = n + m - 1;
    if (idx < K + Hsize)
    {
        // Index is based on Hsl (flattened)
        int flat_ind_Hsl = flatind[idx];

        // Convert to (i, j) in Hsl
        int i = flat_ind_Hsl / Hsize;
        int j = flat_ind_Hsl % Hsize;

        double Hval = 0.0;
        if (i < n)
        {
            // Case 1: i < n -- diagonal elements of Hsl[:n, :n]
            //                  Hsl[i, i] = Trowsums[i]
            Hval = Trowsums[i] + shift;
        }
        else if (j < n)
        {
            // Case 2: i >= n, j < n -- Hsl[i, j] = T'[i - n, j] = T[j, i - n]
            const int Mind = j * m + (i - n);
            Hval = exp((alpha[j] + beta[i - n] - M[Mind]) / reg);
        }
        else
        {
            // Case 3: j >= n -- diagonal elements of Hsl[n:, n:]
            Hval = Tcolsums[j - n] + shift;
        }
        values[idx] = Hval;
    }
}

// CUDA kernel to extract column indices and count the number of elements per row
//
// In: flatind      [nnz]
// Out: colind      [nnz]
// Out: row_counts  [cols]
__global__ void extract_columns_and_count_kernel(
    const int* __restrict__ flatind,
    int* __restrict__ colind,
    int* __restrict__ row_counts,
    int nnz, int cols
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nnz)
    {
        // Convert flattened index to row and column
        int flat_idx = flatind[idx];
        int row = flat_idx / cols;
        int col = flat_idx % cols;

        // Set column index
        colind[idx] = col;

        // Increment row count atomically
        atomicAdd(&row_counts[row], 1);
    }
}

// Helper function to convert sorted flat indices to CSR format
//
// In: d_Hflatind               [nnz] = [K+Hsize] = [K+n+m-1]
// Out: d_colind                [nnz]
// Out: d_rowptr                [Hsize + 1] = [n+m]
// Working space: d_row_counts  [Hsize] = [n+m-1]
void launch_csr_conversion(
    const int* d_Hflatind,
    int* d_colind,
    int* d_rowptr,
    int* d_row_counts,
    int K, int n, int m
)
{
    // Initialize d_row_counts to zero vector
    int Hsize = n + m - 1;
    int nnz = K + Hsize;
    CUDA_CHECK(cudaMemset(d_row_counts, 0, Hsize * sizeof(int)));

    // Extract column indices and count the number of elements per row
    dim3 threadsPerBlock(BLOCK_DIM);
    dim3 numBlocks((nnz + threadsPerBlock.x - 1) / threadsPerBlock.x);

    extract_columns_and_count_kernel<<<numBlocks, threadsPerBlock>>>(
        d_Hflatind, d_colind, d_row_counts, nnz, Hsize
    );

    // Compute row pointers using inclusive sum
    // [a, b, c] -> [a, a + b, a + b + c]

    // The first element of row pointer is always zero
    CUDA_CHECK(cudaMemset(d_rowptr, 0, sizeof(int)));
    // Get memory size for InclusiveSum
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes, d_row_counts, d_rowptr + 1, Hsize
    ));
    // Allocate memory
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run InclusiveSum
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes, d_row_counts, d_rowptr + 1, Hsize
    ));
    // Free memory
    CUDA_CHECK(cudaFree(d_temp_storage));
}

// Helper function to compute objective function value objfn,
// gradient grad, and sparsified Hessian (plus a shift) in CSR form
//
// In: d_gamma      [n+m]  d_gamma = (d_alpha, d_beta)
// In: d_M          [n*m]
// In: d_ab         [n+m]
// Out: d_objfn     [1]
// Out: d_grad      [Hsize] = [n+m-1]
// Out: d_Hvalues   [nnz] = [K+Hsize], reserve [Kmax+Hsize]
// Out: d_Hflatind  [nnz], reserve [Kmax+Hsize]
// Out: d_Hcolind   [nnz], reserve [Kmax+Hsize]
// Out: d_Hrowptr   [Hsize + 1] = [n+m]
// Working space: d_work = (d_Trowsums, d_Tcolsums, d_Tsum, d_values), [n+m+1+n*(m-1)]
// Working space: d_iwork = d_indices/d_row_counts, [max(n*(m-1), n+m-1)]
//
// Stage 1 computes objective function value and gradient
// Stage 2 computes sparsified Hessian
//
// If fixed_indices = true, it means:
// 1. The first (K+Hsize) elements of d_Hflatind contain the sparsify pattern of Hsl
// 2. We keep d_Hflatind unchanged, and compute the corresponding Hsl values
// 3. Write these Hsl values to d_Hvalues
// 4. Continue computing d_Hcolind and d_Hrowptr
void launch_objfn_grad_sphess(
    const double* d_gamma,
    const double* d_M,
    const double* d_ab,
    double reg, double shift,
    int n, int m, int K,
    double* d_objfn, double* d_grad,
    double* d_Hvalues, int* d_Hflatind, int* d_Hcolind, int* d_Hrowptr,
    double* d_work, int* d_iwork,
    bool stage1 = true, bool stage2 = true,
    bool fixed_indices = false
)
{
    // Pointer aliases
    const double* d_alpha = d_gamma;
    const double* d_beta = d_gamma + n;
    double* d_Trowsums = d_work;
    double* d_Tcolsums = d_work + n;
    double* d_Tsum = d_work + (n + m);
    double* d_values = d_work + (n + m + 1);
    int* d_indices = d_iwork;

    if (stage1)
    {
        // If indices are fixed, we do not need to compute
        // d_values and d_indices in the first stage
        bool write_values_and_indices = (!fixed_indices);
        launch_T_computation(
            d_gamma, d_M, d_ab,
            reg, n, m,
            write_values_and_indices,
            d_Trowsums, d_Tcolsums, d_Tsum,
            d_objfn, d_grad,
            d_values, d_indices
        );
    }

    if (stage2)
    {
        // Dimensions
        size_t Te = n * (m - 1);
        size_t Hsize = n + m - 1;
        size_t Ks = max(K, 1);
        Ks = min(Ks, Te);
        size_t KHsize = Ks + Hsize;

        // Get d_Hvalues and d_Hflatind
        if (fixed_indices)
        {
            // If indices are fixed, we directly recompute d_Hvalues
            // according to d_Hflatind

            // The first (K+Hsize) elements of d_Hflatind are already given
            // Now write elements of Hsl to d_Hvalues
            dim3 threadsPerBlock(BLOCK_DIM);
            dim3 numBlocks_recompute_nonzero_values((KHsize + threadsPerBlock.x - 1) / threadsPerBlock.x);
            recompute_nonzero_values_kernel<<<numBlocks_recompute_nonzero_values, threadsPerBlock>>>(
                d_Hflatind, d_Trowsums, d_Tcolsums, d_alpha, d_beta, d_M,
                reg, shift, n, m, Ks, d_Hvalues
            );
        }
        else
        {
            // Otherwise, call launch_H_sparsification to get
            // d_Hvalues and d_Hflatind
            launch_H_sparsification(
                d_values, d_indices,
                d_Trowsums, d_Tcolsums,
                n, m, Ks, shift,
                d_Hvalues, d_Hflatind
            );
        }

        // Finally, call launch_csr_conversion to compute
        // d_Hcolind and d_Hrowptr

        // Now d_indices is no longer used, so we can reuse
        // d_iwork for the working space
        int* d_row_counts = d_iwork;
        launch_csr_conversion(
            d_Hflatind, d_Hcolind, d_Hrowptr, d_row_counts, Ks, n, m
        );
    }
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
    int size
)
{
    // Allocate device memory for scalars
    double* d_ys;
    double* d_yy;
    CUDA_CHECK(cudaMalloc(&d_ys, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_yy, sizeof(double)));

    // Initialize to zero
    CUDA_CHECK(cudaMemset(d_ys, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_yy, 0, sizeof(double)));

    // Call kernel function
    dim3 threadsPerBlock(BLOCK_DIM);
    int numBlocks = (size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    // Limit number of blocks to 256
    // The grid-stride loop in low_rank_fused_kernel()
    // will handle larger sizes
    numBlocks = std::min(numBlocks, 256);
    low_rank_fused_kernel<<<numBlocks, threadsPerBlock>>>(
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
        // term1 = (1 + yinvAy / ys) * sg_ys - yinvAg / ys
        // term2 = sg_ys
        double term1 = (1.0 + yinvAy / ys) * sg_ys - yinvAg / ys;
        
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
// 5. direc += ((1 + yinvAy / ys) * sg_ys - yinvAg / ys) * s - sg_ys * invA_y
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
    int size
)
{
    // Allocate workspace
    double* d_work;
    CUDA_CHECK(cudaMalloc(&d_work, 3 * sizeof(double)));

    // Initialize to zero
    CUDA_CHECK(cudaMemset(d_work, 0, 3 * sizeof(double)));

    // Call first kernel function
    dim3 threadsPerBlock(BLOCK_DIM);
    int numBlocks = (size + threadsPerBlock.x - 1) / threadsPerBlock.x;
    // Limit number of blocks to 256
    // The grid-stride loop in search_direc_dot_kernel()
    // will handle larger sizes
    numBlocks = std::min(numBlocks, 256);
    search_direc_dot_kernel<<<numBlocks, threadsPerBlock>>>(
        d_s, d_g, d_y, d_invA_y, d_direc, 
        d_work, size
    );

    // Call second kernel function
    update_direc_kernel<<<numBlocks, threadsPerBlock>>>(
        d_direc, d_invA_y, d_s, 
        d_work, ys, size
    );

    // Free workspace
    CUDA_CHECK(cudaFree(d_work));
}

// Host function, mainly to test T computation and CSR conversion
void T_computation_sparsify_host(
    int nrun,
    const double* alpha,
    const double* beta,
    const double* M,
    const double* a,
    const double* b,
    double reg, double shift,
    int n, int m, int K,
    double* Trowsums, double* Tcolsums, double* Tsum,
    double* objfn, double* grad,
    double* values, int* indices,
    double* csr_val, int* csr_rowptr, int* csr_colind
)
{
    // Total number of elements of M and T_t
    size_t Me = n * m;
    size_t Te = n * (m - 1);

    // Bound check for K
    size_t Ks = max(K, 1);
    Ks = min(Ks, Te);

    // Size of Hsl
    size_t Hsize = n + m - 1;

    // Number of nonzero elements in Hsl
    size_t nnz = Ks + Hsize;

    // Total number of elements for values and indices
    // In the extreme case, T_t plus diagonal elements of Hsl
    size_t N_total = Te + Hsize;

    // Allocate device memory
    double *d_gamma, *d_M, *d_ab;
    double *d_Trowsums, *d_Tcolsums, *d_Tsum, *d_values;
    double *d_objfn, *d_grad;
    double *d_Hvalues;
    int *d_indices, *d_Hflatind, *d_csr_rowptr, *d_csr_colind, *d_row_counts;

    CUDA_CHECK(cudaMalloc(&d_gamma, (n + m) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_M, Me * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ab, (n + m) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Trowsums, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Tcolsums, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Tsum, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_objfn, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad, Hsize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_values, Te * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_indices, Te * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Hvalues, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Hflatind, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_colind, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_rowptr, (Hsize + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_counts, Hsize * sizeof(int)));

    // Pointer aliases
    double* d_alpha = d_gamma;
    double* d_beta = d_gamma + n;
    double* d_a = d_ab;
    double* d_b = d_ab + n;

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_alpha, alpha, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta, m * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_M, M, Me * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, m * sizeof(double), cudaMemcpyHostToDevice));

    // Launch computation
    // Multiple runs for benchmarking
    for (int i = 0; i < nrun; i++)
    {
        launch_T_computation(
            d_gamma, d_M, d_ab,
            reg, n, m,
            true,
            d_Trowsums, d_Tcolsums, d_Tsum,
            d_objfn, d_grad,
            d_values, d_indices
        );
        launch_H_sparsification(
            d_values, d_indices,
            d_Trowsums, d_Tcolsums,
            n, m, Ks, shift,
            d_Hvalues, d_Hflatind
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
    }

    // Convert flattened elements to CSR format
    launch_csr_conversion(
        d_Hflatind, d_csr_colind, d_csr_rowptr, d_row_counts, Ks, n, m
    );

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(Trowsums, d_Trowsums, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Tcolsums, d_Tcolsums, m * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Tsum, d_Tsum, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(objfn, d_objfn, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grad, d_grad, Hsize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(values, d_values, Te * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(indices, d_indices, Te * sizeof(int), cudaMemcpyDeviceToHost));

    // Copy CSR results
    CUDA_CHECK(cudaMemcpy(csr_val, d_Hvalues, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(csr_colind, d_csr_colind, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(csr_rowptr, d_csr_rowptr, (Hsize + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_ab));
    CUDA_CHECK(cudaFree(d_Trowsums));
    CUDA_CHECK(cudaFree(d_Tcolsums));
    CUDA_CHECK(cudaFree(d_Tsum));
    CUDA_CHECK(cudaFree(d_objfn));
    CUDA_CHECK(cudaFree(d_grad));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_Hvalues));
    CUDA_CHECK(cudaFree(d_Hflatind));
    CUDA_CHECK(cudaFree(d_csr_colind));
    CUDA_CHECK(cudaFree(d_csr_rowptr));
    CUDA_CHECK(cudaFree(d_row_counts));
}
