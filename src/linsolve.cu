#include "linsolve.h"

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

// Define block dimensions (16x16 = 256 threads)
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define BLOCK_DIM 256

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
)
{
    // Create cuDSS data structures and handle initialization
    cudssHandle_t handle;
    cudssConfig_t config;
    cudssData_t data;
    cudssMatrix_t mat_A, vec_b, vec_x;
    CUDSS_CHECK(cudssCreate(&handle));

    // Configuration
    CUDSS_CHECK(cudssConfigCreate(&config));
    // Use CUDSS_ALG_DEFAULT
    cudssAlgType_t reorder_alg = CUDSS_ALG_DEFAULT;
    CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, (void*)&reorder_alg, sizeof(cudssAlgType_t)));

    // Matrix and data object creation
    CUDSS_CHECK(cudssDataCreate(handle, &data));
    // CUDSS_MTYPE_SPD for Cholesky decomposition
    CUDSS_CHECK(cudssMatrixCreateCsr(&mat_A, n, n, nnz, d_rowptr, nullptr, d_colind, d_values, CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SPD, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(&vec_b, n, 1, n, d_rhs, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&vec_x, n, 1, n, d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

    // Step 1: Symbolic analysis
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, mat_A, vec_x, vec_b));

    // Step 2: Factorization
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, mat_A, vec_x, vec_b));

    // Step 3: Solve
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, mat_A, vec_x, vec_b));

    // Step 4: Cleanup
    CUDSS_CHECK(cudssMatrixDestroy(mat_A));
    CUDSS_CHECK(cudssMatrixDestroy(vec_x));
    CUDSS_CHECK(cudssMatrixDestroy(vec_b));
    CUDSS_CHECK(cudssDataDestroy(handle, data));
    CUDSS_CHECK(cudssConfigDestroy(config));
    CUDSS_CHECK(cudssDestroy(handle));
}

// Host function, mainly to test sparse Cholesky solver
void sparse_cholesky_solve_host(
    const double* values,
    const int* colind,
    const int* rowptr,
    const double* rhs,
    double* x,
    int n,
    int nnz
)
{
    // Allocate device memory
    double *d_values, *d_rhs, *d_x;
    int *d_colind, *d_rowptr;

    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rhs, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_colind, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rowptr, (n + 1) * sizeof(int)));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_values, values, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs, rhs, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colind, colind, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowptr, rowptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Call device function
    sparse_cholesky_solve(d_values, d_colind, d_rowptr, d_rhs, d_x, n, nnz);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_rhs));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_colind));
    CUDA_CHECK(cudaFree(d_rowptr));
}
