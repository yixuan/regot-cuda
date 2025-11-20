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

// Class for the SPLR solver
class SPLRSolver
{
private:
    // Problem dimensions
    const int    m_n;
    const int    m_m;
    const size_t m_Me;
    const size_t m_Te;
    const size_t m_Hsize;
    const size_t m_Kmax;
    // Regularization parameter
    const double m_reg;
    // Input matrices and vectors on device
    double*      d_M;
    double*      d_ab;
    // Dual variables on device
    double*      d_gamma;
    double*      d_gamma_prev;
    // Pointer aliases, d_gamma = (d_alpha, d_beta)
    double*      d_alpha;
    double*      d_beta;
    // Objective function value and gradient
    double*      d_objfn;
    double*      d_grad;
    // Search direction
    double*      d_direc;
    // Sparsified Hessian in CSR representation
    double*      d_Hvalues;
    int*         d_Hcolind;
    int*         d_Hrowptr;
    // Working space
    double*      d_work;
    int*         d_iwork;

public:
    // Constructor
    SPLRSolver(const double* M, const double* a, const double* b, double reg, int n, int m, size_t Kmax):
        m_n(n), m_m(m), m_Me(n * m), m_Te(n * (m - 1)), m_Hsize(n + m - 1), m_Kmax(Kmax), m_reg(reg)
    {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_M, m_Me * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_ab, (m_n + m_m) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_gamma, (m_n + m_m) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_gamma_prev, (m_n + m_m) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_objfn, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_grad, m_Hsize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_direc, m_Hsize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Hvalues, (m_Te + m_Hsize) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Hcolind, (Kmax + m_Hsize) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_Hrowptr, (m_Hsize + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_work, (m_n + m_m + 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_iwork, (m_Te + 2 * m_Hsize) * sizeof(int)));

        // Pointer aliases
        d_alpha = d_gamma;
        d_beta = d_gamma + m_n;

        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_M, M, m_Me * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ab, a, m_n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ab + m_n, b, m_m * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Initialize dual variables
    void init_dual(const double* x0)
    {
        // Initialize dual variable gamma
        if (x0 != nullptr)
        {
            // Use provided initial values: x0 contains [alpha (n elements), beta (m elements)]
            // But note that we force beta[m-1]=0, so we do a shifting
            // alpha += beta[m-1], beta -= beta[m-1]
            double* gamma0 = new double[m_n + m_m];
            const double bm1 = x0[m_n + m_m - 1];
            for (int i = 0; i < m_n; i++)
            {
                gamma0[i] = x0[i] + bm1;
            }
            for (int i = 0; i < m_m; i++)
            {
                gamma0[m_n + i] = x0[m_n + i] - bm1;
            }
            CUDA_CHECK(cudaMemcpy(d_gamma, gamma0, (m_n + m_m) * sizeof(double), cudaMemcpyHostToDevice));
            delete [] gamma0;
        }
        else
        {
            // If no initial values are provided, first set beta to zero,
            // and then compute alpha using BCD iteration
            CUDA_CHECK(cudaMemset(d_beta, 0, m_m * sizeof(double)));

            // We also need log(a) vector
            double* d_loga;
            CUDA_CHECK(cudaMalloc(&d_loga, m_n * sizeof(double)));

            // Compute log(a) on device
            compute_log_vector_cuda(d_ab, d_loga, m_n);

            // Optimal alpha given beta
            // Configure kernel launch parameters
            dim3 threadsPerBlock(BLOCK_DIM);
            // Now each block handles one row, so we need n blocks for alpha
            dim3 numBlocks_alpha(m_n);
            // Calculate shared memory size
            size_t sharedMemory_alpha = threadsPerBlock.x * sizeof(double);
            // Optimal alpha given beta
            // d_alpha = d_gamma
            // d_beta = d_gamma + n
            optimal_alpha_kernel<<<numBlocks_alpha, threadsPerBlock, sharedMemory_alpha>>>(
                d_M, d_beta, d_loga, d_alpha, m_reg, m_n, m_m
            );

            // Free log(a) vector
            CUDA_CHECK(cudaFree(d_loga));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Initialize dual variable in previous iteration
        CUDA_CHECK(cudaMemset(d_gamma_prev, 0, (m_n + m_m) * sizeof(double)));
    }

    // Compute objective function value, gradient, and sparsified Hessian
    size_t dual_objfn_grad_sphess(double density)
    {
        // Make sure density is within (0, 1)
        density = std::min(density, 1.0);
        density = std::max(density, 0.0);

        // Keep K elements in T_t
        size_t K = static_cast<size_t>(density * m_Te);
        K = std::min(K, m_Kmax);
        K = std::max(K, size_t(1));

        // launch computation
        launch_objfn_grad_sphess(
            d_gamma, d_M, d_ab,
            m_reg, m_n, m_m, K,
            d_objfn, d_grad,
            d_Hvalues, d_Hcolind, d_Hrowptr,
            d_work, d_iwork
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Return number of nonzeros in sparsified Hessian
        size_t nnz = K + m_Hsize;
        return nnz;
    }

    // Get current gradient norm
    double grad_norm() const
    {
        return compute_l2_norm_cuda(d_grad, m_Hsize);
    }

    // Output results to host -- transport plan and dual variables
    void output_result(double* P, double* dual)
    {
        // Compute final transport plan
        // d_Hvalues is no longer used, and it has at least n*m elements
        // So we use d_Hvalues to hold transport plan
        dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
        dim3 gridDim;
        gridDim.x = (m_m + blockDim.x - 1) / blockDim.x;
        gridDim.y = (m_n + blockDim.y - 1) / blockDim.y;
        double* d_P = d_Hvalues;
        compute_transport_plan_kernel<<<gridDim, blockDim>>>(
            d_M, d_alpha, d_beta, d_P, m_reg, m_n, m_m
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back to host
        if (P != nullptr)
        {
            CUDA_CHECK(cudaMemcpy(P, d_P, m_Me * sizeof(double), cudaMemcpyDeviceToHost));
        }
        if (dual != nullptr)
        {
            CUDA_CHECK(cudaMemcpy(dual, d_gamma, (m_n + m_m) * sizeof(double), cudaMemcpyDeviceToHost));
        }
        
    }

    // Destructor
    ~SPLRSolver()
    {
        // Free device memory
        CUDA_CHECK(cudaFree(d_M));
        CUDA_CHECK(cudaFree(d_ab));
        CUDA_CHECK(cudaFree(d_gamma));
        CUDA_CHECK(cudaFree(d_gamma_prev));
        CUDA_CHECK(cudaFree(d_objfn));
        CUDA_CHECK(cudaFree(d_grad));
        CUDA_CHECK(cudaFree(d_direc));
        CUDA_CHECK(cudaFree(d_Hvalues));
        CUDA_CHECK(cudaFree(d_Hcolind));
        CUDA_CHECK(cudaFree(d_Hrowptr));
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaFree(d_iwork));
    }
};


// CUDA implementation of SPLR algorithm for entropic-regularized OT
void cuda_sinkhorn_splr(
    const double* M, const double* a, const double* b, double* P,
    double reg, int max_iter, double tol, int n, int m, int* niter,
    double density_max,
    const double* x0 = nullptr, double* dual = nullptr
)
{
    // Algorithmic parameters
    density_max = std::min(density_max, 1.0);
    density_max = std::max(density_max, 0.0);
    const double density_min = 0.01 * density_max;
    double density = 0.1 * density_max;
    size_t Kmax = static_cast<size_t>(density_max * n * (m - 1));
    Kmax = std::max(Kmax, size_t(1));

    // Create solver object
    SPLRSolver solver(M, a, b, reg, n, m, Kmax);

    // Initialize dual variables
    solver.init_dual(x0);

    // Initial objective function value, gradient, and sparsified Hessian
    size_t nnz = solver.dual_objfn_grad_sphess(density);
    double gnorm = solver.grad_norm();

    // Main iteration
    for (int iter = 0; iter < max_iter; iter++)
    {
        *niter = iter + 1;
    }

    // Compute final transport plan and output results to host
    solver.output_result(P, dual);
}
