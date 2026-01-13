#include <iostream>
#include <cmath>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Thrust headers
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/inner_product.h>

// Utility functions
#include "utils.h"
#include "timer.h"
#include "sinkhorn.h"

// Linear solver
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

// Define block dimensions (16x16 = 256 threads)
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define BLOCK_DIM 256

// Helper function to compute objective function value objfn,
// gradient grad, and sparsified Hessian in CSR form
// From sinkhorn_splr_kernel.cu
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
);

// Helper function to compute low-rank vectors y and s
// From sinkhorn_splr_kernel.cu
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
);

// Helper function to compute search direction with low-rank terms
// From sinkhorn_splr_kernel.cu
void launch_low_rank_search_direc(
    double* d_direc,
    const double* d_invA_y,
    const double* d_g,
    const double* d_y,
    const double* d_s,
    const double ys,
    const double reg,
    int size
);

// Functor for computing z = a * x + y
template <typename T>
struct axpy_functor
{
    T m_a;
    axpy_functor(T a): m_a(a) {}

    __host__ __device__
    T operator()(const T& x, const T& y) const
    {
        return m_a * x + y;
    }
};

// Class for the SPLR solver
class SPLRSolver
{
private:
    // Problem dimensions
    const int     m_n;
    const int     m_m;
    const size_t  m_Me;
    const size_t  m_Te;
    const size_t  m_Hsize;
    const size_t  m_Kmax;
    // Regularization parameter
    const double  m_reg;
    // Input matrices and vectors on device
    const double* d_M;
    double*       d_M_storage;
    double*       d_ab;
    double*       d_logab;
    // Dual variables on device
    double*       d_gamma;
    double*       d_gamma_prev;
    // Pointer aliases, d_gamma = (d_alpha, d_beta)
    double*       d_alpha;
    double*       d_beta;
    // Objective function value and gradient
    double*       d_objfn;
    double*       d_grad;
    double*       d_grad_prev;
    // Search direction and low-rank vectors
    double*       d_direc;
    double*       d_y;
    double*       d_s;
    double*       d_invA_y;
    // Sparsified Hessian in CSR representation
    double*       d_Hvalues;
    int*          d_Hflatind;
    int*          d_Hcolind;
    int*          d_Hrowptr;
    // Working space
    double*       d_work;
    int*          d_iwork;
    // Sparse Cholesky solver
    SparseCholeskySolver m_linsolver;

    // Simple dot product
    inline double dot(const double* d_x, const double* d_y, int size) const
    {
        thrust::device_ptr<const double> d_x_ptr = thrust::device_pointer_cast(d_x);
        thrust::device_ptr<const double> d_y_ptr = thrust::device_pointer_cast(d_y);
        return thrust::inner_product(d_x_ptr, d_x_ptr + size, d_y_ptr, 0.0);
    }

    // Compute z = a * x + y
    inline void axpy(const double* d_x, const double* d_y, double a, int size, double* d_z) const
    {
        thrust::device_ptr<const double> d_x_ptr = thrust::device_pointer_cast(d_x);
        thrust::device_ptr<const double> d_y_ptr = thrust::device_pointer_cast(d_y);
        thrust::device_ptr<double> d_z_ptr = thrust::device_pointer_cast(d_z);

        thrust::transform(d_x_ptr, d_x_ptr + size,
                          d_y_ptr,
                          d_z_ptr,
                          axpy_functor<double>(a));
    }

public:
    // Constructor
    SPLRSolver(
        const double* M, const double* a, const double* b,
        double reg, int n, int m, size_t Kmax,
        bool input_on_device = false
    ):
        m_n(n), m_m(m), m_Me(n * m), m_Te(n * (m - 1)), m_Hsize(n + m - 1), m_Kmax(Kmax), m_reg(reg),
        d_M_storage(nullptr)
    {
        // If M is already on the device, then directly assign M to d_M
        // Otherwise, allocate device memory and copy from the host pointer
        double *d_M_storage = nullptr;
        if (input_on_device)
        {
            d_M = M;
        }
        else
        {
            // Allocate device memory
            CUDA_CHECK(cudaMalloc(&d_M_storage, m_Me * sizeof(double)));
            // Copy input to device
            CUDA_CHECK(cudaMemcpy(d_M_storage, M, m_Me * sizeof(double), cudaMemcpyHostToDevice));
            // Set input data pointers
            d_M = d_M_storage;
        }

        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_ab, (m_n + m_m) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_logab, (m_n + m_m) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_gamma, (m_n + m_m) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_gamma_prev, (m_n + m_m) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_objfn, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_grad, m_Hsize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_grad_prev, m_Hsize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_direc, m_Hsize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_y, m_Hsize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_s, m_Hsize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_invA_y, m_Hsize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Hvalues, (Kmax + m_Hsize) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Hflatind, (Kmax + m_Hsize) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_Hcolind, (Kmax + m_Hsize) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_Hrowptr, (m_Hsize + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_work, (m_n + m_m + 1 + m_Te) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_iwork, std::max(m_Te, m_Hsize) * sizeof(int)));

        // Pointer aliases
        d_alpha = d_gamma;
        d_beta = d_gamma + m_n;

        // Copy a and b to d_ab
        CUDA_CHECK(cudaMemcpy(
            d_ab, a, m_n * sizeof(double),
            input_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            d_ab + m_n, b, m_m * sizeof(double),
            input_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice
        ));

        // Compute log(a) and log(b)
        compute_log_vector_cuda(d_ab, d_logab, m_n + m_m);

        // Set d_grad_prev to zero
        CUDA_CHECK(cudaMemset(d_grad_prev, 0, m_Hsize * sizeof(double)));
    }

    // Destructor
    ~SPLRSolver()
    {
        // Free device memory
        if (d_M_storage != nullptr)
        {
            CUDA_CHECK(cudaFree(d_M_storage));
        }
        CUDA_CHECK(cudaFree(d_ab));
        CUDA_CHECK(cudaFree(d_logab));
        CUDA_CHECK(cudaFree(d_gamma));
        CUDA_CHECK(cudaFree(d_gamma_prev));
        CUDA_CHECK(cudaFree(d_objfn));
        CUDA_CHECK(cudaFree(d_grad));
        CUDA_CHECK(cudaFree(d_grad_prev));
        CUDA_CHECK(cudaFree(d_direc));
        CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_s));
        CUDA_CHECK(cudaFree(d_invA_y));
        CUDA_CHECK(cudaFree(d_Hvalues));
        CUDA_CHECK(cudaFree(d_Hflatind));
        CUDA_CHECK(cudaFree(d_Hcolind));
        CUDA_CHECK(cudaFree(d_Hrowptr));
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaFree(d_iwork));
    }

    // Sinkhorn iteration to update alpha
    void update_alpha()
    {
        // Get pointer for log(a)
        const double* d_loga = d_logab;

        // Optimal alpha given beta
        // d_alpha = d_gamma
        // d_beta = d_gamma + n
        compute_optimal_alpha(d_M, d_beta, d_loga, d_alpha, m_reg, m_n, m_m);
    }

    // Sinkhorn iteration to update beta
    void update_beta()
    {
        // Get pointer for log(b)
        const double* d_logb = d_logab + m_n;

        // Optimal beta given alpha
        // d_alpha = d_gamma
        // d_beta = d_gamma + n
        compute_optimal_beta(d_M, d_alpha, d_logb, d_beta, m_reg, m_n, m_m);
    }

    // Initialize dual variables
    void init_dual(const double* x0, bool input_on_device = false)
    {
        // Initialize dual variable gamma
        if (x0 != nullptr)
        {
            // Use provided initial values: x0 contains [alpha (n elements), beta (m elements)]
            // But note that we force beta[m-1]=0, so we do a shifting
            // alpha += beta[m-1], beta -= beta[m-1]

            // Copy x0 to d_gamma
            CUDA_CHECK(cudaMemcpy(
                d_gamma, x0, (m_n + m_m) * sizeof(double),
                input_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice
            ));
            // Get shift = beta[m-1] = gamma[n+m-1]
            double shift;
            CUDA_CHECK(cudaMemcpy(
                &shift, d_gamma + (m_n + m_m - 1), sizeof(double),
                cudaMemcpyDeviceToHost
            ));
            // alpha += shift
            thrust::device_ptr<double> d_gamma_ptr(d_gamma);
            thrust::transform(
                d_gamma_ptr,
                d_gamma_ptr + m_n,
                thrust::constant_iterator<double>(shift),
                d_gamma_ptr,
                thrust::plus<double>()
            );
            // beta -= shift
            thrust::transform(
                d_gamma_ptr + m_n,
                d_gamma_ptr + (m_n + m_m),
                thrust::constant_iterator<double>(shift),
                d_gamma_ptr + m_n,
                thrust::minus<double>()
            );
        }
        else
        {
            // If no initial values are provided, first set beta to zero,
            // and then compute alpha using BCD iteration
            CUDA_CHECK(cudaMemset(d_beta, 0, m_m * sizeof(double)));
            update_alpha();
        }

        // Initialize dual variable in previous iteration
        CUDA_CHECK(cudaMemset(d_gamma_prev, 0, (m_n + m_m) * sizeof(double)));
    }

    // Compute objective function value, gradient, and sparsified Hessian
    // Stage 1 computes objective function value and gradient
    // Stage 2 computes sparsified Hessian
    size_t dual_objfn_grad_sphess(
        double density, double shift, double& objfn,
        bool stage1 = true, bool stage2 = true,
        bool fixed_indices = false
    )
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
            m_reg, shift, m_n, m_m, K,
            d_objfn, d_grad,
            d_Hvalues, d_Hflatind, d_Hcolind, d_Hrowptr,
            d_work, d_iwork,
            stage1, stage2,
            fixed_indices
        );

        // Copy d_objfn to host
        CUDA_CHECK(cudaMemcpy(&objfn, d_objfn, sizeof(double), cudaMemcpyDeviceToHost));

        // Return number of nonzeros in sparsified Hessian
        size_t nnz = K + m_Hsize;
        return nnz;
    }

    // Get current gradient norm
    double grad_norm() const
    {
        return compute_l2_norm_cuda(d_grad, m_Hsize);
    }

    // Compute low-rank vectors
    void compute_low_rank(double& ys, double& yy)
    {
        // y = grad - grad_prev
        // s = gamma - gamma_prev
        // ys = y's
        // yy = y'y
        launch_low_rank(d_grad, d_grad_prev, d_gamma, d_gamma_prev, d_y, d_s, ys, yy, m_Hsize);
    }

    // Compute search direction (with low-rank term)
    void compute_search_direc(size_t nnz, double ys, bool low_rank = true, bool analyze_pattern = true, int verbose = 0)
    {
        // Solve (H + UCV) * d = -g
        // U = [u, v], C = diag(a, b), V = U'
        // u = y, v = H * s
        // a = 1 / u's, b = -1 / v's

        // BFGS update rule
        // https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm

        // inv(H + UCV) = inv(H) + (y's + y'inv(H)y)(ss') - inv(H)ys' + sy'inv(H)
        //                         ----------------------   ---------------------   
        //                                 (y's)^2                   y's
        //
        // Therefore,
        // d = -inv(H + UCV)g = -inv(H)g - (y's + y'inv(H)y)(s'g)s + (s'g)inv(H)y + sy'inv(H)g
        //                                 -----------------------   -------------------------
        //                                         (y's)^2                      y's
        //
        // Let d0 = -inv(H)g
        // If no low-rank term is used, then directly return d0

        // Note that we have actually stored A = reg * H,
        // so we first compute
        // x = inv(A)g + (y's / reg + y'inv(A)y)(s'g)s - (s'g)inv(A)y + sy'inv(A)g
        //               -----------------------------   -------------------------
        //                          (y's)^2                         y's
        // and then do the scaling d <- -reg * x
        
        // direc = invA_g = inv(A) * g;
        Timer timer(verbose >= 3);
        timer.tic();
        m_linsolver.set_A(d_Hvalues, d_Hcolind, d_Hrowptr, m_Hsize, nnz);
        m_linsolver.set_b(d_grad, m_Hsize);
        m_linsolver.set_x(d_direc, m_Hsize);

        if (analyze_pattern)
        {
            // For timing, we separate analyze() as reorder() + symfac()
            // Otherwise we run the combined analyze()
            if (verbose >= 3)
            {
                m_linsolver.reorder();
                timer.toc("reorder");
                m_linsolver.symfac();
                timer.toc("symfac");
            }
            else
            {
                m_linsolver.analyze();
            }
        }
        m_linsolver.factorize();
        timer.toc("factorize");
        m_linsolver.solve();
        timer.toc("solve");

        if (low_rank)
        {
            // invA_y = inv(A) * y;
            m_linsolver.set_b(d_y, m_Hsize);
            m_linsolver.set_x(d_invA_y, m_Hsize);
            m_linsolver.solve();

            // BFGS rule
            // sg = s'g, ys = y's
            // yinvAy = y'(invA_y), yinvAg = y'(invA_g) = y'(direc)
            // sg_ys = sg / ys
            // direc += ((1 / reg + yinvAy / ys) * sg_ys - yinvAg / ys) * s - sg_ys * invA_y
            launch_low_rank_search_direc(d_direc, d_invA_y, d_grad, d_y, d_s, ys, m_reg, m_Hsize);
        }
        timer.toc("low_rank");

        // Scaling d <- -reg * x
        thrust::constant_iterator<double> constant_iter(-m_reg);
        thrust::device_ptr<double> d_direc_ptr = thrust::device_pointer_cast(d_direc);
        thrust::transform(
            d_direc_ptr, d_direc_ptr + m_Hsize, constant_iter, d_direc_ptr,
            thrust::multiplies<double>()
        );
        timer.toc("scaling");

        if (verbose >= 3)
        {
            std::cout << "[search_direc_timing]--------------------------------------" << std::endl;
            std::cout << "║ reorder = " << timer["reorder"] << ", symfac = " << timer["symfac"] << std::endl;
            std::cout << "║ factorize = " << timer["factorize"] << ", solve = " << timer["solve"] << std::endl;
            std::cout << "║ low_rank = " << timer["low_rank"] << ", scaling = " << timer["scaling"] << std::endl;
            std::cout << "===========================================================" << std::endl;
        }
    }

    // Save d_gamma to d_gamma_prev, and d_grad to d_grad_prev
    void save_history()
    {
        CUDA_CHECK(cudaMemcpy(d_gamma_prev, d_gamma, m_Hsize * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_prev, d_grad, m_Hsize * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    // More-Thuente line search with Wolfe conditions
    double line_search_wolfe(
        double init_step, double cur_obj, bool& recompute_fg,
        bool fixed_indices = false,
        double c1 = 1e-4, double c2 = 0.9, int max_iter = 20
    )
    {
        // We assume d_gamma has been copied to d_gamma_prev,
        // so new point is computed as
        //     d_gamma = d_gamma_prev + step * d_direc
        // d_gamma_prev and d_direc are read-only during line search,
        // and d_gamma and d_grad will be overwritten

        // Typically the objective function value (f) and gradient (g)
        // have been computed on the new point when line search exits,
        // but there are cases that a different step is returned
        // In such cases, we flag recompute_fg = true
        recompute_fg = false;

        // Initial step size
        double step = init_step, step_max = 2.0;
        double fx = cur_obj, dg = dot(d_grad_prev, d_direc, m_Hsize);

        // Save the function value at the current x
        const double fx_init = cur_obj;
        // Projection of gradient on the search direction
        const double dg_init = dg;
        // Make sure d points to a descent direction
        if (dg_init > 0.0)
        {
            recompute_fg = true;
            return step;
        }

        // Tolerance for convergence test
        // Sufficient decrease
        const double test_decr = c1 * dg_init;
        // Curvature
        const double test_curv = -c2 * dg_init;

        // The bracketing interval
        double I_lo = 0.0, I_hi = std::numeric_limits<double>::infinity();
        double fI_lo = 0.0, fI_hi = std::numeric_limits<double>::infinity();
        double gI_lo = (1.0 - c1) * dg_init, gI_hi = std::numeric_limits<double>::infinity();
        double fx_lo = fx_init, dg_lo = dg_init;

        // Evaluate the current step size
        // gamma = gamma_prev + step * direc
        axpy(d_direc, d_gamma_prev, step, m_Hsize, d_gamma);
        // We only compute f and g, so label stage1 = true, stage2 = false
        // In this case shift and K are not used
        dual_objfn_grad_sphess(0.01, 0.001, fx, true, false, fixed_indices);
        // Get g'(direc)
        dg = dot(d_grad, d_direc, m_Hsize);

        // Convergence test
        if (fx <= fx_init + step * test_decr && abs(dg) <= test_curv)
        {
            return step;
        }

        // Extrapolation factor
        constexpr double delta = 1.1;
        int iter;
        for (iter = 0; iter < max_iter; iter++)
        {
            // ft = psi(step) = f(xp + step * drt) - f(xp) - step * test_decr
            // gt = psi'(step) = dg - mu * dg_init
            // mu = c1
            const double ft = fx - fx_init - step * test_decr;
            const double gt = dg - c1 * dg_init;

            // Update step size and bracketing interval
            double new_step;
            if (ft > fI_lo)
            {
                // Case 1: ft > fl
                new_step = step_selection(I_lo, I_hi, step, fI_lo, fI_hi, ft, gI_lo, gI_hi, gt);
                // Sanity check: if the computed new_step is too small, typically due to
                // extremely large value of ft, switch to the middle point
                if (new_step <= 1e-12)
                    new_step = (I_lo + step) / 2.0;

                I_hi = step;
                fI_hi = ft;
                gI_hi = gt;
            }
            else if (gt * (I_lo - step) > 0.0)
            {
                // Case 2: ft <= fl, gt * (al - at) > 0
                //
                // Page 291 of Moré and Thuente (1994) suggests that
                // newat = min(at + delta * (at - al), amax), delta in [1.1, 4]
                new_step = std::min(step_max, step + delta * (step - I_lo));

                I_lo = step;
                fI_lo = ft;
                gI_lo = gt;
                fx_lo = fx;
                dg_lo = dg;
            }
            else
            {
                // Case 3: ft <= fl, gt * (al - at) <= 0
                new_step = step_selection(I_lo, I_hi, step, fI_lo, fI_hi, ft, gI_lo, gI_hi, gt);

                I_hi = I_lo;
                fI_hi = fI_lo;
                gI_hi = gI_lo;

                I_lo = step;
                fI_lo = ft;
                gI_lo = gt;
                fx_lo = fx;
                dg_lo = dg;
            }

            // Case 1 and 3 are interpolations, whereas Case 2 is extrapolation
            // This means that Case 2 may return new_step = step_max,
            // and we need to decide whether to accept this value
            // 1. If both step and new_step equal to step_max, it means
            //    step will have no further change, so we accept it
            // 2. Otherwise, we need to test the function value and gradient
            //    on step_max, and decide later

            // In case step, new_step, and step_max are equal, directly return the computed x and fx
            if (step == step_max && new_step >= step_max)
            {
                return step;
            }
            // Otherwise, recompute x and fx based on new_step
            step = new_step;

            if (step < 1e-12 || step > 1e12)
            {
                recompute_fg = true;
                return init_step;
            }

            // Update parameter, function value, and gradient
            axpy(d_direc, d_gamma_prev, step, m_Hsize, d_gamma);
            dual_objfn_grad_sphess(0.01, 0.001, fx, true, false, fixed_indices);
            dg = dot(d_grad, d_direc, m_Hsize);

            // Convergence test
            if (fx <= fx_init + step * test_decr && abs(dg) <= test_curv)
            {
                return step;
            }

            // Now assume step = step_max, and we need to decide whether to
            // exit the line search (see the comments above regarding step_max)
            // If we reach here, it means this step size does not pass the convergence
            // test, so either the sufficient decrease condition or the curvature
            // condition is not met yet
            //
            // Typically the curvature condition is harder to meet, and it is
            // possible that no step size in [0, step_max] satisfies the condition
            //
            // But we need to make sure that its psi function value is smaller than
            // the best one so far. If not, go to the next iteration and find a better one
            if (step >= step_max)
            {
                const double ft_bound = fx - fx_init - step * test_decr;
                if (ft_bound <= fI_lo)
                {
                    return step;
                }
            }
        }

        // When we reach here, it means that the maximum number of iterations
        // have been attained
        // If we have used up all line search iterations, then the
        // strong Wolfe condition is not met
        // We choose not to raise an exception (unless no step satisfying
        // sufficient decrease is found), but to return the best step size so far
        //
        // First test whether the last step is better than I_lo
        // If yes, return the last step
        const double ft = fx - fx_init - step * test_decr;
        if (ft <= fI_lo)
            return step;

        // If not, then the best step size so far is I_lo, but it needs to be positive
        if (I_lo <= 0.0)
        {
            recompute_fg = true;
            return init_step;
        }

        // Return everything with _lo
        recompute_fg = true;
        step = I_lo;
        fx = fx_lo;
        dg = dg_lo;
        return step;
    }

    // Update iterate, d_gamma = d_gamma_prev + alpha * d_direc
    void update_gamma(double alpha)
    {
        axpy(d_direc, d_gamma_prev, alpha, m_Hsize, d_gamma);
    }

    // Output results to host -- transport plan and dual variables
    void output_result(double* P, double* dual, bool output_on_device = false)
    {
        // Copy (alpha, beta) to dual
        if (dual != nullptr)
        {
            CUDA_CHECK(cudaMemcpy(
                dual, d_gamma, (m_n + m_m) * sizeof(double),
                output_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
            ));
        }

        // In case P is nullptr
        if (P == nullptr)
        {
            return;
        }

        // Compute final transport plan
        // If P is on device, directly write to P
        // Otherwise, since d_work is no longer used,
        // and it has at least n*m elements, we can use d_work
        // to hold transport plan
        double* d_P = output_on_device ? P : d_work;
        compute_transport_plan(d_M, d_alpha, d_beta, d_P, m_reg, m_n, m_m);

        // Copy result back to host if output_on_device = false
        if (!output_on_device)
        {
            CUDA_CHECK(cudaMemcpy(P, d_P, m_Me * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }
};


// CUDA implementation of SPLR algorithm for entropic-regularized OT
// input_on_device = true means that M, a, b, and x0 (if not nullptr) are device pointers
// output_on_device = true means that P and dual (if not nullptr) are device pointers
void cuda_sinkhorn_splr(
    const double* M, const double* a, const double* b, double* P,
    double reg, int max_iter, double tol, int n, int m, int* niter,
    double density_max, double shift_max, int pattern_cycle, int verbose,
    const double* x0, double* dual,
    bool input_on_device, bool output_on_device
)
{
    // Algorithmic parameters
    // density
    density_max = std::min(density_max, 1.0);
    density_max = std::max(density_max, 0.0);
    const double density_min = 0.01 * density_max;
    double density = 0.1 * density_max;
    // shift
    double shift = shift_max;
    // Kmax -- maximum number of nonzero elements in sparsified T_t
    size_t Kmax = static_cast<size_t>(density_max * n * (m - 1));
    Kmax = std::max(Kmax, size_t(1));
    // pattern_cycle -- cycle length of reusing sparsity pattern
    // const int pattern_cycle = 30;

    // Create solver object
    SPLRSolver solver(M, a, b, reg, n, m, Kmax, input_on_device);

    // Initialize dual variables
    solver.init_dual(x0, input_on_device);

    // Initial objective function value (f), gradient (g),
    // and sparsified Hessian (sphess)
    double objfn;
    // Only compute f and g by setting stage1 = true, stage2 = false
    // Note: shift should be applied to Hessian, but what we compute is Hsl = H * reg
    // Therefore, here we multiply shift by reg before adding to Hsl
    solver.dual_objfn_grad_sphess(density, shift * reg, objfn, true, false);
    // Now we can compute ||grad||, which will be used for updating shift
    double gnorm = solver.grad_norm();
    double gnorm_init = gnorm;
    shift = std::min(gnorm, shift_max);
    // Then continue computing sphess by setting stage1 = false, stage2 = true
    size_t nnz = solver.dual_objfn_grad_sphess(density, shift * reg, objfn, false, true);

    // Main iteration
    // Initial step size
    double alpha = 1.0;
    // Timer
    Timer timer_inner(verbose >= 2);
    for (int iter = 0; iter < max_iter; iter++)
    {
        if (verbose >= 1)
        {
            std::cout << "iter = " << iter << ", objval = " << objfn <<
                ", ||grad|| = " << gnorm << std::endl;
        }

        // Start timing
        timer_inner.tic();

        // Convergence test
        // Also exit if objective function value is not finite
        if ((gnorm < tol) || (!std::isfinite(objfn)))
            break;

        // Compute y = grad - grad_prev and s = gamma - gamma_prev
        // ys = y's, yy = y'y
        double ys, yy;
        solver.compute_low_rank(ys, yy);
        timer_inner.toc("low_rank");

        // Compute search direction
        // We do not do low-rank update in the first iteration
        // When <y, s> is too small, don't use low-rank update
        constexpr double eps = 1e-6;  // Or use std::numeric_limits<double>::epsilon();
        const bool low_rank = (iter > 0) && (ys > (eps * yy));
        // Flags to indicate whether we need to recompute sparsity pattern
        const bool analyze_pattern = (iter % pattern_cycle == 0);
        const bool update_pattern = (iter % pattern_cycle == (pattern_cycle - 1));

        solver.compute_search_direc(nnz, ys, low_rank, analyze_pattern, verbose);
        timer_inner.toc("search_direc");

        // Line search will overwrite d_gamma and d_grad, so
        // save d_gamma to d_gamma_prev, and d_grad to d_grad_prev
        solver.save_history();

        // Wolfe Line Search
        // Will overwrite d_gamma, d_objfn, and d_grad
        // If the updated recompute_fg is false, then the overwritten d_gamma
        // is the new point, and d_objfn and d_grad contain the corresponding
        // objective function value and gradient, respectively
        // Otherwise, we need to recompute d_gamma, d_objfn, and d_grad
        bool recompute_fg = true;
        alpha = solver.line_search_wolfe(
            std::min(1.0, 1.5 * alpha), objfn, recompute_fg, !update_pattern
        );
        timer_inner.toc("line_search");

        // Recompute the new point if needed
        if (recompute_fg)
        {
            // d_gamma = d_gamma_prev + alpha * direc
            solver.update_gamma(alpha);
            // Compute f and g on new point d_gamma
            solver.dual_objfn_grad_sphess(density, shift * reg, objfn, true, false, !update_pattern);
        }
        timer_inner.toc("grad");

        // Adjust density according to gnorm change
        const double gnorm_pre = gnorm;
        gnorm = solver.grad_norm();
        if (update_pattern)
        {
            const bool bad_move = (gnorm_pre < gnorm_init) && (gnorm > gnorm_pre);
            density *= (bad_move ? 1.1 : 0.99);
            density = std::min(density_max, std::max(density_min, density));
        }

        // Compute sphess
        shift = std::min(gnorm, shift_max);
        nnz = solver.dual_objfn_grad_sphess(density, shift * reg, objfn, false, true, !update_pattern);
        timer_inner.toc("sphess");

        if (verbose >= 2)
        {
            std::cout << "[lowrank]--------------------------------------------------" << std::endl;
            std::cout << "║ ys = " << ys << ", yy = " << yy << std::endl;
            std::cout << "║ low_rank = " << low_rank << ", analyze = " << analyze_pattern <<
                ", update = " << update_pattern << std::endl;
            std::cout << "===========================================================" << std::endl;
            std::cout << "[timing]---------------------------------------------------" << std::endl;
            std::cout << "║ low_rank = " << timer_inner["low_rank"] <<
                ", search_direc = " << timer_inner["search_direc"] << std::endl;
            std::cout << "║ line_search = " << timer_inner["line_search"] << std::endl;
            std::cout << "║ grad = " << timer_inner["grad"] <<
                ", sphess = " << timer_inner["sphess"] << std::endl;
            std::cout << "===========================================================" << std::endl << std::endl;
        }

        *niter = iter + 1;
    }

    // Final Sinkhorn iteration
    // solver.update_beta();
    // solver.update_alpha();

    // Compute final transport plan and output results to host
    solver.output_result(P, dual, output_on_device);
    CUDA_CHECK(cudaDeviceSynchronize());
}
