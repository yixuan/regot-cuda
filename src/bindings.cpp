#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>

// CUDA function declaration
extern "C" void cuda_sinkhorn_bcd(
    const double* M, const double* a, const double* b, double* P,
    double reg, int max_iter, double tol, int n, int m, int* niter,
    const double* x0, double* dual
);

extern "C" void T_computation_sparsify(
    int nrun,
    const double* alpha,
    const double* beta,
    const double* M,
    double reg,
    int n, int m, int K,
    double* Trowsums, double* Tcolsums, double* Tsum,
    double* Tvalues, int* indices,
    double* csr_val, int* csr_rowptr, int* csr_colind
);

namespace py = pybind11;

// Python interface for sinkhorn_bcd function
py::dict sinkhorn_bcd(
    py::array_t<double> M,
    py::array_t<double> a,
    py::array_t<double> b,
    double reg,
    double tol,
    int max_iter,
    int verbose,
    const py::kwargs& kwargs
)
{
    // Get input array info
    py::buffer_info M_buf = M.request();
    py::buffer_info a_buf = a.request();
    py::buffer_info b_buf = b.request();

    if (M_buf.ndim != 2)
    {
        throw std::runtime_error("M must be a 2D array");
    }
    if (a_buf.ndim != 1 || b_buf.ndim != 1)
    {
        throw std::runtime_error("a and b must be 1D arrays");
    }

    int n = M_buf.shape[0];
    int m = M_buf.shape[1];

    if (a_buf.shape[0] != n || b_buf.shape[0] != m)
    {
        throw std::runtime_error("Shape mismatch: M is [n x m], a should be length n, b should be length m");
    }

    // Get raw pointers
    const double* M_ptr = static_cast<const double*>(M_buf.ptr);
    const double* a_ptr = static_cast<const double*>(a_buf.ptr);
    const double* b_ptr = static_cast<const double*>(b_buf.ptr);

    // Handle kwargs for initial values (x0)
    const double* x0_ptr = nullptr;
    std::vector<double> x0_storage;

    if (kwargs.contains("x0"))
    {
        py::array_t<double> x0 = py::cast<py::array_t<double>>(kwargs["x0"]);
        py::buffer_info x0_buf = x0.request();

        if (x0_buf.ndim != 1 || x0_buf.shape[0] != n + m)
        {
            throw std::runtime_error("x0 must be a 1D array of length n + m");
        }

        x0_storage.resize(n + m);
        const double* x0_data = static_cast<const double*>(x0_buf.ptr);
        std::copy(x0_data, x0_data + n + m, x0_storage.begin());
        x0_ptr = x0_storage.data();
    }

    // Create output arrays
    py::array_t<double> P = py::array_t<double>({n, m});
    py::buffer_info P_buf = P.request();
    double* P_ptr = static_cast<double*>(P_buf.ptr);

    py::array_t<double> dual = py::array_t<double>({n + m});
    py::buffer_info dual_buf = dual.request();
    double* dual_ptr = static_cast<double*>(dual_buf.ptr);

    // Call CUDA function for BCD algorithm
    int niter = 0;
    cuda_sinkhorn_bcd(M_ptr, a_ptr, b_ptr, P_ptr, reg, max_iter, tol, n, m, &niter, x0_ptr, dual_ptr);

    // Create result dictionary
    py::dict result;
    result["niter"] = niter;
    result["plan"] = P;
    result["dual"] = dual;

    if (verbose > 0)
    {
        std::cout << "sinkhorn_bcd (CUDA implementation) completed" << std::endl;
        std::cout << "Input shape: [" << n << " x " << m << "]" << std::endl;
        std::cout << "Regularization parameter: " << reg << std::endl;
        std::cout << "Tolerance: " << tol << std::endl;
        std::cout << "Max iterations: " << max_iter << std::endl;
        std::cout << "Actual iterations: " << niter << std::endl;
        if (niter == max_iter)
        {
            std::cout << "Warning: Maximum iterations reached without convergence" << std::endl;
        }
    }

    return result;
}

// Python interface for T_computation function with CSR output
py::dict test_T_computation(
    py::array_t<double> alpha,
    py::array_t<double> beta,
    py::array_t<double> M,
    double reg,
    int K,
    int nrun
)
{
    // Get input array info
    py::buffer_info alpha_buf = alpha.request();
    py::buffer_info beta_buf = beta.request();
    py::buffer_info M_buf = M.request();

    if (alpha_buf.ndim != 1 || beta_buf.ndim != 1)
    {
        throw std::runtime_error("alpha and beta must be 1D arrays");
    }
    if (M_buf.ndim != 2)
    {
        throw std::runtime_error("M must be a 2D array");
    }

    int n = alpha_buf.shape[0];
    int m = beta_buf.shape[0];

    if (M_buf.shape[0] != n || M_buf.shape[1] != m)
    {
        throw std::runtime_error("Shape mismatch: M should be [n x m] where n=len(alpha) and m=len(beta)");
    }

    // Bound check for K
    int Ks = std::max(K, 1);
    Ks = std::min(Ks, n * m);

    // Get raw pointers
    const double* alpha_ptr = static_cast<const double*>(alpha_buf.ptr);
    const double* beta_ptr = static_cast<const double*>(beta_buf.ptr);
    const double* M_ptr = static_cast<const double*>(M_buf.ptr);

    // Create output arrays for T computation
    py::array_t<double> Trowsums = py::array_t<double>(n);
    py::array_t<double> Tcolsums = py::array_t<double>(m);
    py::array_t<double> Tvalues = py::array_t<double>({n, m});
    py::array_t<int> indices = py::array_t<int>({n, m});

    // Create CSR output arrays
    py::array_t<double> val = py::array_t<double>(Ks);
    py::array_t<int> rowptr = py::array_t<int>(n + 1);
    py::array_t<int> colind = py::array_t<int>(Ks);

    py::buffer_info Trowsums_buf = Trowsums.request();
    py::buffer_info Tcolsums_buf = Tcolsums.request();
    py::buffer_info Tvalues_buf = Tvalues.request();
    py::buffer_info indices_buf = indices.request();
    py::buffer_info val_buf = val.request();
    py::buffer_info rowptr_buf = rowptr.request();
    py::buffer_info colind_buf = colind.request();

    double* Trowsums_ptr = static_cast<double*>(Trowsums_buf.ptr);
    double* Tcolsums_ptr = static_cast<double*>(Tcolsums_buf.ptr);
    double* Tvalues_ptr = static_cast<double*>(Tvalues_buf.ptr);
    int* indices_ptr = static_cast<int*>(indices_buf.ptr);
    double* val_ptr = static_cast<double*>(val_buf.ptr);
    int* rowptr_ptr = static_cast<int*>(rowptr_buf.ptr);
    int* colind_ptr = static_cast<int*>(colind_buf.ptr);

    // Call optimized CUDA function with CSR conversion
    double Tsum;
    T_computation_sparsify(
        nrun,
        alpha_ptr, beta_ptr, M_ptr, reg, n, m, Ks,
        Trowsums_ptr, Tcolsums_ptr, &Tsum, Tvalues_ptr, indices_ptr,
        val_ptr, rowptr_ptr, colind_ptr
    );

    // Create result dictionary
    py::dict result;
    result["Trowsums"] = Trowsums;
    result["Tcolsums"] = Tcolsums;
    result["Tsum"] = Tsum;
    result["Tvalues"] = Tvalues;
    result["indices"] = indices;

    // Add CSR format results
    result["csr_val"] = val;
    result["csr_rowptr"] = rowptr;
    result["csr_colind"] = colind;
    result["K_actual"] = Ks;

    return result;
}

// PYBIND11 module definition
PYBIND11_MODULE(_internal, m)
{
    m.doc() = "CUDA-accelerated Regularized Optimal Transport (RegOT-CUDA)";

    m.def("sinkhorn_bcd", &sinkhorn_bcd,
          py::arg("M"), py::arg("a"), py::arg("b"), py::arg("reg"),
          py::arg("tol") = 1e-6, py::arg("max_iter") = 1000, py::arg("verbose") = 0,
          "Sinkhorn Block Coordinate Descent algorithm (CUDA implementation)");

    m.def("test_T_computation", &test_T_computation,
          py::arg("alpha"), py::arg("beta"), py::arg("M"), py::arg("reg"), py::arg("K"),
          py::arg("nrun") = 1,
          "Test T computation (CUDA implementation)");
}
