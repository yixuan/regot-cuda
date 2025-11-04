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

// PYBIND11 module definition
PYBIND11_MODULE(_internal, m)
{
    m.doc() = "CUDA-accelerated Regularized Optimal Transport (RegOT-CUDA)";

    m.def("sinkhorn_bcd", &sinkhorn_bcd,
          py::arg("M"), py::arg("a"), py::arg("b"), py::arg("reg"),
          py::arg("tol") = 1e-6, py::arg("max_iter") = 1000, py::arg("verbose") = 0,
          "Sinkhorn Block Coordinate Descent algorithm (CUDA implementation)");
}
