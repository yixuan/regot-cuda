#!/usr/bin/env python3
"""
Test script for sparse Cholesky solver functionality
"""

import numpy as np
import scipy.sparse as sparse

# np.set_printoptions(linewidth=100)

def test_sparse_cholesky_random(n, density=0.01):
    """Test the sparse Cholesky solver with random data"""
    try:
        import curegot
        print("✓ Successfully imported curegot (CUDA) module")
    except ImportError as e:
        print(f"✗ Failed to import curegot: {e}")
        return

    # Create a random sparse square matrix
    spmat = sparse.random(n, n, density=density, format="coo")
    # print(spmat.toarray())

    # Get the lower part
    L = sparse.tril(spmat)

    # Get sparse positive definite matrix
    M = L.dot(L.T) + 0.1 * sparse.eye(n)
    M = sparse.csr_array(M)

    # Only the lower triangular part
    Mlower = sparse.tril(M)
    Mlower = sparse.csr_array(Mlower)

    # Get random rhs
    rhs = np.random.normal(size=n)

    # Get solution using Numpy/SciPy
    x = sparse.linalg.spsolve(M, rhs)

    try:
        # Solve the system
        print("\n  === Test Sparse Cholesky Solver ===")
        print(f"  n = {n}")
        xtest = curegot.test_sparse_cholesky_solve(
            M.data, M.indices, M.indptr, rhs)
        err1 = np.linalg.norm(xtest - x)
        print(f"  err = {err1}")
        
        # Using only the lower triangular part
        print("  === Using only the lower triangular part ===")
        xtest = curegot.test_sparse_cholesky_solve(
            Mlower.data, Mlower.indices, Mlower.indptr, rhs)
        err2 = np.linalg.norm(xtest - x)
        print(f"  err = {err2}")

        if err1 < 1e-8 and err2 < 1e-8:
            print("✓ Test PASSED!")
        else:
            print("✗ Test FAILED!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    np.random.seed(123)
    test_sparse_cholesky_random(10, density=0.1)
    print()
    test_sparse_cholesky_random(100, density=0.01)
    print()
    test_sparse_cholesky_random(1000, density=0.001)
    print()
    test_sparse_cholesky_random(10000, density=0.0001)
