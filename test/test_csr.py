#!/usr/bin/env python3
"""
Test script for CSR conversion functionality
"""

import numpy as np
from scipy.sparse import csr_matrix

def test_csr_conversion():
    """Test the CSR conversion functionality"""
    try:
        import curegot
        print("✓ Successfully imported curegot (CUDA) module")
    except ImportError as e:
        print(f"✗ Failed to import curegot: {e}")
        return

    # Simple test case
    n, m, K = 8, 4, 10
    reg = 0.1

    # Create test data
    alpha = np.random.normal(scale=0.1, size=n)
    beta = np.random.normal(scale=0.1, size=m)
    M = np.abs(np.random.normal(scale=0.1, size=(n, m)))
    T = np.exp((alpha.reshape(n, 1) + beta.reshape(1, m) - M) / reg)

    print("=== Input Data ===")
    print(f"Input shape: {n} x {m}")
    print(f"Regularization parameter: {reg}")
    print(f"K (top elements requested): {K}")
    print(f"alpha: {alpha}")
    print(f"beta: {beta}")
    print("M matrix:")
    print(M)
    print()
    print("T matrix:")
    print(T)
    Tsum = np.sum(T)
    Trowsums = np.sum(T, axis=1)
    Tcolsums = np.sum(T, axis=0)
    print(f"T total sum: {Tsum:.6f}")
    print(f"T row sums: {Trowsums}")
    print(f"T column sums: {Tcolsums}")
    print()

    # Exclude last column
    T[:, -1] = 0
    topk = np.argpartition(T.flatten(), K)[-K:]
    Tsp = np.zeros(n * m)
    Tsp[topk] = T.flatten()[topk]
    Tsp_d = Tsp.reshape(n, m)
    Tsp = csr_matrix(Tsp_d.reshape(n, m))
    print("T sparsified:")
    print(Tsp_d)
    print(f"Data: {Tsp.data}")
    print(f"Col index: {Tsp.indices}")
    print(f"Row pointer: {Tsp.indptr}")

    try:
        # Call the function
        result = curegot.test_T_computation(alpha, beta, M, reg, K, nrun=1)

        print(f"\n=== T Computation Results ===")
        print(f"Total sum: {result['Tsum']:.6f}")
        print(f"Row sums: {result['Trowsums']}")
        print(f"Column sums: {result['Tcolsums']}")
        print()

        csr_val = result["csr_val"]
        csr_colind = result["csr_colind"]
        csr_rowptr = result["csr_rowptr"]
        print(f"CSR data: {csr_val}")
        print(f"CSR col index: {csr_colind}")
        print(f"CSR row pointer: {csr_rowptr}")
        print()

        print(f"=== Verification ===")
        Tsum_diff = np.abs(Tsum - result["Tsum"])
        if Tsum_diff < 1e-8:
            print(f"✓ Tsum matches, difference = {Tsum_diff}")
        else:
            print(f"✗ Tsum does not match, difference = {Tsum_diff}")
        
        Trowsums_diff = np.linalg.norm(Trowsums - result["Trowsums"])
        if Trowsums_diff < 1e-8:
            print(f"✓ Trowsums matches, difference = {Trowsums_diff}")
        else:
            print(f"✗ Trowsums does not match, difference = {Trowsums_diff}")

        Tcolsums_diff = np.linalg.norm(Tcolsums - result["Tcolsums"])
        if Tcolsums_diff < 1e-8:
            print(f"✓ Tcolsums matches, difference = {Tcolsums_diff}")
        else:
            print(f"✗ Tcolsums does not match, difference = {Tcolsums_diff}")

        val_diff = np.linalg.norm(Tsp.data - csr_val)
        if val_diff < 1e-8:
            print(f"✓ CSR data matches, difference = {val_diff}")
        else:
            print(f"✗ CSR data does not match, difference = {val_diff}")

        colind_diff = np.linalg.norm(Tsp.indices - csr_colind)
        if colind_diff < 1e-8:
            print(f"✓ CSR col index matches, difference = {colind_diff}")
        else:
            print(f"✗ CSR col index does not match, difference = {colind_diff}")

        rowptr_diff = np.linalg.norm(Tsp.indptr - csr_rowptr)
        if rowptr_diff < 1e-8:
            print(f"✓ CSR row pointer, difference = {rowptr_diff}")
        else:
            print(f"✗ CSR row pointer does not match, difference = {rowptr_diff}")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    np.random.seed(123)
    test_csr_conversion()
