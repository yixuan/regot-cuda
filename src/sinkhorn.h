#pragma once

#include <iostream>
#include <cmath>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Helper function to compute optimal beta given alpha
void compute_optimal_beta(
    const double* d_M, const double* d_alpha, const double* d_logb,
    double* d_beta, double reg, int n, int m
);

// Helper function to compute optimal alpha given beta
void compute_optimal_alpha(
    const double* d_M, const double* d_beta, const double* d_loga,
    double* d_alpha, double reg, int n, int m
);

// Helper function to compute final transport plan P
void compute_transport_plan(
    const double* d_M, const double* d_alpha, const double* d_beta,
    double* d_P, double reg, int n, int m
);
