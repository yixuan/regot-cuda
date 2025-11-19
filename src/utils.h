#include <iostream>
#include <cmath>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

double compute_l2_distance_cuda(double* d_vec1, double* d_vec2, int size);
double compute_l2_norm_cuda(double* d_vec, int size);
