#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include <stdio.h>
#include <tchar.h>
#include <vector>
#include <map>
#include <iostream>
#include <cublas_v2.h>
#include "./cuda_include/helper_cuda.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;
using namespace thrust;

extern __global__ void gpuAssert(cudaError_t code, const char *file, int line);
//extern __global__ void gpuErrchk(cudaError_t ans);
extern __global__ void
Gaussian2d2_CUDA(double *D, double *Dt, double *Dx, double *Dxx, double *TP, dim3 dimTP, double *CN, dim3 dimCN, double *A, dim3 dimA, double *C, dim3 dimC);
