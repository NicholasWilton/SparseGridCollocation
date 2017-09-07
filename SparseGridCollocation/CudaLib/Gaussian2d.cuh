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

__global__ void test();

__global__ void
Gaussian2d_CUDA(double *D, double *Dt, double *Dx, double *Dxx, double *TP, int TPx, int TPy, double *CN, int CNx, int CNy, double *A, int Ax, int Ay, double *C, int Cx, int Cy);