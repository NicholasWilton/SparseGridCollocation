#pragma once
#include "C:\Users\User\Source\Repos\SparseGridCollocation\SparseGridCollocation\include\eigen-eigen-67e894c6cd8f\Eigen\StdVector"
#include "Common.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>
#include ".\cuda_include\helper_string.h"  // helper for shared functions common to CUDA Samples


#include <cublas_v2.h>
#include <cusolverDn.h>
// CUDA and CUBLAS functions
//#include ".\cuda_include\helper_functions.h"
#include ".\cuda_include\helper_cuda.h"
#include "Gaussian2d.cuh"
#include "Gaussian2d2.cuh"
#include <stdio.h>

#define API _declspec(dllexport)

using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace Eigen;


class API MethodOfLines
{
public:
	static int MoLiteration(double Tend, double Tdone, double dt, double *G, int GRows, int GCols, double *lamb, int lambRows, int lambCols, double inx2, double r, double K, MatrixXd A1, MatrixXd Aend, MatrixXd H);
};

class API CudaRBF
{
public:
	static vector<MatrixXd> Gaussian2D(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C);
};