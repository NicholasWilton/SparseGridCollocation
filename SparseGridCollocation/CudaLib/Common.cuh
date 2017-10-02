#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include <stdio.h>
#include <tchar.h>
#include <vector>
#include <map>
#include <iostream>
//#include <cublas_v2.h>
#include "./cuda_include/helper_cuda.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>



using namespace std;
using namespace thrust;

namespace Leicester
{
	namespace CudaLib
	{
		extern __global__ void
			matrixMul_CUDA(double *C, double *A, double *B, int wA, int wB);

		int matrixMultiply(int block_size, double * h_A, dim3 &dimsA, double * h_B, dim3 &dimsB);

		extern __global__ void
			dumpMatrix_CUDA(double *matrix, dim3 dimMatrix);

		extern __global__ void
			printMatrix_CUDA(double *matrix, dim3 dimMatrix);

		extern void
			printMatrix(const double *matrix, dim3 dimMatrix);

		extern void
			printMatrixFormatted(const double *matrix, dim3 dimMatrix);

		extern __global__ void
			matrixFill_CUDA(double *matrix, dim3 dimMatrix, double fill);

		extern __global__ void
			FAI_CUDA(double *FAI, double a, double *TP, int tpCol, double CN, double c, dim3 dimTP);

		// D = a * (B - c)
		extern __global__ void
			ScalarVectorDifference_CUDA(double *D, double a, double *B, double c, dim3 dimB);

		extern __global__ void
			ElementWiseMultiply_CUDA(double *C, double *A, double *B, int rows, int cols);

		extern __global__ void
			ElementWiseMultiply_CUDA(double *C, double *A, double *B, dim3 dimC);

		extern __global__ void
			ElementWiseMultiply_CUDA2(double *C, double *A, double *B, dim3 dimC);
		extern __global__ void
			ElementWiseMultiply_CUDA3(double *C, double *A, double *B, dim3 dimC);
		extern __global__ void
			ElementWiseMultiply_CUDA4(double *C, double *A, double *B, dim3 dimC);
		extern __global__ void
			ElementWiseMultiply_CUDA5(double *C, double *A, double *B, dim3 dimC);

		//extern __global__ void
		//ElementWiseMultiply_CUDA(double *D, double *A, double *B, double *C, dim3 dimD);
		
		extern __global__ void
			ElementWiseMultiplyThree_CUDA(double *D, double *A, double *B, double *C, dim3 dimD);

		extern __global__ void
			MatrixScalarMultiply_CUDA(double *C, double *A, double b, dim3 dimA);

		extern __global__ void
			MatrixSubtractScalar_CUDA(double *C, double *A, double b, dim3 dimA);

		extern __global__ void
			MatrixAddScalar_CUDA(double *C, double *A, double b, dim3 dimA);

		extern __global__ void
			GetColumn(double * result, int col, double *A, dim3 dimA);

		extern __global__ void
			GetRow(double * result, int row, double *A, dim3 dimA);

		extern __global__ void
			SetColumn(double * matrix, int col, double *vector, dim3 dimMatrix);

		extern __global__ void
			SetColumnLogged(double * matrix, int col, double *vector, dim3 dimMatrix);

		extern __global__ void
			mqd2_CUDA(double *D, double *Dt, double *Dx, double *Dxx, double *TP, dim3 dimTP, double *CN, dim3 dimCN, double *A, dim3 dimA, double *C, dim3 dimC);

		extern __global__ void
			mqd2_CUDA(device_vector<double*> result, double *TP, int TPx, int TPy, double *CN, int CNx, int CNy, double *A, int Ax, int Ay, double *C, int Cx, int Cy);
	}
}