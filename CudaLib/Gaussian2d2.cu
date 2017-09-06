#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include <stdio.h>
#include <tchar.h>
#include <vector>
#include <map>
#include <iostream>
#include <cublas_v2.h>
#include "helper_cuda.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "Common.cuh"
#include "Gaussian2d2.cuh"

using namespace std;
using namespace thrust;

__global__ void gpuErrchk(cudaError_t ans)
{ 
	gpuAssert<<<1,1>>>((ans), __FILE__, __LINE__, false); 
}
__global__ void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess)
	{
		printf("GPUassert: %i %s %d\n", code, file, line);
		//if (abort) exit(code);
	}
}

//template <> void FAI_CUDA<32>(double(&FAI)[32], double a, double *TP, double CN, double c, dim3 dimTP);

__global__ void
Gaussian2d2_CUDA(double *D, double *Dt, double *Dx, double *Dxx, double *TP, dim3 dimTP, double *CN, dim3 dimCN, double *A, dim3 dimA, double *C, dim3 dimC)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	printf("start mqd22_CUDA i=%i j =%i \r\n", i, j);
	
	if (j < dimCN.y)
	{

		printf("mqd22_CUDA i=%i j =%i dimCN.y=%i\r\n", i, j, dimCN.y);
		//__shared__ double D[dimTP.x][dimCN.y];
		//__shared__ double Dt[dimTP.x][dimCN.y];
		//__shared__ double Dx[dimTP.x][dimCN.y];
		//__shared__ double Dxx[dimTP.x][dimCN.y];
		dim3 threads(32, 32);
		//dim3 grid(dimTP.x / threads.x, dimTP.y / threads.y);
		dim3 grid(1, 1);

		double *FAI1 = new double[dimTP.y, 1];
		matrixFill_CUDA << <grid, threads >> > (FAI1, dim3(1, dimTP.y), 1.0);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		__syncthreads();
		printf("FAI1 size=%i", sizeof(FAI1));
		printMatrix_CUDA << <1, 1 >> >(FAI1, dim3(1, dimTP.y));

		//printf("A[0] =%f CN[i] =%f C[0] =%f dimTPx=%i dimTPy=%i\r\n", A[0], CN[i], C[0], dimTP.x, dimTP.y);
		FAI_CUDA<< <threads, grid >> >(FAI1, A[0], TP, 0, CN[i], C[0], dimTP);
		gpuErrchk<<<1,1>>>(cudaPeekAtLastError());
		gpuErrchk<<<1,1>>>(cudaDeviceSynchronize());
		__syncthreads();
		printf("FAI1 size=%i", sizeof(FAI1));
		printMatrix_CUDA<<<1, 1>>>(FAI1, dim3(1, dimTP.y));

		//__syncthreads();

		double *FAI2 = new double[dimTP.y, 1];
		FAI_CUDA<< <threads, grid >> >(FAI2, A[1], TP, 1, CN[j + dimCN.x], C[1], dimTP);
		//__syncthreads();
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		printf("FAI2 size=%i", sizeof(FAI2));
		printMatrix_CUDA << <1, 1 >> >(FAI2, dim3(1, dimTP.y));

		double *DColJ = new double[dimTP.y, 1];
		ElementWiseMultiply_CUDA << <threads, grid >> >(DColJ, FAI1, FAI2, dimTP, dimTP);
		//__syncthreads();
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		printf("DColJ size=%f", sizeof(DColJ));
		printMatrix_CUDA << <1, 1 >> >(DColJ, dim3(1, dimTP.y));

		
		double * tpCol0 = new double[dimTP.y, 1];
		GetColumn << <threads, grid >> >(tpCol0, 0, TP, dimTP);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		printf("tpCol0 size=%f", sizeof(tpCol0));
		printMatrix_CUDA << <1, 1 >> >(tpCol0, dim3(1, dimTP.y));

		double *a3 = new double[dimTP.y, 1];
		dim3 dimTpCol0(0, dimTP.y);
		ScalarVectorDifference_CUDA << <threads, grid >> >(a3, -2 * ((A[0] / C[0]) * (A[0] / C[0])), tpCol0, CN[i], dimTpCol0);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());

		//__syncthreads();
		double *b3 = new double[dimTP.y, 1];
		ElementWiseMultiply_CUDA << <threads, grid >> >(b3, a3, FAI1, dimTP, dimTP);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();

		double *DtColJ = new double[dimTP.y, 1];
		ElementWiseMultiply_CUDA << <threads, grid >> >(DtColJ, b3, FAI2, dimTP, dimTP);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		printf("DtColJ size=%f", sizeof(DtColJ));
		printMatrix_CUDA << <1, 1 >> >(DtColJ, dim3(1, dimTP.y));
		//__syncthreads();

		double *a4 = new double[dimTP.y, 1];
		double * tpCol1 = new double[dimTP.y, 1];
		GetColumn << <threads, grid >> >(tpCol1, 1, TP, dimTP);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();

		dim3 dimTpCol1(0, dimTP.y);
		ScalarVectorDifference_CUDA<< <threads, grid >> >(a4, -2 * ((A[1] / C[1]) * (A[1] / C[1])), tpCol1, CN[i + (1 * dimCN.x)], dimTpCol1);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();

		double *b4 = new double[dimTP.y, 1];
		ElementWiseMultiply_CUDA << <threads, grid >> >(b4, a4, FAI1, dimTP, dimTP);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();

		double *DxColJ = new double[dimTP.y, 1];
		ElementWiseMultiply_CUDA<< <threads, grid >> >(DxColJ, b3, FAI2, dimTP, dimTP);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		printf("DxColJ size=%f", sizeof(DxColJ));
		printMatrix_CUDA << <1, 1 >> >(DxColJ, dim3(1, dimTP.y));
		//__syncthreads();

		double sA = A[1] * A[1];
		double qA = A[1] * A[1] * A[1] * A[1];
		double sC = C[1] * C[1];
		double qC = C[1] * C[1] * C[1] * C[1];

		double *dTpCn = new double[dimTP.y, 1];
		MatrixSubtractScalar_CUDA << <threads, grid >> >(dTpCn, tpCol1, CN[i + (1 * dimCN.x)], dimTpCol1);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();

		double *a5 = new double[dimTP.y, 1];
		double *sdTpCn = new double[dimTP.y, 1];
		ElementWiseMultiply_CUDA << <threads, grid >> >(sdTpCn, dTpCn, dTpCn, dimTpCol1, dimTpCol1);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();
		MatrixScalarMultiply_CUDA << <threads, grid >> >(a5, sdTpCn, 4 * qA / qC, dimTpCol1);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();

		double *b5 = new double[dimTP.y, 1];
		MatrixAddScalar_CUDA << <threads, grid >> >(b5, a5, -2 * sA / sC, dimTpCol1);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();
		double *c5 = new double[dimTP.y, 1];
		ElementWiseMultiply_CUDA << <threads, grid >> >(c5, b5, FAI2, FAI1, dimTpCol1);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();
		double *DxxColJ = new double[dimTP.y, 1];
		ElementWiseMultiply_CUDA << <threads, grid >> >(DxxColJ, tpCol1, tpCol1, c5, dimTpCol1);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		printf("DxxColJ size=%f", sizeof(DxxColJ));
		printMatrix_CUDA << <1, 1 >> >(DxxColJ, dim3(1, dimTP.y));
		//__syncthreads();

		SetColumn << <threads, grid >> >(D, i, DColJ, dim3(dimCN.y, dimTP.x));
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();
		SetColumn << <threads, grid >> >(Dt, i, DtColJ, dim3(dimCN.y, dimTP.x));
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();
		SetColumn << <threads, grid >> >(Dx, i, DxColJ, dim3(dimCN.y, dimTP.x));
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();
		SetColumn << <threads, grid >> >(Dxx, i, DxxColJ, dim3(dimCN.y, dimTP.x));
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		

		delete[] FAI1;
	}
	__syncthreads();
}
