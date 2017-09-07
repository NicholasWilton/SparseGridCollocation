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
	
	if (j < dimCN.x)
	{

		printf("mqd22_CUDA i=%i j =%i dimCN.y=%i\r\n", i, j, dimCN.y);
		//__shared__ double D[dimTP.x][dimCN.y];
		//__shared__ double Dt[dimTP.x][dimCN.y];
		//__shared__ double Dx[dimTP.x][dimCN.y];
		//__shared__ double Dxx[dimTP.x][dimCN.y];
		dim3 threads(32, 32);
		//dim3 grid(dimTP.x / threads.x, dimTP.y / threads.y);
		dim3 grid(1, 1);

		/*
		CALCULATE FAI
		*/
		dim3 dimFAI(dimCN.x, dimTP.y);
		//double *FAI1 = new double[dimTP.y, 1];
		double *FAI1 = (double*)malloc(sizeof(double) * dimFAI.y * dimFAI.x);
		//matrixFill_CUDA << <grid, threads >> > (FAI1, dim3(1, dimTP.y), 1.0);
		//gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		//gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();
		//printf("FAI1 size=%i", sizeof(FAI1));
		//printMatrix_CUDA << <1, 1 >> >(FAI1, dim3(1, dimTP.y));

		//printf("A[0] =%f CN[i] =%f C[0] =%f dimTPx=%i dimTPy=%i\r\n", A[0], CN[i], C[0], dimTP.x, dimTP.y);
		FAI_CUDA<< <threads, grid >> >(FAI1, A[0], TP, 0, CN[i], C[0], dimTP);
		gpuErrchk<<<1,1>>>(cudaPeekAtLastError());
		gpuErrchk<<<1,1>>>(cudaDeviceSynchronize());
		__syncthreads();
		printf("FAI1 size=%i", sizeof(FAI1));
		printMatrix_CUDA<<<1, 1>>>(FAI1, dim3(1, dimTP.y));

		__syncthreads();

		double *FAI2 = (double*)malloc(sizeof(double) * dimTP.y * 1);
		//matrixFill_CUDA << <grid, threads >> > (FAI2, dim3(1, dimTP.y), 1.0);
		//gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		//gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();
		//printf("FAI2 size=%i", sizeof(FAI2));
		//printMatrix_CUDA << <1, 1 >> >(FAI2, dim3(1, dimTP.y));

		FAI_CUDA<< <threads, grid >> >(FAI2, A[1], TP, 1, CN[j + dimCN.x], C[1], dimTP);
		__syncthreads();
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		printf("FAI2 size=%i", sizeof(FAI2));
		printMatrix_CUDA << <1, 1 >> >(FAI2, dim3(1, dimTP.y));
		__syncthreads();

		/*
		CALCULATE D
		*/
		double *DColJ = (double*)malloc(sizeof(double) * dimTP.y * 1);
		ElementWiseMultiply_CUDA << <threads, grid >> >(DColJ, FAI1, FAI2, dimFAI);
		//__syncthreads();
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		printf("DColJ size=%f", sizeof(DColJ));
		printMatrix_CUDA << <1, 1 >> >(DColJ, dim3(1, dimTP.y));
		
		/*
		CALCULATE Dt
		*/
		double * tpCol0 = (double*)malloc(sizeof(double) * dimTP.y * 1);
		GetColumn << <threads, grid >> >(tpCol0, 0, TP, dimTP);

		__syncthreads();
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		__syncthreads();
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		__syncthreads();
		printf("tpCol0 size=%f", sizeof(tpCol0));
		printMatrix_CUDA << <1, 1 >> >(tpCol0, dim3(1, dimTP.y));
		__syncthreads();
		double *a3 = (double*)malloc(sizeof(double) * dimTP.y * 1);
		dim3 dimA3(1, dimTP.y);
		dim3 dimTpCol0(1, dimTP.y);
		double a = 0;
		a = A[0];
		printMatrix_CUDA << <1, 1 >> >(A, dimA);
		__syncthreads();
		double c = C[0];
		printMatrix_CUDA << <1, 1 >> >(C, dimC);
		__syncthreads();
		double scalar1 = -2 * ((a / c) * (a / c));
		double cn = CN[i];
		printf("scalar1=%f cn=%f a=%f c=%f\r\n", scalar1, cn, a, c);
		//double * tpCol1 = tpCol0;
		//cudaMemcpy(tpCol1, tpCol0, sizeof(tpCol0), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
		ScalarVectorDifference_CUDA << <threads, grid >> >(a3, scalar1, tpCol0, cn, dimTpCol0);
		__syncthreads();
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		__syncthreads();
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		__syncthreads();
		printMatrix_CUDA << <1, 1 >> >(a3,dimA3);
		__syncthreads();

		//__syncthreads();
		double *b3 = (double*)malloc(sizeof(double) * dimTP.y * 1);
		ElementWiseMultiply_CUDA << <threads, grid >> >(b3, a3, FAI1, dimTP);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();


		double *DtColJ = (double*)malloc(sizeof(double) * dimTP.y * 1);
		ElementWiseMultiply_CUDA << <threads, grid >> >(DtColJ, b3, FAI2, dimTP);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		printf("DtColJ size=%f", sizeof(DtColJ));
		printMatrix_CUDA << <1, 1 >> >(DtColJ, dim3(1, dimTP.y));
		//__syncthreads();
		
		/*
		CALCULATE Dx
		*/
		double *a4 = (double*)malloc(sizeof(double) * dimTP.y * 1);
		double * tpCol1 = (double*)malloc(sizeof(double) * dimTP.y * 1);
		GetColumn << <threads, grid >> >(tpCol1, 1, TP, dimTP);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();

		dim3 dimTpCol1(1, dimTP.y);
		ScalarVectorDifference_CUDA<< <threads, grid >> >(a4, -2 * ((A[1] / C[1]) * (A[1] / C[1])), tpCol1, CN[i + (1 * dimCN.x)], dimTpCol1);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//printMatrix_CUDA << <1, 1 >> >(a4, dim3(1, dimTP.y));
		//__syncthreads();

		double *b4 = (double*)malloc(sizeof(double) * dimTP.y * 1);
		ElementWiseMultiply_CUDA << <threads, grid >> >(b4, a4, FAI1, dimTP);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//printMatrix_CUDA << <1, 1 >> >(b4, dim3(1, dimTP.y));
		//__syncthreads();

		double *DxColJ = (double*)malloc(sizeof(double) * dimTP.y * 1);
		double *c4 = (double*)malloc(sizeof(double) * dimTP.y * 1);
		ElementWiseMultiply_CUDA<< <threads, grid >> >(c4, b4, FAI2, dimTP);
		ElementWiseMultiply_CUDA << <threads, grid >> >(DxColJ, tpCol1, c4, dimTP);

		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		printf("DxColJ size=%f", sizeof(DxColJ));
		printMatrix_CUDA << <1, 1 >> >(DxColJ, dim3(1, dimTP.y));
		//__syncthreads();


		/*
		CALCULATE Dxx
		*/
		double sA = A[1] * A[1];
		double qA = A[1] * A[1] * A[1] * A[1];
		double sC = C[1] * C[1];
		double qC = C[1] * C[1] * C[1] * C[1];

		double *dTpCn = (double*)malloc(sizeof(double) * dimTP.y * 1);
		double cn1 = CN[j + dimCN.x];
		printf("cn=%f", cn1);
		MatrixSubtractScalar_CUDA << <threads, grid >> >(dTpCn, tpCol1, cn1, dimTpCol1);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		printMatrix_CUDA << <1, 1 >> >(tpCol1, dim3(1, dimTP.y));
		printMatrix_CUDA << <1, 1 >> >(dTpCn, dim3(1, dimTP.y));
		//__syncthreads();

		double *a5 = (double*)malloc(sizeof(double) * dimTP.y * 1);
		double *sdTpCn = (double*)malloc(sizeof(double) * dimTP.y * 1);
		ElementWiseMultiply_CUDA << <threads, grid >> >(sdTpCn, dTpCn, dTpCn, dimTpCol1);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();
		MatrixScalarMultiply_CUDA << <threads, grid >> >(a5, sdTpCn, 4 * qA / qC, dimTpCol1);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();

		double *b5 = (double*)malloc(sizeof(double) * dimTP.y * 1);
		MatrixAddScalar_CUDA << <threads, grid >> >(b5, a5, -2 * sA / sC, dimTpCol1);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();
		double *c5 = (double*)malloc(sizeof(double) * dimTP.y * 1);
		double *sFAI = (double*)malloc(sizeof(double) * dimTP.y * 1);
		ElementWiseMultiply_CUDA << <threads, grid >> >(sFAI, FAI2, FAI1, dimTpCol1);
		ElementWiseMultiply_CUDA << <threads, grid >> >(c5, b5, sFAI, dimTpCol1);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();


		double *DxxColJ = (double*)malloc(sizeof(double) * dimTP.y * 1);
		double *sTPCol1 = (double*)malloc(sizeof(double) * dimTP.y * 1);
		ElementWiseMultiply_CUDA << <threads, grid >> >(sTPCol1, tpCol1, tpCol1, dimTpCol1);
		ElementWiseMultiply_CUDA << <threads, grid >> >(DxxColJ, sTPCol1, c5, dimTpCol1);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		printf("DxxColJ size=%f", sizeof(DxxColJ));
		printMatrix_CUDA << <1, 1 >> >(DxxColJ, dim3(1, dimTP.y));
		//__syncthreads();

		dim3 dimResult(dimCN.x, dimTP.y);
		SetColumn << <threads, grid >> >(D, j, DColJ, dimResult);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();
		SetColumn << <threads, grid >> >(Dt, j, DtColJ, dimResult);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();
		SetColumn << <threads, grid >> >(Dx, j, DxColJ, dimResult);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		//__syncthreads();
		SetColumn << <threads, grid >> >(Dxx, j, DxxColJ, dimResult);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
		
		printMatrix_CUDA<<<1,1>>>(D, dimResult);
		delete[] FAI1;
	}
	__syncthreads();
}
