#include "cuda_runtime.h"
#include "math_functions.h"
#include "device_launch_parameters.h"
//#include "math.h"
#include <stdio.h>
#include <tchar.h>
#include <vector>
#include <map>
#include <iostream>
//#include <cublas_v2.h>
#include "helper_cuda.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "Common.cuh"
#include "Gaussian2d2.cuh"

using namespace std;
using namespace thrust;

namespace Leicester
{
	namespace CudaLib
	{
		//__global__ void gpuErrchk(cudaError_t ans)
		//{
		//	gpuAssert << <1, 1 >> >((ans), __FILE__, __LINE__);
		//}


		__global__ void gpuAssert(cudaError_t code, const char *file, int line)
		{
			if (code != cudaSuccess)
			{
				printf("GPUassert: %i %s %d\n", code, file, line);
			}
		}

		//template <> void FAI_CUDA<32>(double(&FAI)[32], double a, double *TP, double CN, double c, dim3 dimTP);

		__global__ void
			Gaussian2d2_CUDA(double *D, double *Dt, double *Dx, double *Dxx, double *TP, dim3 dimTP, double *CN, dim3 dimCN, double *A, dim3 dimA, double *C, dim3 dimC)
		{
			int i = blockDim.y * blockIdx.y + threadIdx.y;
			int j = blockDim.x * blockIdx.x + threadIdx.x;
			//printf("start mqd22_CUDA i=%i j =%i \r\n", i, j);

			if (i < dimCN.y)
				//if (i == 0)
			{
				int test = 0;
				//printf("mqd22_CUDA i=%i j =%i dimCN.y=%i\r\n", i, j, dimCN.y);
				dim3 threads(32, 32);
				//dim3 threads(8, 8);
				//dim3 grid(dimTP.x / threads.x, dimTP.y / threads.y);
				int gridRows = 1 + (dimCN.y / threads.y);
				//printf("rows=%i\r\n", gridRows);
				dim3 grid(1, (int)gridRows);
				/*printf("TP size=%i\r\n", dimTP.x * dimTP.y);
				printMatrix_CUDA << <1, 1 >> >(TP, dimTP);*/
				/*
				CALCULATE FAI
				*/
				dim3 dimFAI(1, dimTP.y);
				//double *FAI1 = new double[dimTP.y, 1];
				double *FAI1 = (double*)malloc(sizeof(double) * dimFAI.y * dimFAI.x);
				//matrixFill_CUDA << <grid, threads >> > (FAI1, dim3(1, dimTP.y), 1.0);
				//gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
				//gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
				//__syncthreads();
				//printf("FAI1 size=%i", sizeof(FAI1));
				//printMatrix_CUDA << <1, 1 >> >(FAI1, dim3(1, dimTP.y));

				//printf("A[0] =%f CN[i] =%f C[0] =%f dimTPx=%i dimTPy=%i\r\n", A[0], CN[i], C[0], dimTP.x, dimTP.y);
				FAI_CUDA << <1, threads >> > (FAI1, A[0], TP, 0, CN[i], C[0], dimFAI);
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				__syncthreads();
				//if (i == test)
				//{
				//	printf("FAI1 size=%i", sizeof(FAI1));
				//	printMatrix_CUDA << <1, 1 >> > (FAI1, dim3(1, dimTP.y));
				//}
				__syncthreads();

				double *FAI2 = (double*)malloc(sizeof(double) * dimTP.y * 1);
				//matrixFill_CUDA << <1, threads >> > (FAI2, dim3(1, dimTP.y), 1.0);
				//gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
				//gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());
				__syncthreads();
				//printf("FAI2 size=%i", sizeof(FAI2));
				//printMatrix_CUDA << <1, 1 >> >(FAI2, dim3(1, dimTP.y));

				FAI_CUDA << <1, threads >> > (FAI2, A[1], TP, 1, CN[i + dimCN.y], C[1], dimFAI);
				__syncthreads();
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				//if (i == test)
				//{
				//	printf("FAI2 size=%i", sizeof(FAI2));
				//	printMatrix_CUDA << <1, 1 >> >(FAI2, dim3(1, dimTP.y));
				//}
				__syncthreads();
				/*
				CALCULATE D
				*/
				double *DColJ = (double*)malloc(sizeof(double) * dimTP.y * 1);
				ElementWiseMultiply_CUDA << <threads, grid >> > (DColJ, FAI1, FAI2, dimFAI);
				//__syncthreads();
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				//if (i == test)
				//{
				//	printf("DColJ size=%f", sizeof(DColJ));
				//	printMatrix_CUDA << <1, 1 >> > (DColJ, dim3(1, dimTP.y));
				//}
				/*
				CALCULATE Dt
				*/
				double * tpCol0 = (double*)malloc(sizeof(double) * dimTP.y * 1);
				GetColumn << <threads, grid >> > (tpCol0, 0, TP, dimFAI);

				__syncthreads();
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				__syncthreads();
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				__syncthreads();
				//if (i == test)
				//{
				//	printf("tpCol0 size=%f", sizeof(tpCol0));
				//	printMatrix_CUDA << <1, 1 >> > (tpCol0, dim3(1, dimTP.y));
				//	__syncthreads();
				//}
				double *a3 = (double*)malloc(sizeof(double) * dimTP.y * 1);
				dim3 dimA3(1, dimTP.y);
				dim3 dimTpCol0(1, dimTP.y);
				double a = 0;
				a = A[0];
				//printMatrix_CUDA << <1, 1 >> >(A, dimA);
				__syncthreads();
				double c = C[0];
				//printMatrix_CUDA << <1, 1 >> >(C, dimC);
				__syncthreads();
				double scalar1 = -2 * ((a / c) * (a / c));
				double cn = CN[i];
				//printf("scalar1=%f cn=%f a=%f c=%f\r\n", scalar1, cn, a, c);
				//double * tpCol1 = tpCol0;
				//cudaMemcpy(tpCol1, tpCol0, sizeof(tpCol0), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
				ScalarVectorDifference_CUDA << <1, threads >> > (a3, scalar1, tpCol0, cn, dimTpCol0);
				__syncthreads();
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				__syncthreads();
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				__syncthreads();
				//if (i == test)
				//{
				//	printf("a3 size=%f\r\n", sizeof(a3));
				//	printMatrix_CUDA << <1, 1 >> > (a3, dimTpCol0);
				//}
				__syncthreads();

				//__syncthreads();
				double *b3 = (double*)malloc(sizeof(double) * dimTP.y * 1);
				ElementWiseMultiply_CUDA << <1, threads >> > (b3, a3, FAI1, dimTpCol0);
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				__syncthreads();
				//if (i == test)
				//{
				//	printMatrix_CUDA << <1, 1 >> > (b3, dimTpCol0);
				//}

				double *DtColJ = (double*)malloc(sizeof(double) * dimTP.y * 1);
				ElementWiseMultiply_CUDA << <1, threads >> > (DtColJ, b3, FAI2, dimTpCol0);
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				free(a3);
				free(b3);
				//if (i == test)
				//{
				//	printf("DtColJ size=%f", sizeof(DtColJ));
				//	printMatrix_CUDA << <1, 1 >> > (DtColJ, dim3(1, dimTP.y));
				//	//__syncthreads();
				//}
				/*
				CALCULATE Dx
				*/
				double *a4 = (double*)malloc(sizeof(double) * dimTP.y * 1);
				double * tpCol1 = (double*)malloc(sizeof(double) * dimTP.y * 1);
				double * tpCol1a = (double*)malloc(sizeof(double) * dimTP.y * 1);
				GetColumn << <1, threads >> > (tpCol1, 1, TP, dimTP);
				GetColumn << <1, threads >> > (tpCol1a, 1, TP, dimTP);
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				__syncthreads();
				//if (i == test)
				//{
				//printf("tpCol1 size=%f", sizeof(tpCol1));
				//printMatrix_CUDA << <1, 1 >> >(tpCol1, dim3(1, dimTP.y));
				//}

				dim3 dimTpCol1(1, dimTP.y);
				//printf("A[1]=%f, C[1]=%f, CN[1]=%f", A[1], C[1], CN[i + (1 * dimCN.y)]);
				ScalarVectorDifference_CUDA << <1, threads >> > (a4, -2 * ((A[1] / C[1]) * (A[1] / C[1])), tpCol1, CN[i + (1 * dimCN.y)], dimTpCol1);
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				//if (i == test)
				//{
				//	printMatrix_CUDA << <1, 1 >> > (a4, dim3(1, dimTP.y));
				//}
				__syncthreads();

				double *b4 = (double*)malloc(sizeof(double) * dimTP.y * 1);
				ElementWiseMultiply_CUDA << <1, threads >> > (b4, a4, FAI1, dimTpCol1);
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				//if (i == test)
				//{
				//	printMatrix_CUDA << <1, 1 >> >(b4, dim3(1, dimTP.y));
				//}
				__syncthreads();

				double *DxColJ = (double*)malloc(sizeof(double) * dimTP.y * 1);
				double *c4 = (double*)malloc(sizeof(double) * dimTP.y * 1);
				ElementWiseMultiply_CUDA << <1, threads >> > (c4, b4, FAI2, dimTpCol1);
				//printMatrix_CUDA << <1, 1 >> >(c4, dim3(1, dimTP.y));
				ElementWiseMultiply_CUDA << <1, threads >> > (DxColJ, tpCol1, c4, dimTpCol1);

				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				free(a4);
				free(b4);
				free(c4);
				//if (i == test)
				//{
				//	printf("DxColJ size=%f", sizeof(DxColJ));
				//	printMatrix_CUDA << <1, 1 >> > (DxColJ, dim3(1, dimTP.y));
				//	//__syncthreads();
				//}

				/*
				CALCULATE Dxx
				*/
				double sA = A[1] * A[1];
				double qA = A[1] * A[1] * A[1] * A[1];
				double sC = C[1] * C[1];
				double qC = C[1] * C[1] * C[1] * C[1];

				double *dTpCn = (double*)malloc(sizeof(double) * dimTP.y * 1);
				double cn1 = CN[i + dimCN.y];
				//printf("cn=%f", cn1);
				MatrixSubtractScalar_CUDA << <1, threads >> > (dTpCn, tpCol1, cn1, dimTpCol1);
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				//printMatrix_CUDA << <1, 1 >> >(tpCol1, dim3(1, dimTP.y));
				//printMatrix_CUDA << <1, 1 >> >(dTpCn, dim3(1, dimTP.y));
				//__syncthreads();

				double *a5 = (double*)malloc(sizeof(double) * dimTP.y * 1);
				double *sdTpCn = (double*)malloc(sizeof(double) * dimTP.y * 1);
				ElementWiseMultiply_CUDA << <1, threads >> > (sdTpCn, dTpCn, dTpCn, dimTpCol1);
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				//__syncthreads();
				MatrixScalarMultiply_CUDA << <1, threads >> > (a5, sdTpCn, 4 * qA / qC, dimTpCol1);
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				//__syncthreads();
				//if (i == test)
				//{
				//	printf("a5 size=%f", sizeof(a5));
				//	printMatrix_CUDA << <1, 1 >> > (a5, dim3(1, dimTpCol1.y));
				//	//__syncthreads();
				//}

				double *b5 = (double*)malloc(sizeof(double) * dimTP.y * 1);
				MatrixAddScalar_CUDA << <1, threads >> > (b5, a5, -2 * sA / sC, dimTpCol1);
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				//__syncthreads();
				//if (i == test)
				//{
				//	printf("b5 size=%f", sizeof(b5));
				//	printMatrix_CUDA << <1, 1 >> > (b5, dim3(1, dimTpCol1.y));
				//	//__syncthreads();
				//}
				double *c5 = (double*)malloc(sizeof(double) * dimTP.y * 1);
				double *sFAI = (double*)malloc(sizeof(double) * dimTP.y * 1);
				ElementWiseMultiply_CUDA << <1, threads >> > (sFAI, FAI2, FAI1, dimTpCol1);
				ElementWiseMultiply_CUDA << <1, threads >> > (c5, b5, sFAI, dimTpCol1);
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				//__syncthreads();
				//if (i == test)
				//{
				//	printf("c5 size=%f", sizeof(a5));
				//	printMatrix_CUDA << <1, 1 >> > (c5, dim3(1, dimTpCol1.y));
				//	//__syncthreads();
				//}

				double *DxxColJ = (double*)malloc(sizeof(double) * dimTP.y * 1);
				/*double *sTPCol1 = (double*)malloc(sizeof(double) * ( dimTP.y +1) * 1);
				sTPCol1[0] = dimTP.y;*/
				double *sTPCol1 = (double*)malloc(sizeof(double) * (dimTP.y) * 1);

				ElementWiseMultiply_CUDA << <1, threads >> > (sTPCol1, tpCol1, tpCol1a, dimTpCol1);
				//if (i == test)
				//{
				//	printf("sTPCol1 size=%f", sizeof(sTPCol1));
				//	printMatrix_CUDA << <1, 1 >> > (sTPCol1, dim3(1, dimTpCol1.y));
				//	//__syncthreads();
				//}
				ElementWiseMultiply_CUDA << <1, threads >> > (DxxColJ, sTPCol1, c5, dimTpCol1);
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				free(a5);
				free(b5);
				free(c5);
				//if (i == test)
				//{
				//	printf("DxxColJ size=%f", sizeof(DxxColJ));
				//	printMatrix_CUDA << <1, 1 >> > (DxxColJ, dim3(1, dimTP.y));
				//	//__syncthreads();
				//}
				dim3 dimResult(dimCN.x, dimTP.y);

				int size = dimTP.y * dimTP.y;
				int threadsn = threads.x * threads.y;
				int grids = (size / (threadsn)) + 1;
				//printf("Set Column=%i size=%i threads=%i grids=%i\r\n", i, size, threadsn, grids);
				SetColumn << <grids, threads >> > (D, i, DColJ, dimResult);

				__syncthreads();
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				if (i == test)
				{
					printf("D size=%f\r\n", sizeof(D));
					dumpMatrix_CUDA << <1, 1 >> > (D, dim3(dimTP.y, dimTP.y));
					//__syncthreads();
				}
				__syncthreads();
				SetColumn << <grids, threads >> > (Dt, i, DtColJ, dimResult);
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				__syncthreads();
				SetColumn << <grids, threads >> > (Dx, i, DxColJ, dimResult);
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);
				__syncthreads();
				SetColumn << <grids, threads >> > (Dxx, i, DxxColJ, dimResult);
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);

				//printMatrix_CUDA<<<1,1>>>(D, dimResult);
				free(FAI1);
				free(FAI2);
				free(DColJ);

				free(DtColJ);

				free(DxColJ);
				free(dTpCn);
				free(sdTpCn);

				free(sFAI);
				free(DxxColJ);
				free(sTPCol1);
			}
			__syncthreads();
		}


	}
}

