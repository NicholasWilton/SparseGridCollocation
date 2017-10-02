#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include <stdio.h>
#include <tchar.h>
#include <vector>
#include <map>
#include <iostream>
//#include <cublas_v2.h>
#include "helper_cuda.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "Common.cuh"

using namespace std;
using namespace thrust; 





__global__ void
Leicester::CudaLib::matrixMul_CUDA(double *C, double *A, double *B, int wA, int wB)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * 32 * by;
	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;
	// Step size used to iterate through the sub-matrices of A
	int aStep = 32;
	// Index of the first sub-matrix of B processed by the block
	int bBegin = 32 * bx;
	// Step size used to iterate through the sub-matrices of B
	int bStep = 23 * wB;
	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	double Csub = 0;
	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
		a <= aEnd;
		a += aStep, b += bStep)
	{
		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ double As[32][32];
		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ double Bs[23][23];
		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];
		// Synchronize to make sure the matrices are loaded
		__syncthreads();
		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll
		for (int k = 0; k < 32; ++k)
		{
			Csub += As[ty][k] * Bs[k][tx];
		}
		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = wB * 32 * by + 32 * bx;
	C[c + wB * ty + tx] = Csub;
}

int Leicester::CudaLib::matrixMultiply(int block_size, double * h_A, dim3 &dimsA, double * h_B, dim3 &dimsB)
{
	// Allocate host memory for matrices A and B
	unsigned int size_A = dimsA.x * dimsA.y;
	unsigned int mem_size_A = sizeof(double) * size_A;

	unsigned int size_B = dimsB.x * dimsB.y;
	unsigned int mem_size_B = sizeof(double) * size_B;

	// Allocate device memory
	double *d_A, *d_B, *d_C;

	// Allocate host matrix C
	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(double);
	double *h_C = (double *)malloc(mem_size_C);

	if (h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

	cudaError_t error;

	error = cudaMalloc((void **)&d_A, mem_size_A);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_B, mem_size_B);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_C, mem_size_C);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// copy host memory to device
	error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_B,h_B) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Setup execution parameters
	dim3 threads(block_size, block_size);
	dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

	// Create and start timer
	printf("Computing result using CUDA Kernel...\n");
	//matrixMulCUDA<32> << < grid, threads >> > (d_C, d_A, d_B, dimsA.x, dimsB.x);
	cudaDeviceSynchronize();

	// Copy result from device to host
	error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

__global__ void
Leicester::CudaLib::dumpMatrix_CUDA(double *matrix, dim3 dimMatrix)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int mSize = sizeof(matrix);
	//char **output = new char*[dimMatrix.x * dimMatrix.y]();
	//char *buff = new char[dimMatrix.x * dimMatrix.y * 30];

	printf("dumping matrix data=");
	for (int x = 0; x < dimMatrix.x * dimMatrix.y; x++)
		printf("%f,", matrix[x]);
	printf("\r\n");

}

__global__ void
Leicester::CudaLib::printMatrix_CUDA(double *matrix, dim3 dimMatrix)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int mSize = sizeof(matrix);
	//char **output = new char*[dimMatrix.x * dimMatrix.y]();
	//char *buff = new char[dimMatrix.x * dimMatrix.y * 30];
	
	printf("printing matrix data=");
	for (int x = 0; x < dimMatrix.x * dimMatrix.y; x++)
		printf("%f,", matrix[x]);
	printf("\r\n");
	printf("rows=%i cols=%i\r\n", dimMatrix.y, dimMatrix.x);
	
	if (i == 0 & j ==0)
	{
		for (int y = 0; y < dimMatrix.y; y++)
		{
			for (int x = 0; x < dimMatrix.x; x++)
			{
				//int idx = (y * dimMatrix.x) + x;
				int idx = (x * dimMatrix.y) + y;
				//if ( mSize > idx)
				printf("indx=%i value=%16.10f\t", idx, matrix[idx]);
			}
			printf("\r\n");
		}
	}
}

void
Leicester::CudaLib::printMatrixFormatted(const double *matrix, dim3 dimMatrix)
{
	int mSize = sizeof(matrix);

	printf("printing matrix data=");
	for (int x = 0; x < dimMatrix.x * dimMatrix.y; x++)
		printf("%f,", matrix[x]);
	printf("\r\n");
	printf("rows=%i cols=%i\r\n", dimMatrix.y, dimMatrix.x);

	for (int y = 0; y < dimMatrix.y; y++)
	{
		for (int x = 0; x < dimMatrix.x; x++)
		{
			int idx = ( x * dimMatrix.y) + y;
			//if ( mSize > idx)
			printf("indx=%i value=%f\t", idx, matrix[idx]);
		}
		printf("\r\n");
	}
}

void
Leicester::CudaLib::printMatrix(const double *matrix, dim3 dimMatrix)
{
	int mSize = sizeof(matrix);

	printf("printing matrix data=");
	for (int x = 0; x < dimMatrix.x * dimMatrix.y; x++)
		printf("%f,", matrix[x]);
	printf("\r\n");
	printf("rows=%i cols=%i\r\n", dimMatrix.y, dimMatrix.x);

	for (int y = 0; y < dimMatrix.y; y++)
	{
		for (int x = 0; x < dimMatrix.x; x++)
		{
			int idx = (x * dimMatrix.y) + y;
			printf("%f ", matrix[idx]);
		}
		printf("\r\n");
	}
}

__global__ void
Leicester::CudaLib::matrixFill_CUDA(double *matrix, dim3 dimMatrix, double fill)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	if (i == 0 & j == 0)
	{
		for (int y = 0; y < dimMatrix.y; y++)
		{
			for (int x = 0; x < dimMatrix.x; x++)
			{
				int idx = (y * dimMatrix.x) + x;
				matrix[idx] = fill;
			}
		}
	}
}


__global__ void
Leicester::CudaLib::FAI_CUDA(double *FAI, double a, double *TP, int tpCol, double CN, double c, dim3 dimTP)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int sourceMin = dimTP.y * tpCol;
	int sourceMax = sourceMin + dimTP.y;
	//if (threadIdx.y == 0 & threadIdx.x == 0)
	//{
	//	printf("start FAI_CUDA blocky=%i, blockx=%i, blockDimx=%i, blockDimy=%i\r\n", blockIdx.y, blockIdx.x, blockDim.x, blockDim.y);
	//	printf("start FAI_CUDA sourcemini=%i, sourcemax=%i\r\n", sourceMin,sourceMax);
	//}
	//printf("start FAI_CUDA i=%i, j=%i\r\n", i,j);
	//printf("start FAI_CUDA Blocky=%i, Blockx=%i\r\n", blockIdx.y, blockIdx.x);
	
	int sourceIndex = i + (j * blockDim.y);
	int targetIndex = sourceIndex - (tpCol * dimTP.y);
	//printf("start FAI_CUDA source=%i, target=%i\r\n", sourceIndex, targetIndex);
	//printf("start FAI_CUDA limitsource=%i, limittarget=%i\r\n", sourceLength -1, dimTP.y -1);
	if ( (sourceIndex < sourceMax ) & (sourceIndex >= sourceMin) & (targetIndex >= 0) & (targetIndex <= dimTP.y -1) )
	{
		//printf("start FAI_CUDA i=%i, j=%i\r\n", i,j);
		//printf("start FAI_CUDA source=%i, target=%i\r\n", sourceIndex, targetIndex);
		//printf("start FAI_CUDA sourcerange=%i-%i, limittarget=%i\r\n", sourceMin, sourceMax, dimTP.y - 1);
		__syncthreads();
		
		//if (targetIndex == 1) 
			//printf("input FAI_CUDA TP[i]=%f CN=%f, a=%f, c=%f\r\n", TP[sourceIndex], CN, a, c);
		double a1 = a * (TP[sourceIndex] - CN);
		//if (targetIndex == 1) 
			//printf("FAI_CUDA a1=%f\r\n", a1);
		double b1 = -(a1 * a1) / (c * c);
		//if (targetIndex == 1) 
			//printf("FAI_CUDA b1=%f\r\n", b1);
		double e1 = expm1(b1) + 1;
		//if (targetIndex == 32) 
		//printf("FAI_CUDA e1=%f Source=%i, Target=%i, ylimit=%i, blocklength=%i, blockidxY=%i, threadidY=%i, blockidxX=%i, threadidX=%i\r\n", e1, sourceIndex, targetIndex, dimTP.y, blockDim.y, blockIdx.y, threadIdx.y, blockIdx.x, threadIdx.x);
		//if(targetIndex == 1)
			//printf("FAI_CUDA Source=%i Target=%i, e1=%f\r\n", sourceIndex, targetIndex, e1);
		FAI[targetIndex] = e1;
		
		//printf("FAI_CUDA FAI[%i]=%f\r\n",i, FAI[i]);
		
	}
	__syncthreads();
}

// D = a * (B - c)
__global__ void
Leicester::CudaLib::ScalarVectorDifference_CUDA(double *D, double a, double *B, double c, dim3 dimB)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	
	int sourceLength = dimB.x * dimB.y;
	int sourceIndex = i + (j * blockDim.y);
	int targetIndex = i + (j * blockDim.y);
	/*if (i == 0 & j == 0)
		printf("limitSource=%i limitTarget=%i\r\n", sourceLength - 1, dimB.y);*/
	
	//printf("sourceIndex=%i targetIndex=%i\r\n", sourceIndex, targetIndex);
	if ((sourceIndex <= sourceLength - 1))
	{
		//int idx = i + (j * dimB.y);
		//int idx = j + (i * j);
		double b = B[sourceIndex];
		D[targetIndex] = a * (b - c);
		//if(sourceIndex == 31) printf("i=%i, j=%i targetIndex=%i | %f = %f * (%f - %f)\r\n", i, j, targetIndex, D[targetIndex], a, b, c);

	}
}

__global__ void
Leicester::CudaLib::ElementWiseMultiply_CUDA(double *C, double *A, double *B, int rows, int cols)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	int sourceLength = cols * rows;
	int sourceIndex = i + (j * blockDim.y);
	int targetIndex = i + (j * blockDim.y);
	if ((sourceIndex <= sourceLength - 1) & (targetIndex < rows))
	{
		//if (i == 0 & j == 0)
		//{
		//	printf("ElementWiseMultiply_CUDA, matrix A:\r\n");
		//	printMatrix_CUDA << <1, 1 >> > (A, dimA);
		//	printf("ElementWiseMultiply_CUDA, matrix B:\r\n");
		//	printMatrix_CUDA << <1, 1 >> > (B, dimB);
		//}
		//int idx = i + (j * dimC.y);
		double a = A[sourceIndex];
		double b = B[sourceIndex];
		C[targetIndex] = a * b;
		//printf("i=%i, j=%i idx=%i | %i = %i * %i\r\n", i, j, idx, C[idx], a, b);
	}
}

__global__ void
Leicester::CudaLib::ElementWiseMultiply_CUDA(double *C, double *A, double *B, dim3 dimC)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	int sourceLength = dimC.x * dimC.y;
	int sourceIndex = i + (j * blockDim.y);
	int targetIndex = i + (j * blockDim.y);
	if ((sourceIndex <= sourceLength - 1) & (targetIndex < dimC.y))
	{
		//if (i == 0 & j == 0)
		//{
		//	printf("ElementWiseMultiply_CUDA, matrix A:\r\n");
		//	printMatrix_CUDA << <1, 1 >> > (A, dimA);
		//	printf("ElementWiseMultiply_CUDA, matrix B:\r\n");
		//	printMatrix_CUDA << <1, 1 >> > (B, dimB);
		//}
		//int idx = i + (j * dimC.y);
		double a = A[sourceIndex];
		double b = B[sourceIndex];
		C[targetIndex] = a * b;
		//printf("i=%i, j=%i idx=%i | %i = %i * %i\r\n", i, j, idx, C[idx], a, b);
	}
}

__global__ void
Leicester::CudaLib::ElementWiseMultiply_CUDA2(double *C, double *A, double *B, dim3 dimC)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	int sourceLength = dimC.x * dimC.y;
	int sourceIndex = i + (j * blockDim.y);
	int targetIndex = i + (j * blockDim.y);
	if ((sourceIndex <= sourceLength - 1) & (targetIndex < dimC.y))
	{
		//if (i == 0 & j == 0)
		//{
		//	printf("ElementWiseMultiply_CUDA, matrix A:\r\n");
		//	printMatrix_CUDA << <1, 1 >> > (A, dimA);
		//	printf("ElementWiseMultiply_CUDA, matrix B:\r\n");
		//	printMatrix_CUDA << <1, 1 >> > (B, dimB);
		//}
		//int idx = i + (j * dimC.y);
		double a = A[sourceIndex];
		double b = B[sourceIndex];
		C[targetIndex] = a * b;
		//printf("i=%i, j=%i idx=%i | %i = %i * %i\r\n", i, j, idx, C[idx], a, b);
	}
}

__global__ void
Leicester::CudaLib::ElementWiseMultiply_CUDA3(double *C, double *A, double *B, dim3 dimC)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	int sourceLength = dimC.x * dimC.y;
	int sourceIndex = i + (j * blockDim.y);
	int targetIndex = i + (j * blockDim.y);
	if ((sourceIndex <= sourceLength - 1) & (targetIndex < dimC.y -1))
	{
		//if (i == 0 & j == 0)
		//{
		//	printf("ElementWiseMultiply_CUDA, matrix A:\r\n");
		//	printMatrix_CUDA << <1, 1 >> > (A, dimA);
		//	printf("ElementWiseMultiply_CUDA, matrix B:\r\n");
		//	printMatrix_CUDA << <1, 1 >> > (B, dimB);
		//}
		//int idx = i + (j * dimC.y);
		double a = A[sourceIndex];
		double b = B[sourceIndex];
		//if (targetIndex > dimC.y - 1)
		//	printf("targetIndex=%i yMax=%i xMax=%i\r\n", targetIndex, dimC.y-1, dimC.x-1);
		//if (i == 0 & j == 0)
		//{
		//	printf("C elements=%i\r\n", C[0]);
		//}
		C[targetIndex + 1] = a * b;
		//printf("i=%i, j=%i idx=%i | %i = %i * %i\r\n", i, j, idx, C[idx], a, b);
	}
}

__global__ void
Leicester::CudaLib::ElementWiseMultiply_CUDA4(double *C, double *A, double *B, dim3 dimC)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	int sourceLength = dimC.x * dimC.y;
	int sourceIndex = i + (j * blockDim.y);
	int targetIndex = i + (j * blockDim.y);
	if ((sourceIndex <= sourceLength - 1) & (targetIndex < dimC.y))
	{
		//if (i == 0 & j == 0)
		//{
		//	printf("ElementWiseMultiply_CUDA, matrix A:\r\n");
		//	printMatrix_CUDA << <1, 1 >> > (A, dimA);
		//	printf("ElementWiseMultiply_CUDA, matrix B:\r\n");
		//	printMatrix_CUDA << <1, 1 >> > (B, dimB);
		//}
		//int idx = i + (j * dimC.y);
		double a = A[sourceIndex];
		double b = B[sourceIndex];
		C[targetIndex] = a * b;
		//printf("i=%i, j=%i idx=%i | %i = %i * %i\r\n", i, j, idx, C[idx], a, b);
	}
}

__global__ void
Leicester::CudaLib::ElementWiseMultiply_CUDA5(double *C, double *A, double *B, dim3 dimC)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	int sourceLength = dimC.x * dimC.y;
	int sourceIndex = i + (j * blockDim.y);
	int targetIndex = i + (j * blockDim.y);
	if ((sourceIndex <= sourceLength - 1) & (targetIndex < dimC.y))
	{
		//if (i == 0 & j == 0)
		//{
		//	printf("ElementWiseMultiply_CUDA, matrix A:\r\n");
		//	printMatrix_CUDA << <1, 1 >> > (A, dimA);
		//	printf("ElementWiseMultiply_CUDA, matrix B:\r\n");
		//	printMatrix_CUDA << <1, 1 >> > (B, dimB);
		//}
		//int idx = i + (j * dimC.y);
		double a = A[sourceIndex];
		double b = B[sourceIndex];
		C[targetIndex] = a * b;
		//printf("i=%i, j=%i idx=%i | %i = %i * %i\r\n", i, j, idx, C[idx], a, b);
	}
}

__global__ void
Leicester::CudaLib::ElementWiseMultiplyThree_CUDA(double *D, double *A, double *B, double *C, dim3 dimD)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int sourceLength = dimD.x * dimD.y;
	int sourceIndex = i + (j * blockDim.y);
	int targetIndex = i + (j * blockDim.y);
	if ((sourceIndex <= sourceLength - 1) & (targetIndex <= dimD.y))
	{
	
		//int idx = i + (j * dimD.x);
		D[targetIndex] = A[sourceIndex] * B[sourceIndex] * C[sourceIndex];
	}
}

__global__ void
Leicester::CudaLib::MatrixScalarMultiply_CUDA(double *C, double *A, double b, dim3 dimA)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int sourceLength = dimA.x * dimA.y;
	int sourceIndex = i + (j * blockDim.y);
	int targetIndex = i + (j * blockDim.y);
	if ((sourceIndex <= sourceLength - 1) & (targetIndex <= dimA.y))
	{
		//int idx = i + (j * dimA.x);
		double a = A[sourceIndex];
		C[targetIndex] = a * b;
	}
}

__global__ void
Leicester::CudaLib::MatrixSubtractScalar_CUDA(double *C, double *A, double b, dim3 dimA)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	int sourceLength = dimA.x * dimA.y;
	int sourceIndex = i + (j * blockDim.y);
	int targetIndex = i + (j * blockDim.y);
	if ((sourceIndex <= sourceLength - 1) & (targetIndex <= dimA.y))
	{
		//int idx = i + (j * dimA.x);
		double a = A[sourceIndex];
		C[targetIndex] = a - b;
		//printf("i=%i, j=%i idx=%i | %i = %i - %i\r\n", i, j, idx, C[idx], a, b);
	}
}

__global__ void
Leicester::CudaLib::MatrixAddScalar_CUDA(double *C, double *A, double b, dim3 dimA)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int sourceLength = dimA.x * dimA.y;
	int sourceIndex = i + (j * blockDim.y);
	int targetIndex = i + (j * blockDim.y);
	if ((sourceIndex <= sourceLength - 1) & (targetIndex <= dimA.y))
	{
		//int idx = j + (i * dimA.x);
		double a = A[sourceIndex];
		C[targetIndex] = a + b;
	}
}

__global__ void
Leicester::CudaLib::GetColumn(double * result, int col, double *A, dim3 dimA)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int sourceMin = dimA.y * col;
	int sourceMax = sourceMin + dimA.y;

	int sourceIndex = i + (j * blockDim.y);
	int targetIndex = sourceIndex - (col * dimA.y);
	if ((sourceIndex < sourceMax) & (sourceIndex >= sourceMin) & (targetIndex >= 0) & (targetIndex <= dimA.y - 1))
	{
		result[targetIndex] = A[sourceIndex];
	}
}

__global__ void
Leicester::CudaLib::GetRow(double * result, int row, double *A, dim3 dimA)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	if (i == row & j < dimA.x)
	{
		int idx = i + (j * dimA.y);
		result[j] = A[idx];
	}
}

__global__ void
Leicester::CudaLib::SetColumn(double * matrix, int col, double *vector, dim3 dimMatrix)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int targetMin = dimMatrix.y * col;
	int targetMax = targetMin + dimMatrix.y;

	int targetIndex = i + (j * blockDim.y);
	int sourceIndex = targetIndex - (col * dimMatrix.y);

	if ((targetIndex < targetMax) & (targetIndex >= targetMin) & (sourceIndex >= 0) & (sourceIndex < dimMatrix.y))
	{
		matrix[targetIndex] = vector[sourceIndex];
	}
}

__global__ void
Leicester::CudaLib::SetColumnLogged(double * matrix, int col, double *vector, dim3 dimMatrix)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int targetMin = dimMatrix.y * col;
	int targetMax = targetMin + dimMatrix.y;

	int targetIndex = i + (j * blockDim.y);
	int sourceIndex = targetIndex - (col * dimMatrix.y);
	//printf("targetMax=%i targetMin=%i sourceMax=%i\r\n", targetMax, targetMin, dimMatrix.y);
	//printf("source=%i target=%i\r\n", sourceIndex, targetIndex);
	if ((targetIndex < targetMax) & (targetIndex >= targetMin) & (sourceIndex >= 0) & (sourceIndex < dimMatrix.y))
	{
		//printf("source=%i target=%i val=%i\r\n", sourceIndex, targetIndex, vector[sourceIndex]);
		printf("targetMax=%i targetMin=%i sourceMax=%i\r\n", targetMax, targetMin, dimMatrix.y);
		//int idx = i + (j * dimMatrix.y);
		matrix[targetIndex] = vector[sourceIndex];
	}
}
