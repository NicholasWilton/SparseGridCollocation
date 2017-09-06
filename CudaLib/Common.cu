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

using namespace std;
using namespace thrust; 



__global__ void
matrixMul_CUDA(double *C, double *A, double *B, int wA, int wB)
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

int matrixMultiply(int block_size, double * h_A, dim3 &dimsA, double * h_B, dim3 &dimsB)
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
printMatrix_CUDA(double *matrix, dim3 dimMatrix)
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
				int idx = (y * dimMatrix.x) + x;
				//if ( mSize > idx)
				printf("indx=%i value=%f\t", idx, matrix[idx]);
			}
			printf("\r\n");
		}
	}
}

__global__ void
matrixFill_CUDA(double *matrix, dim3 dimMatrix, double fill)
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
FAI_CUDA(double *FAI, double a, double *TP, int tpCol, double CN, double c, dim3 dimTP)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	//printf("start FAI_CUDA i=%i\r\n", i);

	if (i < dimTP.y & j == tpCol)
	{
		int idx = j + (i * dimTP.x);
		if (i== 9) printf("input FAI_CUDA TP[i]=%f CN=%f, a=%f, c=%f\r\n", TP[idx], CN, a, c);
		double a1 = a * (TP[idx] - CN);
		if (i == 9) printf("FAI_CUDA a1=%f\r\n", a1);
		double b1 = -(a1 * a1) / (c * c);
		if (i == 9) printf("FAI_CUDA b1=%f\r\n", b1);
		double e1 = expm1(b1) + 1;
		if (i == 9) printf("FAI_CUDA e1=%f\r\n", e1);
		FAI[i] = e1;
		if (i == 9) printf("FAI_CUDA e1=%f\r\n", e1);
		
		printf("FAI_CUDA FAI[%i]=%f\r\n",i, FAI[i]);
	}
}

// D = a * (B - c)
__global__ void
ScalarVectorDifference_CUDA(double *D, double a, double *B, double c, dim3 dimB)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < dimB.y & j < dimB.x)
	{
		int idx = j + (i * j);

		D[idx] = a * (B[idx] - c);
	}
}

__global__ void
ElementWiseMultiply_CUDA(double *C, double *A, double *B, dim3 dimA, dim3 dimB)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < dimA.y & j < dimB.x)
	{
		int idx = j + (i * dimB.x);
		//printf("i=%i, j=%i idx=%i\r\n", i, j, idx);
		C[idx] = A[idx] * B[idx];
	}
}

__global__ void
ElementWiseMultiply_CUDA(double *D, double *A, double *B, double *C, dim3 dimD)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < dimD.y & j < dimD.x)
	{
		int idx = j + (i * dimD.x);
		D[idx] = A[idx] * B[idx] * C[idx];
	}
}

__global__ void
MatrixScalarMultiply_CUDA(double *C, double *A, double b, dim3 dimA)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < dimA.y & j < dimA.x)
	{
		int idx = j + (i * dimA.x);
		C[idx] = A[idx] * b;
	}
}

__global__ void
MatrixSubtractScalar_CUDA(double *C, double *A, double b, dim3 dimA)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < dimA.y & j < dimA.x)
	{
		int idx = j + (i * dimA.x);
		C[idx] = A[idx] - b;
	}
}

__global__ void
MatrixAddScalar_CUDA(double *C, double *A, double b, dim3 dimA)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < dimA.y & j < dimA.x)
	{
		int idx = j + (i * dimA.x);
		C[idx] = A[idx] + b;
	}
}

__global__ void
GetColumn(double * result, int col, double *A, dim3 dimA)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < dimA.y & j == col)
	{
		int idx = j + (i * dimA.x);
		result[i] = A[idx];
	}
}

__global__ void
GetRow(double * result, int row, double *A, dim3 dimA)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	if (i == row & j < dimA.x)
	{
		int idx = j + (i * dimA.x);
		result[j] = A[idx];
	}
}

__global__ void
SetColumn(double * matrix, int col, double *vector, dim3 dimMatrix)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < dimMatrix.y & j == col)
	{
		int idx = j + (i * dimMatrix.x);
		matrix[idx] = vector[i];
	}
}
