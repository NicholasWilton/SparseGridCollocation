
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
#include "mqd2.cuh"


using namespace std;
using namespace thrust;

#define gpuErrchk1(ans) { gpuAssert1((ans), __FILE__, __LINE__, false); }
inline void gpuAssert1(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

//index of storage array in column-major format
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


//vector<MatrixXd> mqd2(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C)
//{
//	vector<MatrixXd> result;
//	int Num = CN.rows();
//	int N = TP.rows();
//
//	MatrixXd D(N, Num);
//	D.fill(1.0);
//	MatrixXd Dt(N, Num);
//	Dt.fill(1.0);
//	MatrixXd Dx(N, Num);
//	Dx.fill(1.0);
//	MatrixXd Dxx(N, Num);
//	Dxx.fill(1.0);
//
//	for (int j = 0; j < Num; j++)
//	{
//		VectorXd a1 = A(0, 0)*(TP.col(0).array() - CN(j, 0));
//		VectorXd b1 = -(a1.array() * a1.array()) / (C(0, 0) *C(0, 0));
//		VectorXd FAI1 = b1.array().exp();
//		//VectorXd FAI1 = RBF::exp(b1);
//
//		VectorXd a2 = A(0, 1)*(TP.col(1).array() - CN(j, 1));
//		VectorXd b2 = -(a2.array() * a2.array()) / (C(0, 1) *C(0, 1));
//
//		VectorXd FAI2 = b2.array().exp();
//		//VectorXd FAI2 = RBF::exp(b2);
//		D.col(j) = FAI1.array() * FAI2.array();
//
//		VectorXd a3 = -2 * (A(0, 0) / C(0, 0)) * (A(0, 0) / C(0, 0)) * (TP.col(0).array() - CN(j, 0));
//		VectorXd b3 = a3.array() * FAI1.array();
//		VectorXd c3 = b3.array() * FAI2.array();
//		Dt.col(j) = c3;
//
//		VectorXd a4 = -2 * (A(0, 1) / C(0, 1)) * (A(0, 1) / C(0, 1)) * (TP.col(1).array() - CN(j, 1));
//		VectorXd b4 = TP.col(1).array() * a4.array() * FAI1.array();
//		VectorXd c4 = b4.array() * FAI2.array();
//		Dx.col(j) = c4;
//
//		double sA = A(0, 1) * A(0, 1);
//		double qA = A(0, 1) * A(0, 1) * A(0, 1) * A(0, 1);
//		double sC = C(0, 1) * C(0, 1);
//		double qC = C(0, 1) * C(0, 1) * C(0, 1) * C(0, 1);
//		VectorXd dTpCn = TP.col(1).array() - CN(j, 1);
//
//		VectorXd a5 = 4 * qA * (dTpCn.array() * dTpCn.array() / qC);
//		VectorXd b5 = -2 * sA / sC + a5.array();
//		VectorXd c5 = b5.array()  * FAI2.array() * FAI1.array();
//		VectorXd d5 = (TP.col(1).array() * TP.col(1).array()).array() * c5.array();
//		Dxx.col(j) = d5;
//	}
//	result.push_back(D);
//	result.push_back(Dt);
//	result.push_back(Dx);
//	result.push_back(Dxx);
//	return result;
//}


/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
* wA is A's width and wB is B's width
*/



void mqd2()
{

	//MatrixXd C(1, 2);
	//C << 1.73, 600;
	double C[2] = { 1.73, 600 };
	//MatrixXd A(1, 2);
	//A << 2, 4;
	double A[2] = { 2, 4 };

	//MatrixXd TX = MatrixXd(15, 2);
	//TX << 0, 0, 0, 75, 0, 150, 0, 225, 0, 300, 0.4325, 0, 0.4325, 75, 0.4325, 150, 0.4325, 225, 0.4325, 300, 0.865, 0, 0.865, 75, 0.865, 150, 0.865, 225, 0.865, 300;
	//double TX[30] = { 0,0,0,0,0,0.432500000000000,0.432500000000000,0.432500000000000,0.432500000000000,0.432500000000000,0.865000000000000,0.865000000000000,0.865000000000000,0.865000000000000,0.865000000000000,0,75,150,225,300,0,75,150,225,300,0,75,150,225,300 };
	double TX[30] = { 0, 0, 0, 75, 0, 150, 0, 225, 0, 300, 0.4325, 0, 0.4325, 75, 0.4325, 150, 0.4325, 225, 0.4325, 300, 0.865, 0, 0.865, 75, 0.865, 150, 0.865, 225, 0.865, 300 };
	double TX1[30] = { 0, 0, 0, 75, 0, 150, 0, 225, 0, 300, 0.4325, 0, 0.4325, 75, 0.4325, 150, 0.4325, 225, 0.4325, 300, 0.865, 0, 0.865, 75, 0.865, 150, 0.865, 225, 0.865, 300 };
	//double TX1[30] = { 0,0,0,0,0,0.432500000000000,0.432500000000000,0.432500000000000,0.432500000000000,0.432500000000000,0.865000000000000,0.865000000000000,0.865000000000000,0.865000000000000,0.865000000000000,0,75,150,225,300,0,75,150,225,300,0,75,150,225,300 };
	//MatrixXd TX1 = MatrixXd(TX);

	double *a, *c, *tx, *tx1;
	a = A;
	c = C;
	tx = TX;
	tx1 = TX1;
	double *d_a, *d_c, *d_tx, *d_tx1;
	
	cudaError_t e = cudaMalloc((void **)&d_a, 2 * sizeof(double));
	if (e != cudaSuccess)
		printf("cudaMalloc d_a returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);
	e = cudaMalloc((void **)&d_c, 2 * sizeof(double));
	if (e != cudaSuccess)
		printf("cudaMalloc d_c returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);
	e = cudaMalloc((void **)&d_tx, 30 * sizeof(double));
	if (e != cudaSuccess)
		printf("cudaMalloc d_tx returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);
	e = cudaMalloc((void **)&d_tx1, 30 * sizeof(double));
	if (e != cudaSuccess)
		printf("cudaMalloc d_tx1 returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);

	e = cudaMemcpy(d_a, a, sizeof(double) * 2, cudaMemcpyKind::cudaMemcpyHostToDevice);
	if (e != cudaSuccess)
		printf("cudaMemcpy d_a returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);
	e = cudaMemcpy(d_c, c, sizeof(double) * 2, cudaMemcpyKind::cudaMemcpyHostToDevice);
	if (e != cudaSuccess)
		printf("cudaMemcpy d_c returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);
	e = cudaMemcpy(d_tx, tx, sizeof(double) * 30, cudaMemcpyKind::cudaMemcpyHostToDevice);
	if (e != cudaSuccess)
		printf("cudaMemcpy d_tx returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);
	e = cudaMemcpy(d_tx1, tx, sizeof(double) * 30, cudaMemcpyKind::cudaMemcpyHostToDevice);
	if (e != cudaSuccess)
		printf("cudaMemcpy d_tx1 returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);

	double **h_result(0);
	double **d_result;
	e = cudaMalloc((void **)&d_result, 4 * sizeof(double));
	if (e != cudaSuccess)
		printf("cudaMalloc d_result returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);

	cudaDeviceProp deviceProp;

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
	int block_size = (deviceProp.major < 2) ? 16 : 32;
	dim3 dimTx(2,15);
	dim3 dimA(2, 1);
	dim3 dimC(2, 1);
	dim3 threads(block_size, block_size);
	//dim3 grid(dimTx.x / threads.x, dimTx.y / threads.y);
	dim3 grid(1, 1);
	//test << < grid, threads >> > ();
	//cudaDeviceSynchronize();

	mqd2_CUDA<<< grid,threads>> > (d_result, d_tx, dimTx.x, dimTx.y, d_tx1, dimTx.x, dimTx.y, d_a, dimA.x, dimA.y, d_c, dimC.x, dimC.y);
	//mqd2_CUDA<32>(d_result, d_tx, dimTx.x, dimTx.y, d_tx1, dimTx.x, dimTx.y, d_a, dimA.x, dimA.y, d_c, dimC.x, dimC.y);
	gpuErrchk1(cudaPeekAtLastError());
	gpuErrchk1(cudaDeviceSynchronize());

	e = cudaMemcpy(d_result, h_result, sizeof(double) * 15 * 15, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	if (e != cudaSuccess)
		printf("cudaMemcpy d_result returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);

}
int main()
{
	int devID = 0;
	cudaError_t error;
	cudaDeviceProp deviceProp;
	error = cudaGetDevice(&devID);
	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}
	else
	{
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	}

	int block_size = (deviceProp.major < 2) ? 16 : 32;

	dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
	dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);




	//int matrix_result = matrixMultiply(block_size, dimsA, dimsB);

	mqd2();

    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
