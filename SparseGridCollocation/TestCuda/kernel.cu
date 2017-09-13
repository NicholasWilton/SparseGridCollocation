
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "Common.cuh"
#include <stdio.h>
#include "C:\Users\User\Source\Repos\SparseGridCollocation\SparseGridCollocation\include\eigen-eigen-67e894c6cd8f\Eigen\StdVector"


using namespace Eigen;

__global__ void gpuAssert(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess)
	{
		printf("GPUassert: %i %s %d\n", code, file, line);
	}
}

__global__ void
ElementWiseMultiply_CUDA(double *C, double *A, double *B, int rows, int cols)
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

void SetupElementWiseTest(VectorXd &a, VectorXd &b, VectorXd &c)
{
	int rows = 1025;
	a = VectorXd::Zero(rows);
	b = VectorXd::Zero(rows);
	c = VectorXd::Zero(rows);
	for (int i = 0; i < rows; i++)
	{
		a[i] = (double)i;
		b[i] = (double)i+1;
	}
}
bool AssertElementWiseTest(VectorXd &a, VectorXd &b, VectorXd &c)
{
	VectorXd expected = a.array() * b.array();
	bool error = false;
	for (int i = 0; i < c.rows(); i++)
	{
		if (expected[i] != c[i])
		{
			printf("error at index=%i, expected=%f!=%f\r\n", i, expected[i], c[i]);
			error = true;
		}
	}
	return error;
}
int main()
{
	gpuAssert << <1, 1 >> >(cudaDeviceSynchronize(), __FILE__, __LINE__);

	VectorXd A;
	VectorXd B;
	VectorXd C;
	SetupElementWiseTest(A, B, C);
	//Allocate the input on host and device
	double *a, *b;
	a = A.data();
	b = B.data();
	double *c = (double*)malloc(sizeof(double) * A.rows() * A.cols());
	
	double *d_a, *d_b, *d_c;
	
	cudaError_t e = cudaMalloc((void **)&d_a, A.rows() * A.cols() * sizeof(double));
	if (e != cudaSuccess)
		printf("cudaMalloc d_a returned error %s (code %d), line(%d) when allocating %i bytes\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * A.rows() * A.cols());
	e = cudaMalloc((void **)&d_b, B.rows() * B.cols() * sizeof(double));
	if (e != cudaSuccess)
		printf("cudaMalloc d_b returned error %s (code %d), line(%d) when allocating %i bytes\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * B.rows() * B.cols());
	e = cudaMalloc((void **)&d_c, C.rows() * C.cols() * sizeof(double));
	if (e != cudaSuccess)
		printf("cudaMalloc d_c returned error %s (code %d), line(%d) when allocating %i bytes\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * C.rows() * C.cols());
	
	e = cudaMemcpy(d_a, a, sizeof(double) * A.rows() * A.cols(), cudaMemcpyKind::cudaMemcpyHostToDevice);
	if (e != cudaSuccess)
		printf("cudaMemcpy d_a returned error %s (code %d), line(%d) when copying %i bytes\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * A.rows() * A.cols());
	e = cudaMemcpy(d_b, b, sizeof(double) * B.rows() * B.cols(), cudaMemcpyKind::cudaMemcpyHostToDevice);
	if (e != cudaSuccess)
		printf("cudaMemcpy d_b returned error %s (code %d), line(%d) when copying %i bytes\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * B.rows() * B.cols());
	
	cudaDeviceProp deviceProp;

	cudaGetDeviceProperties(&deviceProp, 0);
	int block_size = (deviceProp.major < 2) ? 16 : 32;
	
	dim3 dimA(A.cols(), A.rows());
	dim3 dimB(B.cols(), B.rows());
	dim3 dimC(C.cols(), C.rows());
	dim3 threads(block_size, block_size);
	
	dim3 grid(1, 1);
	
	ElementWiseMultiply_CUDA << < grid, threads >> > (d_c, d_a, d_b, A.rows(), A.cols());
	
	gpuAssert << <1, 1 >> >(cudaDeviceSynchronize(), __FILE__, __LINE__);

	gpuAssert << <1, 1 >> >(cudaPeekAtLastError(), __FILE__, __LINE__);

	e = cudaMemcpy(c, d_c, sizeof(double) * C.rows() * C.cols(), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	if (e != cudaSuccess)
		printf("cudaMemcpy d_D returned error %s (code %d), line(%d) when copying%i\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * A.rows() * A.cols());
	
	
	gpuAssert << <1, 1 >> >(cudaDeviceSynchronize(), __FILE__, __LINE__);

	
	Eigen::Map<Eigen::MatrixXd> dataMapD(c, C.rows(), C.cols());
	C = dataMapD.eval();

	if (!AssertElementWiseTest(A, B, C))
		printf("passed");
	else
		printf("failed");

	free(a);
	free(b);
	free(c);
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	cudaDeviceReset();
	
}
