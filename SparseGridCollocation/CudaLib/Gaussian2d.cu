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
#include "Gaussian2d2.cuh"

using namespace std;
using namespace thrust;


namespace Leicester
{
	namespace CudaLib
	{
		__global__ void
			Gaussian2d_CUDA(double *D, double *Dt, double *Dx, double *Dxx,
				double *TP, int TPx, int TPy, double *CN, int CNx, int CNy, double *A, int Ax, int Ay, double *C, int Cx, int Cy)
		{
			int i = blockDim.y * blockIdx.y + threadIdx.y;
			int j = blockDim.x * blockIdx.x + threadIdx.x;
			__syncthreads();
			if (i == 0 & j == 0)
			{
				//printf("start mqd2_CUDA i=%i j =%i \r\n", i, j);
				//double* D = (double *)malloc(sizeof(double) * TPx *CNy);
				//double* Dt = (double *)malloc(sizeof(double) * TPx *CNy);
				//double* Dx = (double *)malloc(sizeof(double) * TPx *CNy);
				//double* Dxx = (double *)malloc(sizeof(double) * TPx *CNy);
				//printf("allocated arrays mqd2_CUDA i=%i j =%i \r\n", i, j);

				dim3 threads(32, 32);
				//dim3 grid(CNx / threads.x, CNy / threads.y);
				dim3 grid(1, 1);
				dim3 dimTP(TPx, TPy);
				dim3 dimCN(CNx, CNy);
				dim3 dimA(Ax, Ay);
				dim3 dimC(Cx, Cy);
				//printf("TP size=%f", sizeof(TP));
				//printMatrix_CUDA << < dim3(1,1), dim3(1,1)>> > (TP, dimTP);
				//gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
				//gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());

				//printf("dimTPx=%i dimTPy=%i dimCNx=%i dimCNy=%i dimAx=%i dimAy=%i dimCx=%i dimCy=%i\r\n", dimTP.x, dimTP.y, dimCN.x, dimCN.y, dimA.x, dimA.y, dimC.x, dimC.y);
				Gaussian2d2_CUDA << <1, dim3(1, CNy) >> > (D, Dt, Dx, Dxx, TP, dimTP, CN, dimCN, A, dimA, C, dimC);
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);

				//printf("D size=%f", sizeof(D));
				//printMatrix_CUDA << < dim3(1, 1), dim3(1, 1) >> > (D, dim3(TPy, TPy));
				//gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
				//gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());

				//__syncthreads();
				//result[0] = D;
				//result[1] = Dt;
				//result[2] = Dx;
				//result[3] = Dxx;
			}
			__syncthreads();
			//printf("end mqd2_CUDA");
		}
	}
}