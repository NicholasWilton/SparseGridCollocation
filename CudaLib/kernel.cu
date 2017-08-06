#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>
#include ".\cuda_include\helper_string.h"  // helper for shared functions common to CUDA Samples


#include <cublas_v2.h>
#include <cusolverDn.h>
// CUDA and CUBLAS functions
//#include ".\cuda_include\helper_functions.h"
#include ".\cuda_include\helper_cuda.h"

#include <stdio.h>




using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace Eigen;


__global__ void addKernel(int *c, const int *a, const int *b)
{
}

VectorXd PushAndQueue(double push, VectorXd A, double queue)
{
	VectorXd result(A.rows() + 2);
	result[0] = push;
	for (int i = 0; i < A.rows(); i++)
	{
		result[i] = A[i];
	}
	result[A.rows() + 1] = queue;
	return result;
}


int MethodOfLines::MoLiteration(double Tend, double Tdone, double dt, double *G, int GRows, int GCols, double *lamb, int lambRows, int lambCols, double inx2, double r, double K, MatrixXd A1, MatrixXd Aend, MatrixXd H)
{
	int count = 0;
	while (Tend - Tdone > 1E-8)
	{
		Tdone += dt;
		
		int sizeG = GRows * GCols;
		int sizeLamb = lambRows * lambCols;
		int memG = sizeof(double) * sizeG;
		int memLamb = sizeof(double) * sizeLamb;

		double *d_G, *d_lamb, *d_FFF;
		int sizeFFF = GRows * lambCols;
		int memFFF = sizeof(double)* sizeFFF;

		double *h_FFF = (double *)malloc(memFFF);
		double *h_CUBLAS = (double *)malloc(memFFF);

		checkCudaErrors(cudaMalloc((void **)&d_G, memG));
		checkCudaErrors(cudaMalloc((void **)&d_lamb, memLamb));
		checkCudaErrors(cudaMemcpy(d_G, G, memG, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_lamb, lamb, memLamb, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void **)&d_FFF, memFFF));

		cublasHandle_t handle;
		checkCudaErrors(cublasCreate(&handle));
		const double alpha = 1.0;
		const double beta = 1.0;
		checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, GRows, lambCols, GCols, &alpha, d_G, GRows, d_lamb, lambRows, &beta, d_FFF, GRows));
		
		checkCudaErrors(cudaMemcpy(h_FFF, d_FFF, memFFF, cudaMemcpyDeviceToHost));
		printf("after cublasDgemm:\r\n");
		//double i[] = h_FFF;
		VectorXd FFF = Map<VectorXd >(h_FFF, GRows, lambCols);
		VectorXd fff = PushAndQueue(0, FFF, inx2 - exp(-r*Tdone)*K);
		printf("after PushAndQueue:\r\n");
		MatrixXd HH(A1.cols(), A1.cols());
		HH.row(0) = A1;
		HH.middleRows(1, HH.rows() - 2) = H;
		HH.row(HH.rows() - 1) = Aend;
		printf("after HH construction:\r\n");
		//LLT<MatrixXd> lltOfA(HH);
		//lamb = lltOfA.solve(fff);

		cusolverDnHandle_t cusolverH = NULL; 
		cublasHandle_t cublasH = NULL; 
		cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS; 
		cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS; 
		cudaError_t cudaStat1 = cudaSuccess; 
		cudaError_t cudaStat2 = cudaSuccess; 
		cudaError_t cudaStat3 = cudaSuccess; 
		cudaError_t cudaStat4 = cudaSuccess; 
		const int m = HH.rows(); const int lda = m; const int ldb = m; const int nrhs = 1; // number of right hand side vectors
		double *XC = new double[ldb*nrhs];
		
		double *d_A = NULL; // linear memory of GPU 
		double *d_tau = NULL; // linear memory of GPU 
		double *d_B = NULL; int *devInfo = NULL; // info in gpu (device copy) 
		double *d_work = NULL; 
		int lwork = 0; 
		int info_gpu = 0; 
		const double one = 1;

		cusolver_status = cusolverDnCreate(&cusolverH); 
		assert(CUSOLVER_STATUS_SUCCESS == cusolver_status); 
		printf("after cusolver create:\r\n");
		cublas_status = cublasCreate(&cublasH); 
		assert(CUBLAS_STATUS_SUCCESS == cublas_status);
		printf("after cublas create:\r\n");

		cudaStat1 = cudaMalloc((void**)&d_A, sizeof(double) * lda * m); 
		cudaStat2 = cudaMalloc((void**)&d_tau, sizeof(double) * m); 
		cudaStat3 = cudaMalloc((void**)&d_B, sizeof(double) * ldb * nrhs); 
		cudaStat4 = cudaMalloc((void**)&devInfo, sizeof(int)); 
		assert(cudaSuccess == cudaStat1); 
		assert(cudaSuccess == cudaStat2); 
		assert(cudaSuccess == cudaStat3); 
		assert(cudaSuccess == cudaStat4); 
		cudaStat1 = cudaMemcpy(d_A, HH.data(), sizeof(double) * lda * m, cudaMemcpyHostToDevice); 
		cudaStat2 = cudaMemcpy(d_B, fff.data(), sizeof(double) * ldb * nrhs, cudaMemcpyHostToDevice); 
		assert(cudaSuccess == cudaStat1); assert(cudaSuccess == cudaStat2);

		// step 3: query working space of geqrf and ormqr 
		cusolver_status = cusolverDnDgeqrf_bufferSize( cusolverH, m, m, d_A, lda, &lwork); 
		assert (cusolver_status == CUSOLVER_STATUS_SUCCESS); 
		cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork); 
		printf("after initialisation:\r\n");
		assert(cudaSuccess == cudaStat1); 
		// step 4: compute QR factorization 
		cusolver_status = cusolverDnDgeqrf( cusolverH, m, m, d_A, lda, d_tau, d_work, lwork, devInfo); 
		cudaStat1 = cudaDeviceSynchronize(); 
		assert(CUSOLVER_STATUS_SUCCESS == cusolver_status); 
		assert(cudaSuccess == cudaStat1); 
		printf("after QR factorization:\r\n");
		// check if QR is good or not 
		cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost); 
		assert(cudaSuccess == cudaStat1); 
		printf("after geqrf: info_gpu = %d\n", info_gpu); 
		assert(0 == info_gpu); 
		// step 5: compute Q^T*B 
		cusolver_status= cusolverDnDormqr( cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, m, d_A, lda, d_tau, d_B, ldb, d_work, lwork, devInfo); 
		cudaStat1 = cudaDeviceSynchronize(); 
		assert(CUSOLVER_STATUS_SUCCESS == cusolver_status); 
		assert(cudaSuccess == cudaStat1);

		// check if QR is good or not 
		cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost); 
		assert(cudaSuccess == cudaStat1); 
		printf("after ormqr: info_gpu = %d\n", info_gpu); 
		assert(0 == info_gpu); 
		// step 6: compute x = R \ Q^T*B 
		cublas_status = cublasDtrsm( cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, nrhs, &one, d_A, lda, d_B, ldb); 
		cudaStat1 = cudaDeviceSynchronize(); assert(CUBLAS_STATUS_SUCCESS == cublas_status); 
		assert(cudaSuccess == cudaStat1); 
		cudaStat1 = cudaMemcpy(XC, d_B, sizeof(double)*ldb*nrhs, cudaMemcpyDeviceToHost); 
		assert(cudaSuccess == cudaStat1); 
		
		/*printf("X = (matlab base-1)\n"); 
		printMatrix(m, nrhs, XC, ldb, "X"); */

		// free resources 
		if (d_A ) cudaFree(d_A); 
		if (d_tau ) cudaFree(d_tau); 
		if (d_B ) cudaFree(d_B); 
		if (devInfo) cudaFree(devInfo); 
		if (d_work ) cudaFree(d_work); 
		if (cublasH ) cublasDestroy(cublasH); 
		if (cusolverH) cusolverDnDestroy(cusolverH); 
		cudaDeviceReset();

		
		
		count++;
		printf("%i\r\n", count);
	}
    return 0;
}

void main()
{}

