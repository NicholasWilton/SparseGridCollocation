#include "kernel.h"


using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
using namespace Eigen;
using namespace thrust;

namespace Leicester
{
	namespace CudaLib
	{
		vector<MatrixXd> GaussianND(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C)
		{
			vector<MatrixXd> result;// V, Vt, Vx Vxy
			int Num = CN.rows();
			int N = TP.rows();
			int dimensions = TP.cols();

			MatrixXd D(N, Num);
			D.fill(1.0);

			vector<MatrixXd> Derivatives;
			Derivatives.push_back(D);
			for (int d = 0; d < 3; d++)
			{
				MatrixXd Dx(N, Num);
				Dx.fill(1.0);
				Derivatives.push_back(Dx);
			}


			for (int j = 0; j < Num; j++)
			{
				vector<VectorXd> FAIn;
				for (int d = 0; d < dimensions; d++)
				{
					VectorXd a1 = A(0, d)*(TP.col(d).array() - CN(j, d));
					VectorXd FAI = (-((A(0, d)*(TP.col(d).array() - CN(j, d))).array() * (A(0, d)*(TP.col(d).array() - CN(j, d))).array()) / (C(0, d) *C(0, d))).array().exp();
					Derivatives[0].col(j).array() *= FAI.array();
					FAIn.push_back(FAI);
				}

				VectorXd vt = -2 * (A(0, 0) / C(0, 0)) * (A(0, 0) / C(0, 0)) * (TP.col(0).array() - CN(j, 0));
				Derivatives[1].col(j) = vt;

				VectorXd sumij = VectorXd::Zero(TP.rows());
				MatrixXd dS(TP.rows(), dimensions - 1);
				for (int d = 1; d < dimensions; d++)
				{
					dS.col(d - 1) = (-2 * (A(0, d) / C(0, d)) * (A(0, d) / C(0, d)) * (TP.col(d).array() - CN(j, d))).array() * TP.col(d).array();
					VectorXd sumi = VectorXd::Zero(TP.rows());
					for (int i = 1; i < TP.cols(); i++)
					{
						sumi.array() = sumi.array() + TP.col(d).array() * TP.col(i).array() * (-2 * (A(0, d) * A(0, d)) / (C(0, d) *C(0, d)) + (4 * (A(0, d) * A(0, d)* A(0, d) * A(0, d)) * ((TP.col(d).array() - CN(j, i)).array() * (TP.col(d).array() - CN(j, i)).array() / (C(0, d) *C(0, d)*C(0, d) *C(0, d)))).array()).array();
					}
					sumij.array() = sumij.array() + sumi.array();

				}
				VectorXd sum = dS.rowwise().sum();
				Derivatives[2].col(j) = sum;

				Derivatives[3].col(j) = sumij;

				for (int d = 1; d < Derivatives.size(); d++)
					Derivatives[d].col(j).array() *= Derivatives[0].col(j).array();

			}
			return Derivatives;
		}

		__global__ void GaussianND_CUDA(double** result, double *TP, dim3 dTP, double *CN, dim3 dCN, double *A, dim3 dA, double *C, dim3 dC)
		{
			int i = blockDim.y * blockIdx.y + threadIdx.y;
			int j = blockDim.x * blockIdx.x + threadIdx.x;
			__syncthreads();
			if (i == 0 & j == 0)
			{
				printf("start mqd2_CUDA i=%i j =%i \r\n", i, j);
				double* D = (double *)malloc(sizeof(double) * dTP.x * dCN.y);
				double* Dt = (double *)malloc(sizeof(double) * dTP.x *dCN.y);
				double* Dx = (double *)malloc(sizeof(double) * dTP.x *dCN.y);
				double* Dxx = (double *)malloc(sizeof(double) * dTP.x *dCN.y);
				printf("allocated arrays mqd2_CUDA i=%i j =%i \r\n", i, j);


				dim3 threads(32, 32);
				//dim3 grid(CNx / threads.x, CNy / threads.y);
				dim3 grid(1, 1);
				dim3 dimTP(dTP.x, dTP.y);
				dim3 dimCN(dCN.x, dCN.y);
				dim3 dimA(dA.x, dA.y);
				dim3 dimC(dC.x, dC.y);
				//printf("TP size=%f", sizeof(TP));
				//printMatrix_CUDA << < dim3(1, 1), dim3(1, 1) >> > (TP, dimTP);
				//gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
				//gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());

				printf("dimTPx=%i dimTPy=%i dimCNx=%i dimCNy=%i dimAx=%i dimAy=%i dimCx=%i dimCy=%i\r\n", dimTP.x, dimTP.y, dimCN.x, dimCN.y, dimA.x, dimA.y, dimC.x, dimC.y);
				Gaussian2d2_CUDA << <1, 1 >> > (D, Dt, Dx, Dxx, TP, dimTP, CN, dimCN, A, dimA, C, dimC);
				gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);
				gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);

				//printf("D size=%f", sizeof(D));
				//printMatrix_CUDA << < dim3(1, 1), dim3(1, 1) >> > (D, dim3(dTP.y, dTP.y));
				//gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
				//gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());

				//__syncthreads();
				result[0] = D;
				result[1] = Dt;
				result[2] = Dx;
				result[3] = Dxx;
			}
			__syncthreads();
			//printf("end mqd2_CUDA");
		}


		vector<MatrixXd> CudaRBF::Gaussian2D(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C)
		{
			gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);

			//Allocate the input on host and device
			const double *a, *c, *tx, *cn;
			a = A.data();
			c = C.data();
			tx = TP.data();
			cn = CN.data();
			double *d_a, *d_c, *d_tx, *d_cn;

			//size_t *pValue;

			//cudaDeviceGetLimit(pValue, cudaLimit::cudaLimitMallocHeapSize);
			//printf("Heap limit=%i\r\n", &pValue);

			cudaError_t e = cudaMalloc((void **)&d_a, A.rows() * A.cols() * sizeof(double));
			if (e != cudaSuccess)
				printf("cudaMalloc d_a returned error %s (code %d), line(%d) when allocating %i bytes\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * A.rows() * A.cols());
			e = cudaMalloc((void **)&d_c, C.rows() * C.cols() * sizeof(double));
			if (e != cudaSuccess)
				printf("cudaMalloc d_c returned error %s (code %d), line(%d) when allocating %i bytes\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * C.rows() * C.cols());
			e = cudaMalloc((void **)&d_tx, TP.rows() * TP.cols() * sizeof(double));
			if (e != cudaSuccess)
				printf("cudaMalloc d_tx returned error %s (code %d), line(%d) when allocating %i bytes\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * TP.rows() * TP.cols());
			e = cudaMalloc((void **)&d_cn, CN.rows() * CN.cols() * sizeof(double));
			if (e != cudaSuccess)
				printf("cudaMalloc d_tx1 returned error %s (code %d), line(%d) when allocating %i bytes\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * CN.rows() * CN.cols());

			e = cudaMemcpy(d_a, a, sizeof(double) * A.rows() * A.cols(), cudaMemcpyKind::cudaMemcpyHostToDevice);
			if (e != cudaSuccess)
				printf("cudaMemcpy d_a returned error %s (code %d), line(%d) when copying %i bytes\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * A.rows() * A.cols());
			e = cudaMemcpy(d_c, c, sizeof(double) * C.rows() * C.cols(), cudaMemcpyKind::cudaMemcpyHostToDevice);
			if (e != cudaSuccess)
				printf("cudaMemcpy d_c returned error %s (code %d), line(%d) when copying %i bytes\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * C.rows() * C.cols());
			e = cudaMemcpy(d_tx, tx, sizeof(double) * TP.rows() * TP.cols(), cudaMemcpyKind::cudaMemcpyHostToDevice);
			if (e != cudaSuccess)
				printf("cudaMemcpy d_tx returned error %s (code %d), line(%d) when copying %i bytes\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * TP.rows() * TP.cols());
			e = cudaMemcpy(d_cn, tx, sizeof(double) * CN.rows() * CN.cols(), cudaMemcpyKind::cudaMemcpyHostToDevice);
			if (e != cudaSuccess)
				printf("cudaMemcpy d_tx1 returned error %s (code %d), line(%d) when copying %i bytes\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * CN.rows() * CN.cols());

			//Allocate the output on host and device
			double *d_FAI, *d_D, *d_Dt, *d_Dx, *d_Dxx;
			double *h_FAI, *h_D, *h_Dt, *h_Dx, *h_Dxx;

			int rows = CN.rows();
			int cols = TP.rows();
			//h_FAI = new double[rows * cols];
			h_D = (double*)malloc(sizeof(double) * rows * cols);
			h_Dt = (double*)malloc(sizeof(double) * rows * cols);
			h_Dx = (double*)malloc(sizeof(double) * rows * cols);
			h_Dxx = (double*)malloc(sizeof(double) * rows * cols);

			/*e = cudaMalloc((void **)&d_FAI, rows * cols * sizeof(double));
			if (e != cudaSuccess)
				printf("cudaMalloc d_FAI returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);*/
			e = cudaMalloc((void **)&d_D, rows * cols * sizeof(double));
			if (e != cudaSuccess)
				printf("cudaMalloc d_D returned error %s (code %d), line(%d) when allocating %i bytes\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			e = cudaMalloc((void **)&d_Dt, rows * cols * sizeof(double));
			if (e != cudaSuccess)
				printf("cudaMalloc d_Dt returned error %s (code %d), line(%d) when allocating %i bytes\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			e = cudaMalloc((void **)&d_Dx, rows * cols * sizeof(double));
			if (e != cudaSuccess)
				printf("cudaMalloc d_Dx returned error %s (code %d), line(%d) when allocating %i bytes\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			e = cudaMalloc((void **)&d_Dxx, rows * cols * sizeof(double));
			if (e != cudaSuccess)
				printf("cudaMalloc d_Dxx returned error %s (code %d), line(%d) when allocating %i bytes\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);

			cudaDeviceProp deviceProp;

			checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
			int block_size = (deviceProp.major < 2) ? 16 : 32;
			dim3 dimTx(TP.cols(), TP.rows());
			dim3 dimA(A.cols(), A.rows());
			dim3 dimC(C.cols(), C.rows());
			dim3 threads(block_size, block_size);
			//dim3 grid(dimTx.x / threads.x, dimTx.y / threads.y);
			dim3 grid(1, 1);
			//test << < grid, threads >> > ();
			//cudaDeviceSynchronize();
			//printMatrix(tx, dimTx);
			Gaussian2d_CUDA << < grid, threads >> > (d_D, d_Dt, d_Dx, d_Dxx, d_tx, dimTx.x, dimTx.y, d_cn, dimTx.x, dimTx.y, d_a, dimA.x, dimA.y, d_c, dimC.x, dimC.y);
			//mqd2_CUDA<32>(d_result, d_tx, dimTx.x, dimTx.y, d_tx1, dimTx.x, dimTx.y, d_a, dimA.x, dimA.y, d_c, dimC.x, dimC.y);

			gpuAssert << <1, 1 >> > (cudaThreadSynchronize(), __FILE__, __LINE__);
			gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);

			gpuAssert << <1, 1 >> > (cudaPeekAtLastError(), __FILE__, __LINE__);

			e = cudaMemcpy(h_D, d_D, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (e != cudaSuccess)
				printf("cudaMemcpy d_D returned error %s (code %d), line(%d) when copying%i\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			e = cudaMemcpy(h_Dt, d_Dt, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (e != cudaSuccess)
				printf("cudaMemcpy d_Dt returned error %s (code %d), line(%d) when copying%i\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			e = cudaMemcpy(h_Dx, d_Dx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (e != cudaSuccess)
				printf("cudaMemcpy d_Dx returned error %s (code %d), line(%d) when copying%i\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			e = cudaMemcpy(h_Dxx, d_Dxx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (e != cudaSuccess)
				printf("cudaMemcpy d_Dxx returned error %s (code %d), line(%d) when copying%i\n", cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);

			//gpuAssert << <1, 1 >> > (cudaThreadSynchronize());
			gpuAssert << <1, 1 >> > (cudaDeviceSynchronize(), __FILE__, __LINE__);

			//printMatrix(h_D, dimTx);
			MatrixXd D(rows, cols);
			Eigen::Map<Eigen::MatrixXd> dataMapD(h_D, rows, cols);
			D = dataMapD.eval();

			MatrixXd Dt(rows, cols);
			Eigen::Map<Eigen::MatrixXd> dataMapDt(h_Dt, rows, cols);
			Dt = dataMapDt.eval();

			//	printMatrix(h_Dx, dim3(15, 15));
			MatrixXd Dx(rows, cols);
			Eigen::Map<Eigen::MatrixXd> dataMapDx(h_Dx, rows, cols);
			Dx = dataMapDx.eval();

			MatrixXd Dxx(rows, cols);
			Eigen::Map<Eigen::MatrixXd> dataMapDxx(h_Dxx, rows, cols);
			Dxx = dataMapDxx.eval();

			free(h_D);
			free(h_Dt);
			free(h_Dx);
			free(h_Dxx);
			cudaFree(d_D);
			cudaFree(d_Dt);
			cudaFree(d_Dx);
			cudaFree(d_Dxx);
			cudaFree(d_a);
			cudaFree(d_c);
			cudaFree(d_tx);
			cudaFree(d_cn);
			//cudaDeviceReset();
			return { D, Dt, Dx, Dxx };
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


		//int MethodOfLines::MoLiteration(double Tend, double Tdone, double dt, double *G, int GRows, int GCols, double *lamb, int lambRows, int lambCols, double inx2, double r, double K, MatrixXd A1, MatrixXd Aend, MatrixXd H)
		//{
		//	int count = 0;
		//	while (Tend - Tdone > 1E-8)
		//	{
		//		Tdone += dt;
		//		
		//		int sizeG = GRows * GCols;
		//		int sizeLamb = lambRows * lambCols;
		//		int memG = sizeof(double) * sizeG;
		//		int memLamb = sizeof(double) * sizeLamb;
		//
		//		double *d_G, *d_lamb, *d_FFF;
		//		int sizeFFF = GRows * lambCols;
		//		int memFFF = sizeof(double)* sizeFFF;
		//
		//		double *h_FFF = (double *)malloc(memFFF);
		//		double *h_CUBLAS = (double *)malloc(memFFF);
		//
		//		checkCudaErrors(cudaMalloc((void **)&d_G, memG));
		//		checkCudaErrors(cudaMalloc((void **)&d_lamb, memLamb));
		//		checkCudaErrors(cudaMemcpy(d_G, G, memG, cudaMemcpyHostToDevice));
		//		checkCudaErrors(cudaMemcpy(d_lamb, lamb, memLamb, cudaMemcpyHostToDevice));
		//		checkCudaErrors(cudaMalloc((void **)&d_FFF, memFFF));
		//
		//		cublasHandle_t handle;
		//		checkCudaErrors(cublasCreate(&handle));
		//		const double alpha = 1.0;
		//		const double beta = 1.0;
		//		checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, GRows, lambCols, GCols, &alpha, d_G, GRows, d_lamb, lambRows, &beta, d_FFF, GRows));
		//		
		//		checkCudaErrors(cudaMemcpy(h_FFF, d_FFF, memFFF, cudaMemcpyDeviceToHost));
		//		printf("after cublasDgemm:\r\n");
		//		//double i[] = h_FFF;
		//		VectorXd FFF = Map<VectorXd >(h_FFF, GRows, lambCols);
		//		VectorXd fff = PushAndQueue(0, FFF, inx2 - exp(-r*Tdone)*K);
		//		printf("after PushAndQueue:\r\n");
		//		MatrixXd HH(A1.cols(), A1.cols());
		//		HH.row(0) = A1;
		//		HH.middleRows(1, HH.rows() - 2) = H;
		//		HH.row(HH.rows() - 1) = Aend;
		//		printf("after HH construction:\r\n");
		//		//LLT<MatrixXd> lltOfA(HH);
		//		//lamb = lltOfA.solve(fff);
		//
		//		cusolverDnHandle_t cusolverH = NULL; 
		//		cublasHandle_t cublasH = NULL; 
		//		cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS; 
		//		cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS; 
		//		cudaError_t cudaStat1 = cudaSuccess; 
		//		cudaError_t cudaStat2 = cudaSuccess; 
		//		cudaError_t cudaStat3 = cudaSuccess; 
		//		cudaError_t cudaStat4 = cudaSuccess; 
		//		const int m = HH.rows(); const int lda = m; const int ldb = m; const int nrhs = 1; // number of right hand side vectors
		//		double *XC = new double[ldb*nrhs];
		//		
		//		double *d_A = NULL; // linear memory of GPU 
		//		double *d_tau = NULL; // linear memory of GPU 
		//		double *d_B = NULL; int *devInfo = NULL; // info in gpu (device copy) 
		//		double *d_work = NULL; 
		//		int lwork = 0; 
		//		int info_gpu = 0; 
		//		const double one = 1;
		//
		//		cusolver_status = cusolverDnCreate(&cusolverH); 
		//		assert(CUSOLVER_STATUS_SUCCESS == cusolver_status); 
		//		printf("after cusolver create:\r\n");
		//		cublas_status = cublasCreate(&cublasH); 
		//		assert(CUBLAS_STATUS_SUCCESS == cublas_status);
		//		printf("after cublas create:\r\n");
		//
		//		cudaStat1 = cudaMalloc((void**)&d_A, sizeof(double) * lda * m); 
		//		cudaStat2 = cudaMalloc((void**)&d_tau, sizeof(double) * m); 
		//		cudaStat3 = cudaMalloc((void**)&d_B, sizeof(double) * ldb * nrhs); 
		//		cudaStat4 = cudaMalloc((void**)&devInfo, sizeof(int)); 
		//		assert(cudaSuccess == cudaStat1); 
		//		assert(cudaSuccess == cudaStat2); 
		//		assert(cudaSuccess == cudaStat3); 
		//		assert(cudaSuccess == cudaStat4); 
		//		cudaStat1 = cudaMemcpy(d_A, HH.data(), sizeof(double) * lda * m, cudaMemcpyHostToDevice); 
		//		cudaStat2 = cudaMemcpy(d_B, fff.data(), sizeof(double) * ldb * nrhs, cudaMemcpyHostToDevice); 
		//		assert(cudaSuccess == cudaStat1); assert(cudaSuccess == cudaStat2);
		//
		//		// step 3: query working space of geqrf and ormqr 
		//		cusolver_status = cusolverDnDgeqrf_bufferSize( cusolverH, m, m, d_A, lda, &lwork); 
		//		assert (cusolver_status == CUSOLVER_STATUS_SUCCESS); 
		//		cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork); 
		//		printf("after initialisation:\r\n");
		//		assert(cudaSuccess == cudaStat1); 
		//		// step 4: compute QR factorization 
		//		cusolver_status = cusolverDnDgeqrf( cusolverH, m, m, d_A, lda, d_tau, d_work, lwork, devInfo); 
		//		cudaStat1 = cudaDeviceSynchronize(); 
		//		assert(CUSOLVER_STATUS_SUCCESS == cusolver_status); 
		//		assert(cudaSuccess == cudaStat1); 
		//		printf("after QR factorization:\r\n");
		//		// check if QR is good or not 
		//		cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost); 
		//		assert(cudaSuccess == cudaStat1); 
		//		printf("after geqrf: info_gpu = %d\n", info_gpu); 
		//		assert(0 == info_gpu); 
		//		// step 5: compute Q^T*B 
		//		cusolver_status= cusolverDnDormqr( cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, m, d_A, lda, d_tau, d_B, ldb, d_work, lwork, devInfo); 
		//		cudaStat1 = cudaDeviceSynchronize(); 
		//		assert(CUSOLVER_STATUS_SUCCESS == cusolver_status); 
		//		assert(cudaSuccess == cudaStat1);
		//
		//		// check if QR is good or not 
		//		cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost); 
		//		assert(cudaSuccess == cudaStat1); 
		//		printf("after ormqr: info_gpu = %d\n", info_gpu); 
		//		assert(0 == info_gpu); 
		//		// step 6: compute x = R \ Q^T*B 
		//		cublas_status = cublasDtrsm( cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, nrhs, &one, d_A, lda, d_B, ldb); 
		//		cudaStat1 = cudaDeviceSynchronize(); assert(CUBLAS_STATUS_SUCCESS == cublas_status); 
		//		assert(cudaSuccess == cudaStat1); 
		//		cudaStat1 = cudaMemcpy(XC, d_B, sizeof(double)*ldb*nrhs, cudaMemcpyDeviceToHost); 
		//		assert(cudaSuccess == cudaStat1); 
		//		
		//		/*printf("X = (matlab base-1)\n"); 
		//		printMatrix(m, nrhs, XC, ldb, "X"); */
		//
		//		// free resources 
		//		if (d_A ) cudaFree(d_A); 
		//		if (d_tau ) cudaFree(d_tau); 
		//		if (d_B ) cudaFree(d_B); 
		//		if (devInfo) cudaFree(devInfo); 
		//		if (d_work ) cudaFree(d_work); 
		//		if (cublasH ) cublasDestroy(cublasH); 
		//		if (cusolverH) cusolverDnDestroy(cusolverH); 
		//		cudaDeviceReset();
		//
		//		
		//		
		//		count++;
		//		printf("%i\r\n", count);
		//	}
		//    return 0;
		//}

	}
}

void SetupLevel2(Eigen::MatrixXd &TX1, Eigen::MatrixXd &CN, Eigen::MatrixXd &A, Eigen::MatrixXd &C)
{
	TX1(15, 2);
	CN(15, 2);
	C(1, 2);
	C << 1.73, 600;
	A(1, 2);
	A << 2,4;

	TX1(0, 0) = 0;
	TX1(1, 0) = 0;
	TX1(2, 0) = 0;
	TX1(3, 0) = 0;
	TX1(4, 0) = 0;
	TX1(5, 0) = 0.432499999999999;
	TX1(6, 0) = 0.432499999999999;
	TX1(7, 0) = 0.432499999999999;
	TX1(8, 0) = 0.432499999999999;
	TX1(9, 0) = 0.432499999999999;
	TX1(10, 0) = 0.864999999999999;
	TX1(11, 0) = 0.864999999999999;
	TX1(12, 0) = 0.864999999999999;
	TX1(13, 0) = 0.864999999999999;
	TX1(14, 0) = 0.864999999999999;
	TX1(0, 1) = 0;
	TX1(1, 1) = 75;
	TX1(2, 1) = 150;
	TX1(3, 1) = 225;
	TX1(4, 1) = 300;
	TX1(5, 1) = 0;
	TX1(6, 1) = 75;
	TX1(7, 1) = 150;
	TX1(8, 1) = 225;
	TX1(9, 1) = 300;
	TX1(10, 1) = 0;
	TX1(11, 1) = 75;
	TX1(12, 1) = 150;
	TX1(13, 1) = 225;
	TX1(14, 1) = 300;
	CN(0, 0) = 0;
	CN(1, 0) = 0;
	CN(2, 0) = 0;
	CN(3, 0) = 0;
	CN(4, 0) = 0;
	CN(5, 0) = 0.432499999999999;
	CN(6, 0) = 0.432499999999999;
	CN(7, 0) = 0.432499999999999;
	CN(8, 0) = 0.432499999999999;
	CN(9, 0) = 0.432499999999999;
	CN(10, 0) = 0.864999999999999;
	CN(11, 0) = 0.864999999999999;
	CN(12, 0) = 0.864999999999999;
	CN(13, 0) = 0.864999999999999;
	CN(14, 0) = 0.864999999999999;
	CN(0, 1) = 0;
	CN(1, 1) = 75;
	CN(2, 1) = 150;
	CN(3, 1) = 225;
	CN(4, 1) = 300;
	CN(5, 1) = 0;
	CN(6, 1) = 75;
	CN(7, 1) = 150;
	CN(8, 1) = 225;
	CN(9, 1) = 300;
	CN(10, 1) = 0;
	CN(11, 1) = 75;
	CN(12, 1) = 150;
	CN(13, 1) = 225;
	CN(14, 1) = 300;

}

MatrixXd GetTX7()
{
	MatrixXd TX1(325, 2);
	TX1(0, 0) = 0;
	TX1(1, 0) = 0;
	TX1(2, 0) = 0;
	TX1(3, 0) = 0;
	TX1(4, 0) = 0;
	TX1(5, 0) = 0;
	TX1(6, 0) = 0;
	TX1(7, 0) = 0;
	TX1(8, 0) = 0;
	TX1(9, 0) = 0;
	TX1(10, 0) = 0;
	TX1(11, 0) = 0;
	TX1(12, 0) = 0;
	TX1(13, 0) = 0;
	TX1(14, 0) = 0;
	TX1(15, 0) = 0;
	TX1(16, 0) = 0;
	TX1(17, 0) = 0;
	TX1(18, 0) = 0;
	TX1(19, 0) = 0;
	TX1(20, 0) = 0;
	TX1(21, 0) = 0;
	TX1(22, 0) = 0;
	TX1(23, 0) = 0;
	TX1(24, 0) = 0;
	TX1(25, 0) = 0;
	TX1(26, 0) = 0;
	TX1(27, 0) = 0;
	TX1(28, 0) = 0;
	TX1(29, 0) = 0;
	TX1(30, 0) = 0;
	TX1(31, 0) = 0;
	TX1(32, 0) = 0;
	TX1(33, 0) = 0;
	TX1(34, 0) = 0;
	TX1(35, 0) = 0;
	TX1(36, 0) = 0;
	TX1(37, 0) = 0;
	TX1(38, 0) = 0;
	TX1(39, 0) = 0;
	TX1(40, 0) = 0;
	TX1(41, 0) = 0;
	TX1(42, 0) = 0;
	TX1(43, 0) = 0;
	TX1(44, 0) = 0;
	TX1(45, 0) = 0;
	TX1(46, 0) = 0;
	TX1(47, 0) = 0;
	TX1(48, 0) = 0;
	TX1(49, 0) = 0;
	TX1(50, 0) = 0;
	TX1(51, 0) = 0;
	TX1(52, 0) = 0;
	TX1(53, 0) = 0;
	TX1(54, 0) = 0;
	TX1(55, 0) = 0;
	TX1(56, 0) = 0;
	TX1(57, 0) = 0;
	TX1(58, 0) = 0;
	TX1(59, 0) = 0;
	TX1(60, 0) = 0;
	TX1(61, 0) = 0;
	TX1(62, 0) = 0;
	TX1(63, 0) = 0;
	TX1(64, 0) = 0;
	TX1(65, 0) = 0.21625;
	TX1(66, 0) = 0.21625;
	TX1(67, 0) = 0.21625;
	TX1(68, 0) = 0.21625;
	TX1(69, 0) = 0.21625;
	TX1(70, 0) = 0.21625;
	TX1(71, 0) = 0.21625;
	TX1(72, 0) = 0.21625;
	TX1(73, 0) = 0.21625;
	TX1(74, 0) = 0.21625;
	TX1(75, 0) = 0.21625;
	TX1(76, 0) = 0.21625;
	TX1(77, 0) = 0.21625;
	TX1(78, 0) = 0.21625;
	TX1(79, 0) = 0.21625;
	TX1(80, 0) = 0.21625;
	TX1(81, 0) = 0.21625;
	TX1(82, 0) = 0.21625;
	TX1(83, 0) = 0.21625;
	TX1(84, 0) = 0.21625;
	TX1(85, 0) = 0.21625;
	TX1(86, 0) = 0.21625;
	TX1(87, 0) = 0.21625;
	TX1(88, 0) = 0.21625;
	TX1(89, 0) = 0.21625;
	TX1(90, 0) = 0.21625;
	TX1(91, 0) = 0.21625;
	TX1(92, 0) = 0.21625;
	TX1(93, 0) = 0.21625;
	TX1(94, 0) = 0.21625;
	TX1(95, 0) = 0.21625;
	TX1(96, 0) = 0.21625;
	TX1(97, 0) = 0.21625;
	TX1(98, 0) = 0.21625;
	TX1(99, 0) = 0.21625;
	TX1(100, 0) = 0.21625;
	TX1(101, 0) = 0.21625;
	TX1(102, 0) = 0.21625;
	TX1(103, 0) = 0.21625;
	TX1(104, 0) = 0.21625;
	TX1(105, 0) = 0.21625;
	TX1(106, 0) = 0.21625;
	TX1(107, 0) = 0.21625;
	TX1(108, 0) = 0.21625;
	TX1(109, 0) = 0.21625;
	TX1(110, 0) = 0.21625;
	TX1(111, 0) = 0.21625;
	TX1(112, 0) = 0.21625;
	TX1(113, 0) = 0.21625;
	TX1(114, 0) = 0.21625;
	TX1(115, 0) = 0.21625;
	TX1(116, 0) = 0.21625;
	TX1(117, 0) = 0.21625;
	TX1(118, 0) = 0.21625;
	TX1(119, 0) = 0.21625;
	TX1(120, 0) = 0.21625;
	TX1(121, 0) = 0.21625;
	TX1(122, 0) = 0.21625;
	TX1(123, 0) = 0.21625;
	TX1(124, 0) = 0.21625;
	TX1(125, 0) = 0.21625;
	TX1(126, 0) = 0.21625;
	TX1(127, 0) = 0.21625;
	TX1(128, 0) = 0.21625;
	TX1(129, 0) = 0.21625;
	TX1(130, 0) = 0.432499999999999;
	TX1(131, 0) = 0.432499999999999;
	TX1(132, 0) = 0.432499999999999;
	TX1(133, 0) = 0.432499999999999;
	TX1(134, 0) = 0.432499999999999;
	TX1(135, 0) = 0.432499999999999;
	TX1(136, 0) = 0.432499999999999;
	TX1(137, 0) = 0.432499999999999;
	TX1(138, 0) = 0.432499999999999;
	TX1(139, 0) = 0.432499999999999;
	TX1(140, 0) = 0.432499999999999;
	TX1(141, 0) = 0.432499999999999;
	TX1(142, 0) = 0.432499999999999;
	TX1(143, 0) = 0.432499999999999;
	TX1(144, 0) = 0.432499999999999;
	TX1(145, 0) = 0.432499999999999;
	TX1(146, 0) = 0.432499999999999;
	TX1(147, 0) = 0.432499999999999;
	TX1(148, 0) = 0.432499999999999;
	TX1(149, 0) = 0.432499999999999;
	TX1(150, 0) = 0.432499999999999;
	TX1(151, 0) = 0.432499999999999;
	TX1(152, 0) = 0.432499999999999;
	TX1(153, 0) = 0.432499999999999;
	TX1(154, 0) = 0.432499999999999;
	TX1(155, 0) = 0.432499999999999;
	TX1(156, 0) = 0.432499999999999;
	TX1(157, 0) = 0.432499999999999;
	TX1(158, 0) = 0.432499999999999;
	TX1(159, 0) = 0.432499999999999;
	TX1(160, 0) = 0.432499999999999;
	TX1(161, 0) = 0.432499999999999;
	TX1(162, 0) = 0.432499999999999;
	TX1(163, 0) = 0.432499999999999;
	TX1(164, 0) = 0.432499999999999;
	TX1(165, 0) = 0.432499999999999;
	TX1(166, 0) = 0.432499999999999;
	TX1(167, 0) = 0.432499999999999;
	TX1(168, 0) = 0.432499999999999;
	TX1(169, 0) = 0.432499999999999;
	TX1(170, 0) = 0.432499999999999;
	TX1(171, 0) = 0.432499999999999;
	TX1(172, 0) = 0.432499999999999;
	TX1(173, 0) = 0.432499999999999;
	TX1(174, 0) = 0.432499999999999;
	TX1(175, 0) = 0.432499999999999;
	TX1(176, 0) = 0.432499999999999;
	TX1(177, 0) = 0.432499999999999;
	TX1(178, 0) = 0.432499999999999;
	TX1(179, 0) = 0.432499999999999;
	TX1(180, 0) = 0.432499999999999;
	TX1(181, 0) = 0.432499999999999;
	TX1(182, 0) = 0.432499999999999;
	TX1(183, 0) = 0.432499999999999;
	TX1(184, 0) = 0.432499999999999;
	TX1(185, 0) = 0.432499999999999;
	TX1(186, 0) = 0.432499999999999;
	TX1(187, 0) = 0.432499999999999;
	TX1(188, 0) = 0.432499999999999;
	TX1(189, 0) = 0.432499999999999;
	TX1(190, 0) = 0.432499999999999;
	TX1(191, 0) = 0.432499999999999;
	TX1(192, 0) = 0.432499999999999;
	TX1(193, 0) = 0.432499999999999;
	TX1(194, 0) = 0.432499999999999;
	TX1(195, 0) = 0.648749999999999;
	TX1(196, 0) = 0.648749999999999;
	TX1(197, 0) = 0.648749999999999;
	TX1(198, 0) = 0.648749999999999;
	TX1(199, 0) = 0.648749999999999;
	TX1(200, 0) = 0.648749999999999;
	TX1(201, 0) = 0.648749999999999;
	TX1(202, 0) = 0.648749999999999;
	TX1(203, 0) = 0.648749999999999;
	TX1(204, 0) = 0.648749999999999;
	TX1(205, 0) = 0.648749999999999;
	TX1(206, 0) = 0.648749999999999;
	TX1(207, 0) = 0.648749999999999;
	TX1(208, 0) = 0.648749999999999;
	TX1(209, 0) = 0.648749999999999;
	TX1(210, 0) = 0.648749999999999;
	TX1(211, 0) = 0.648749999999999;
	TX1(212, 0) = 0.648749999999999;
	TX1(213, 0) = 0.648749999999999;
	TX1(214, 0) = 0.648749999999999;
	TX1(215, 0) = 0.648749999999999;
	TX1(216, 0) = 0.648749999999999;
	TX1(217, 0) = 0.648749999999999;
	TX1(218, 0) = 0.648749999999999;
	TX1(219, 0) = 0.648749999999999;
	TX1(220, 0) = 0.648749999999999;
	TX1(221, 0) = 0.648749999999999;
	TX1(222, 0) = 0.648749999999999;
	TX1(223, 0) = 0.648749999999999;
	TX1(224, 0) = 0.648749999999999;
	TX1(225, 0) = 0.648749999999999;
	TX1(226, 0) = 0.648749999999999;
	TX1(227, 0) = 0.648749999999999;
	TX1(228, 0) = 0.648749999999999;
	TX1(229, 0) = 0.648749999999999;
	TX1(230, 0) = 0.648749999999999;
	TX1(231, 0) = 0.648749999999999;
	TX1(232, 0) = 0.648749999999999;
	TX1(233, 0) = 0.648749999999999;
	TX1(234, 0) = 0.648749999999999;
	TX1(235, 0) = 0.648749999999999;
	TX1(236, 0) = 0.648749999999999;
	TX1(237, 0) = 0.648749999999999;
	TX1(238, 0) = 0.648749999999999;
	TX1(239, 0) = 0.648749999999999;
	TX1(240, 0) = 0.648749999999999;
	TX1(241, 0) = 0.648749999999999;
	TX1(242, 0) = 0.648749999999999;
	TX1(243, 0) = 0.648749999999999;
	TX1(244, 0) = 0.648749999999999;
	TX1(245, 0) = 0.648749999999999;
	TX1(246, 0) = 0.648749999999999;
	TX1(247, 0) = 0.648749999999999;
	TX1(248, 0) = 0.648749999999999;
	TX1(249, 0) = 0.648749999999999;
	TX1(250, 0) = 0.648749999999999;
	TX1(251, 0) = 0.648749999999999;
	TX1(252, 0) = 0.648749999999999;
	TX1(253, 0) = 0.648749999999999;
	TX1(254, 0) = 0.648749999999999;
	TX1(255, 0) = 0.648749999999999;
	TX1(256, 0) = 0.648749999999999;
	TX1(257, 0) = 0.648749999999999;
	TX1(258, 0) = 0.648749999999999;
	TX1(259, 0) = 0.648749999999999;
	TX1(260, 0) = 0.864999999999999;
	TX1(261, 0) = 0.864999999999999;
	TX1(262, 0) = 0.864999999999999;
	TX1(263, 0) = 0.864999999999999;
	TX1(264, 0) = 0.864999999999999;
	TX1(265, 0) = 0.864999999999999;
	TX1(266, 0) = 0.864999999999999;
	TX1(267, 0) = 0.864999999999999;
	TX1(268, 0) = 0.864999999999999;
	TX1(269, 0) = 0.864999999999999;
	TX1(270, 0) = 0.864999999999999;
	TX1(271, 0) = 0.864999999999999;
	TX1(272, 0) = 0.864999999999999;
	TX1(273, 0) = 0.864999999999999;
	TX1(274, 0) = 0.864999999999999;
	TX1(275, 0) = 0.864999999999999;
	TX1(276, 0) = 0.864999999999999;
	TX1(277, 0) = 0.864999999999999;
	TX1(278, 0) = 0.864999999999999;
	TX1(279, 0) = 0.864999999999999;
	TX1(280, 0) = 0.864999999999999;
	TX1(281, 0) = 0.864999999999999;
	TX1(282, 0) = 0.864999999999999;
	TX1(283, 0) = 0.864999999999999;
	TX1(284, 0) = 0.864999999999999;
	TX1(285, 0) = 0.864999999999999;
	TX1(286, 0) = 0.864999999999999;
	TX1(287, 0) = 0.864999999999999;
	TX1(288, 0) = 0.864999999999999;
	TX1(289, 0) = 0.864999999999999;
	TX1(290, 0) = 0.864999999999999;
	TX1(291, 0) = 0.864999999999999;
	TX1(292, 0) = 0.864999999999999;
	TX1(293, 0) = 0.864999999999999;
	TX1(294, 0) = 0.864999999999999;
	TX1(295, 0) = 0.864999999999999;
	TX1(296, 0) = 0.864999999999999;
	TX1(297, 0) = 0.864999999999999;
	TX1(298, 0) = 0.864999999999999;
	TX1(299, 0) = 0.864999999999999;
	TX1(300, 0) = 0.864999999999999;
	TX1(301, 0) = 0.864999999999999;
	TX1(302, 0) = 0.864999999999999;
	TX1(303, 0) = 0.864999999999999;
	TX1(304, 0) = 0.864999999999999;
	TX1(305, 0) = 0.864999999999999;
	TX1(306, 0) = 0.864999999999999;
	TX1(307, 0) = 0.864999999999999;
	TX1(308, 0) = 0.864999999999999;
	TX1(309, 0) = 0.864999999999999;
	TX1(310, 0) = 0.864999999999999;
	TX1(311, 0) = 0.864999999999999;
	TX1(312, 0) = 0.864999999999999;
	TX1(313, 0) = 0.864999999999999;
	TX1(314, 0) = 0.864999999999999;
	TX1(315, 0) = 0.864999999999999;
	TX1(316, 0) = 0.864999999999999;
	TX1(317, 0) = 0.864999999999999;
	TX1(318, 0) = 0.864999999999999;
	TX1(319, 0) = 0.864999999999999;
	TX1(320, 0) = 0.864999999999999;
	TX1(321, 0) = 0.864999999999999;
	TX1(322, 0) = 0.864999999999999;
	TX1(323, 0) = 0.864999999999999;
	TX1(324, 0) = 0.864999999999999;
	TX1(0, 1) = 0;
	TX1(1, 1) = 4.6875;
	TX1(2, 1) = 9.375;
	TX1(3, 1) = 14.0625;
	TX1(4, 1) = 18.75;
	TX1(5, 1) = 23.4375;
	TX1(6, 1) = 28.125;
	TX1(7, 1) = 32.8125;
	TX1(8, 1) = 37.5;
	TX1(9, 1) = 42.1875;
	TX1(10, 1) = 46.875;
	TX1(11, 1) = 51.5625;
	TX1(12, 1) = 56.25;
	TX1(13, 1) = 60.9375;
	TX1(14, 1) = 65.625;
	TX1(15, 1) = 70.3125;
	TX1(16, 1) = 75;
	TX1(17, 1) = 79.6875;
	TX1(18, 1) = 84.375;
	TX1(19, 1) = 89.0625;
	TX1(20, 1) = 93.75;
	TX1(21, 1) = 98.4375;
	TX1(22, 1) = 103.125;
	TX1(23, 1) = 107.8125;
	TX1(24, 1) = 112.5;
	TX1(25, 1) = 117.1875;
	TX1(26, 1) = 121.875;
	TX1(27, 1) = 126.5625;
	TX1(28, 1) = 131.25;
	TX1(29, 1) = 135.9375;
	TX1(30, 1) = 140.625;
	TX1(31, 1) = 145.3125;
	TX1(32, 1) = 150;
	TX1(33, 1) = 154.6875;
	TX1(34, 1) = 159.375;
	TX1(35, 1) = 164.0625;
	TX1(36, 1) = 168.75;
	TX1(37, 1) = 173.4375;
	TX1(38, 1) = 178.125;
	TX1(39, 1) = 182.8125;
	TX1(40, 1) = 187.5;
	TX1(41, 1) = 192.1875;
	TX1(42, 1) = 196.875;
	TX1(43, 1) = 201.5625;
	TX1(44, 1) = 206.25;
	TX1(45, 1) = 210.9375;
	TX1(46, 1) = 215.625;
	TX1(47, 1) = 220.3125;
	TX1(48, 1) = 225;
	TX1(49, 1) = 229.6875;
	TX1(50, 1) = 234.375;
	TX1(51, 1) = 239.0625;
	TX1(52, 1) = 243.75;
	TX1(53, 1) = 248.4375;
	TX1(54, 1) = 253.125;
	TX1(55, 1) = 257.8125;
	TX1(56, 1) = 262.5;
	TX1(57, 1) = 267.1875;
	TX1(58, 1) = 271.875;
	TX1(59, 1) = 276.5625;
	TX1(60, 1) = 281.25;
	TX1(61, 1) = 285.9375;
	TX1(62, 1) = 290.625;
	TX1(63, 1) = 295.3125;
	TX1(64, 1) = 300;
	TX1(65, 1) = 0;
	TX1(66, 1) = 4.6875;
	TX1(67, 1) = 9.375;
	TX1(68, 1) = 14.0625;
	TX1(69, 1) = 18.75;
	TX1(70, 1) = 23.4375;
	TX1(71, 1) = 28.125;
	TX1(72, 1) = 32.8125;
	TX1(73, 1) = 37.5;
	TX1(74, 1) = 42.1875;
	TX1(75, 1) = 46.875;
	TX1(76, 1) = 51.5625;
	TX1(77, 1) = 56.25;
	TX1(78, 1) = 60.9375;
	TX1(79, 1) = 65.625;
	TX1(80, 1) = 70.3125;
	TX1(81, 1) = 75;
	TX1(82, 1) = 79.6875;
	TX1(83, 1) = 84.375;
	TX1(84, 1) = 89.0625;
	TX1(85, 1) = 93.75;
	TX1(86, 1) = 98.4375;
	TX1(87, 1) = 103.125;
	TX1(88, 1) = 107.8125;
	TX1(89, 1) = 112.5;
	TX1(90, 1) = 117.1875;
	TX1(91, 1) = 121.875;
	TX1(92, 1) = 126.5625;
	TX1(93, 1) = 131.25;
	TX1(94, 1) = 135.9375;
	TX1(95, 1) = 140.625;
	TX1(96, 1) = 145.3125;
	TX1(97, 1) = 150;
	TX1(98, 1) = 154.6875;
	TX1(99, 1) = 159.375;
	TX1(100, 1) = 164.0625;
	TX1(101, 1) = 168.75;
	TX1(102, 1) = 173.4375;
	TX1(103, 1) = 178.125;
	TX1(104, 1) = 182.8125;
	TX1(105, 1) = 187.5;
	TX1(106, 1) = 192.1875;
	TX1(107, 1) = 196.875;
	TX1(108, 1) = 201.5625;
	TX1(109, 1) = 206.25;
	TX1(110, 1) = 210.9375;
	TX1(111, 1) = 215.625;
	TX1(112, 1) = 220.3125;
	TX1(113, 1) = 225;
	TX1(114, 1) = 229.6875;
	TX1(115, 1) = 234.375;
	TX1(116, 1) = 239.0625;
	TX1(117, 1) = 243.75;
	TX1(118, 1) = 248.4375;
	TX1(119, 1) = 253.125;
	TX1(120, 1) = 257.8125;
	TX1(121, 1) = 262.5;
	TX1(122, 1) = 267.1875;
	TX1(123, 1) = 271.875;
	TX1(124, 1) = 276.5625;
	TX1(125, 1) = 281.25;
	TX1(126, 1) = 285.9375;
	TX1(127, 1) = 290.625;
	TX1(128, 1) = 295.3125;
	TX1(129, 1) = 300;
	TX1(130, 1) = 0;
	TX1(131, 1) = 4.6875;
	TX1(132, 1) = 9.375;
	TX1(133, 1) = 14.0625;
	TX1(134, 1) = 18.75;
	TX1(135, 1) = 23.4375;
	TX1(136, 1) = 28.125;
	TX1(137, 1) = 32.8125;
	TX1(138, 1) = 37.5;
	TX1(139, 1) = 42.1875;
	TX1(140, 1) = 46.875;
	TX1(141, 1) = 51.5625;
	TX1(142, 1) = 56.25;
	TX1(143, 1) = 60.9375;
	TX1(144, 1) = 65.625;
	TX1(145, 1) = 70.3125;
	TX1(146, 1) = 75;
	TX1(147, 1) = 79.6875;
	TX1(148, 1) = 84.375;
	TX1(149, 1) = 89.0625;
	TX1(150, 1) = 93.75;
	TX1(151, 1) = 98.4375;
	TX1(152, 1) = 103.125;
	TX1(153, 1) = 107.8125;
	TX1(154, 1) = 112.5;
	TX1(155, 1) = 117.1875;
	TX1(156, 1) = 121.875;
	TX1(157, 1) = 126.5625;
	TX1(158, 1) = 131.25;
	TX1(159, 1) = 135.9375;
	TX1(160, 1) = 140.625;
	TX1(161, 1) = 145.3125;
	TX1(162, 1) = 150;
	TX1(163, 1) = 154.6875;
	TX1(164, 1) = 159.375;
	TX1(165, 1) = 164.0625;
	TX1(166, 1) = 168.75;
	TX1(167, 1) = 173.4375;
	TX1(168, 1) = 178.125;
	TX1(169, 1) = 182.8125;
	TX1(170, 1) = 187.5;
	TX1(171, 1) = 192.1875;
	TX1(172, 1) = 196.875;
	TX1(173, 1) = 201.5625;
	TX1(174, 1) = 206.25;
	TX1(175, 1) = 210.9375;
	TX1(176, 1) = 215.625;
	TX1(177, 1) = 220.3125;
	TX1(178, 1) = 225;
	TX1(179, 1) = 229.6875;
	TX1(180, 1) = 234.375;
	TX1(181, 1) = 239.0625;
	TX1(182, 1) = 243.75;
	TX1(183, 1) = 248.4375;
	TX1(184, 1) = 253.125;
	TX1(185, 1) = 257.8125;
	TX1(186, 1) = 262.5;
	TX1(187, 1) = 267.1875;
	TX1(188, 1) = 271.875;
	TX1(189, 1) = 276.5625;
	TX1(190, 1) = 281.25;
	TX1(191, 1) = 285.9375;
	TX1(192, 1) = 290.625;
	TX1(193, 1) = 295.3125;
	TX1(194, 1) = 300;
	TX1(195, 1) = 0;
	TX1(196, 1) = 4.6875;
	TX1(197, 1) = 9.375;
	TX1(198, 1) = 14.0625;
	TX1(199, 1) = 18.75;
	TX1(200, 1) = 23.4375;
	TX1(201, 1) = 28.125;
	TX1(202, 1) = 32.8125;
	TX1(203, 1) = 37.5;
	TX1(204, 1) = 42.1875;
	TX1(205, 1) = 46.875;
	TX1(206, 1) = 51.5625;
	TX1(207, 1) = 56.25;
	TX1(208, 1) = 60.9375;
	TX1(209, 1) = 65.625;
	TX1(210, 1) = 70.3125;
	TX1(211, 1) = 75;
	TX1(212, 1) = 79.6875;
	TX1(213, 1) = 84.375;
	TX1(214, 1) = 89.0625;
	TX1(215, 1) = 93.75;
	TX1(216, 1) = 98.4375;
	TX1(217, 1) = 103.125;
	TX1(218, 1) = 107.8125;
	TX1(219, 1) = 112.5;
	TX1(220, 1) = 117.1875;
	TX1(221, 1) = 121.875;
	TX1(222, 1) = 126.5625;
	TX1(223, 1) = 131.25;
	TX1(224, 1) = 135.9375;
	TX1(225, 1) = 140.625;
	TX1(226, 1) = 145.3125;
	TX1(227, 1) = 150;
	TX1(228, 1) = 154.6875;
	TX1(229, 1) = 159.375;
	TX1(230, 1) = 164.0625;
	TX1(231, 1) = 168.75;
	TX1(232, 1) = 173.4375;
	TX1(233, 1) = 178.125;
	TX1(234, 1) = 182.8125;
	TX1(235, 1) = 187.5;
	TX1(236, 1) = 192.1875;
	TX1(237, 1) = 196.875;
	TX1(238, 1) = 201.5625;
	TX1(239, 1) = 206.25;
	TX1(240, 1) = 210.9375;
	TX1(241, 1) = 215.625;
	TX1(242, 1) = 220.3125;
	TX1(243, 1) = 225;
	TX1(244, 1) = 229.6875;
	TX1(245, 1) = 234.375;
	TX1(246, 1) = 239.0625;
	TX1(247, 1) = 243.75;
	TX1(248, 1) = 248.4375;
	TX1(249, 1) = 253.125;
	TX1(250, 1) = 257.8125;
	TX1(251, 1) = 262.5;
	TX1(252, 1) = 267.1875;
	TX1(253, 1) = 271.875;
	TX1(254, 1) = 276.5625;
	TX1(255, 1) = 281.25;
	TX1(256, 1) = 285.9375;
	TX1(257, 1) = 290.625;
	TX1(258, 1) = 295.3125;
	TX1(259, 1) = 300;
	TX1(260, 1) = 0;
	TX1(261, 1) = 4.6875;
	TX1(262, 1) = 9.375;
	TX1(263, 1) = 14.0625;
	TX1(264, 1) = 18.75;
	TX1(265, 1) = 23.4375;
	TX1(266, 1) = 28.125;
	TX1(267, 1) = 32.8125;
	TX1(268, 1) = 37.5;
	TX1(269, 1) = 42.1875;
	TX1(270, 1) = 46.875;
	TX1(271, 1) = 51.5625;
	TX1(272, 1) = 56.25;
	TX1(273, 1) = 60.9375;
	TX1(274, 1) = 65.625;
	TX1(275, 1) = 70.3125;
	TX1(276, 1) = 75;
	TX1(277, 1) = 79.6875;
	TX1(278, 1) = 84.375;
	TX1(279, 1) = 89.0625;
	TX1(280, 1) = 93.75;
	TX1(281, 1) = 98.4375;
	TX1(282, 1) = 103.125;
	TX1(283, 1) = 107.8125;
	TX1(284, 1) = 112.5;
	TX1(285, 1) = 117.1875;
	TX1(286, 1) = 121.875;
	TX1(287, 1) = 126.5625;
	TX1(288, 1) = 131.25;
	TX1(289, 1) = 135.9375;
	TX1(290, 1) = 140.625;
	TX1(291, 1) = 145.3125;
	TX1(292, 1) = 150;
	TX1(293, 1) = 154.6875;
	TX1(294, 1) = 159.375;
	TX1(295, 1) = 164.0625;
	TX1(296, 1) = 168.75;
	TX1(297, 1) = 173.4375;
	TX1(298, 1) = 178.125;
	TX1(299, 1) = 182.8125;
	TX1(300, 1) = 187.5;
	TX1(301, 1) = 192.1875;
	TX1(302, 1) = 196.875;
	TX1(303, 1) = 201.5625;
	TX1(304, 1) = 206.25;
	TX1(305, 1) = 210.9375;
	TX1(306, 1) = 215.625;
	TX1(307, 1) = 220.3125;
	TX1(308, 1) = 225;
	TX1(309, 1) = 229.6875;
	TX1(310, 1) = 234.375;
	TX1(311, 1) = 239.0625;
	TX1(312, 1) = 243.75;
	TX1(313, 1) = 248.4375;
	TX1(314, 1) = 253.125;
	TX1(315, 1) = 257.8125;
	TX1(316, 1) = 262.5;
	TX1(317, 1) = 267.1875;
	TX1(318, 1) = 271.875;
	TX1(319, 1) = 276.5625;
	TX1(320, 1) = 281.25;
	TX1(321, 1) = 285.9375;
	TX1(322, 1) = 290.625;
	TX1(323, 1) = 295.3125;
	TX1(324, 1) = 300;
	return TX1;
}
void SetupLevel7(Eigen::MatrixXd &TX1, Eigen::MatrixXd &CN, Eigen::MatrixXd &A, Eigen::MatrixXd &C)
{
	
	CN(325, 2);
	//C(1, 2);
	//C(0, 1) = 1.73;
	//C(0, 2) = 600;
	//A(1, 2);
	//A(0, 1) = 2;
	//A(0, 2) = 64;
	
}

int main() {

	MatrixXd TX1 = GetTX7();
	MatrixXd CN = GetTX7();
	MatrixXd C(1, 2);
	MatrixXd A(1, 2);
	C << 1.73,600;
	A << 2, 64;
	
	for (int i = 0; i < 1; i++)
	{
		printf("i=%i", i);
		Leicester::CudaLib::CudaRBF::Gaussian2D(TX1, CN, A, C);
	}

	return 0;
}

