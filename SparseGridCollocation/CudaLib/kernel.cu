#include "kernel.h"


using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
using namespace Eigen;

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
		printf("TP size=%f", sizeof(TP));
		printMatrix_CUDA << < dim3(1, 1), dim3(1, 1) >> > (TP, dimTP);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());

		printf("dimTPx=%i dimTPy=%i dimCNx=%i dimCNy=%i dimAx=%i dimAy=%i dimCx=%i dimCy=%i\r\n", dimTP.x, dimTP.y, dimCN.x, dimCN.y, dimA.x, dimA.y, dimC.x, dimC.y);
		Gaussian2d2_CUDA << <1, 1 >> > (D, Dt, Dx, Dxx, TP, dimTP, CN, dimCN, A, dimA, C, dimC);
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());

		printf("D size=%f", sizeof(D));
		printMatrix_CUDA << < dim3(1, 1), dim3(1, 1) >> > (D, dim3(dTP.y, dTP.y));
		gpuErrchk << <1, 1 >> >(cudaPeekAtLastError());
		gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());

		//__syncthreads();
		result[0] = D;
		result[1] = Dt;
		result[2] = Dx;
		result[3] = Dxx;
	}
	__syncthreads();
	//printf("end mqd2_CUDA");
}

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

vector<MatrixXd> CudaRBF::Gaussian2D(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C)
{
	
	const double *a, *c, *tx, *cn;
	a = A.data();
	c = C.data();
	tx = TP.data();
	cn = CN.data();
	double *d_a, *d_c, *d_tx, *d_cn;

	cudaError_t e = cudaMalloc((void **)&d_a, A.rows() * A.cols() * sizeof(double));
	if (e != cudaSuccess)
		printf("cudaMalloc d_a returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);
	e = cudaMalloc((void **)&d_c, C.rows() * C.cols() * sizeof(double));
	if (e != cudaSuccess)
		printf("cudaMalloc d_c returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);
	e = cudaMalloc((void **)&d_tx, TP.rows() * TP.cols() * sizeof(double));
	if (e != cudaSuccess)
		printf("cudaMalloc d_tx returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);
	e = cudaMalloc((void **)&d_cn, CN.rows() * CN.cols() * sizeof(double));
	if (e != cudaSuccess)
		printf("cudaMalloc d_tx1 returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);

	e = cudaMemcpy(d_a, a, sizeof(double) * A.rows() * A.cols(), cudaMemcpyKind::cudaMemcpyHostToDevice);
	if (e != cudaSuccess)
		printf("cudaMemcpy d_a returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);
	e = cudaMemcpy(d_c, c, sizeof(double) * C.rows() * C.cols(), cudaMemcpyKind::cudaMemcpyHostToDevice);
	if (e != cudaSuccess)
		printf("cudaMemcpy d_c returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);
	e = cudaMemcpy(d_tx, tx, sizeof(double) * TP.rows() * TP.cols(), cudaMemcpyKind::cudaMemcpyHostToDevice);
	if (e != cudaSuccess)
		printf("cudaMemcpy d_tx returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);
	e = cudaMemcpy(d_cn, tx, sizeof(double) * CN.rows() * CN.cols(), cudaMemcpyKind::cudaMemcpyHostToDevice);
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
	dim3 dimTx(2, 15);
	dim3 dimA(2, 1);
	dim3 dimC(2, 1);
	dim3 threads(block_size, block_size);
	//dim3 grid(dimTx.x / threads.x, dimTx.y / threads.y);
	dim3 grid(1, 1);
	//test << < grid, threads >> > ();
	//cudaDeviceSynchronize();
	printMatrix(tx, dimTx);
	Gaussian2d_CUDA << < grid, threads >> > (d_result, d_tx, dimTx.x, dimTx.y, d_cn, dimTx.x, dimTx.y, d_a, dimA.x, dimA.y, d_c, dimC.x, dimC.y);
	//mqd2_CUDA<32>(d_result, d_tx, dimTx.x, dimTx.y, d_tx1, dimTx.x, dimTx.y, d_a, dimA.x, dimA.y, d_c, dimC.x, dimC.y);
	gpuErrchk<<<1,1>>>(cudaPeekAtLastError());
	gpuErrchk << <1, 1 >> >(cudaDeviceSynchronize());

	e = cudaMemcpy(d_result, h_result, sizeof(double) * 15 * 15, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	if (e != cudaSuccess)
		printf("cudaMemcpy d_result returned error %s (code %d), line(%d)\n", cudaGetErrorString(e), e, __LINE__);

	return {};
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

