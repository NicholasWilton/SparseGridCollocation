#include "Gaussian2d.h"
#include "NodeRegistry.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
using namespace Eigen;
using namespace thrust;


namespace Leicester
{
	namespace ThrustLib
	{

		//void printMatrix(const double *matrix, dim3 dimMatrix)
		//{
		//	int mSize = sizeof(matrix);

		//	printf("printing matrix data=");
		//	for (int x = 0; x < dimMatrix.x * dimMatrix.y; x++)
		//		printf("%f,", matrix[x]);
		//	printf("\r\n");
		//	printf("rows=%i cols=%i\r\n", dimMatrix.y, dimMatrix.x);

		//	for (int y = 0; y < dimMatrix.y; y++)
		//	{
		//		for (int x = 0; x < dimMatrix.x; x++)
		//		{
		//			int idx = (x * dimMatrix.y) + y;
		//			printf("%.16f ", matrix[idx]);
		//		}
		//		printf("\r\n");
		//	}
		//}

		void CopyToMatrix(MatrixXd &m, double* buffer, dim3 size)
		{
			int ptr = 0;
			for (int i = 0; i < size.x; i++)
				for (int j = 0; j < size.y; j++)
					m(i, j) = buffer[ptr];
		}

		


		Gaussian::Gaussian(MatrixXd testNodes, MatrixXd centralNodes)
		{
			cudaStream_t s1, s2;
			cudaStreamCreate(&s1);
			cudaStreamCreate(&s2);
			
			this->rows = testNodes.rows();
			this->cols = centralNodes.rows();

			int sizeTestNodes = sizeof(double) * testNodes.rows() * testNodes.cols();
			const double *h_testNodes = testNodes.data();
			double *d_testNodes;
			cudaMalloc((void**)&d_testNodes, sizeTestNodes);
			device_ptr<double> dp_testNodes = thrust::device_pointer_cast<double>(d_testNodes);
			cudaMemcpyAsync(d_testNodes, h_testNodes, sizeTestNodes, cudaMemcpyKind::cudaMemcpyHostToDevice, s1);
			
			//device_vector<double> dv_testNodes(dp_testNodes, dp_testNodes + sizeTestNodes);
			//this->testNodes = dv_testNodes;

			//pinnedVector ph_testNodes(h_testNodes, h_testNodes + sizeTestNodes);
			//device_vector<mytype> pd_testNodes(sizeTestNodes);
			//cudaMemcpyAsync(thrust::raw_pointer_cast(pd_testNodes.data()), thrust::raw_pointer_cast(ph_testNodes.data()), 
			//	pd_testNodes.size() * sizeof(mytype), cudaMemcpyHostToDevice, s1);
			//this->testNodes = pd_testNodes;

			int sizeCentralNodes = sizeof(double) * centralNodes.rows() * centralNodes.cols();
			const double *h_centralNodes = centralNodes.data();
			double *d_centralNodes;
			cudaMalloc((void**)&d_centralNodes, sizeCentralNodes);
			device_ptr<double> dp_centralNodes = thrust::device_pointer_cast<double>(d_centralNodes);
			cudaMemcpyAsync(d_centralNodes, h_centralNodes, sizeCentralNodes, cudaMemcpyKind::cudaMemcpyHostToDevice, s2);
			
			//device_vector<double> dv_centralNodes(dp_centralNodes, dp_centralNodes + sizeCentralNodes);
			//this->testNodes = dv_centralNodes;

			/*pinnedVector ph_centralNodes(h_centralNodes, h_centralNodes + sizeCentralNodes);
			device_vector<mytype> pd_centralNodes(sizeCentralNodes);
			cudaMemcpyAsync(thrust::raw_pointer_cast(pd_centralNodes.data()), thrust::raw_pointer_cast(ph_centralNodes.data()),
				pd_centralNodes.size() * sizeof(mytype), cudaMemcpyHostToDevice, s2);
			this->centralNodes = pd_centralNodes;
			*/

			
			//cudaStreamSynchronize(s1);
			//cudaStreamSynchronize(s2);
			cudaDeviceSynchronize();

			//device_vector<double> dv_testNodes(d_testNodes, d_testNodes + sizeTestNodes);
			device_vector<double> dv_testNodes(dp_testNodes, dp_testNodes + (testNodes.rows() * testNodes.cols()));
			this->testNodes = dv_testNodes;

			//device_vector<double> dv_centralNodes(d_centralNodes, d_centralNodes + sizeCentralNodes);
			device_vector<double> dv_centralNodes(dp_centralNodes, dp_centralNodes + (centralNodes.rows() * centralNodes.cols()));
			this->centralNodes = dv_centralNodes;
		}

		Gaussian::Gaussian(MatrixXd testNodes)
		{
			cudaStream_t s1;
			cudaStreamCreate(&s1);
			this->rows = testNodes.rows();
			const double *h_testNodes = testNodes.data();
			//device_vector<double> d_testNodes(h_testNodes, h_testNodes + (testNodes.rows() * testNodes.cols()));
			//this->testNodes = d_testNodes;
			//this->rows = testNodes.rows();
			//const double *h_testNodes = testNodes.data();
			double *d_testNodes;
			int sizeTestNodes = sizeof(double) * testNodes.rows() * testNodes.cols();
			device_ptr<double> dp_testNodes = thrust::device_pointer_cast<double>(d_testNodes);
			cudaError_t e = cudaMemcpyAsync(d_testNodes, h_testNodes, sizeTestNodes, cudaMemcpyKind::cudaMemcpyHostToDevice, s1);
			
			/*pinnedVector ph_testNodes(h_testNodes, h_testNodes + sizeTestNodes);
			device_vector<mytype> pd_testNodes(sizeTestNodes);
			cudaMemcpyAsync(thrust::raw_pointer_cast(pd_testNodes.data()), thrust::raw_pointer_cast(ph_testNodes.data()),
				pd_testNodes.size() * sizeof(mytype), cudaMemcpyHostToDevice, s1);
			*/
			cudaStreamSynchronize(s1);
			cudaDeviceSynchronize();
			device_vector<double> dv_testNodes(dp_testNodes, dp_testNodes + (testNodes.rows() * testNodes.cols()));

			this->testNodes = dv_testNodes;
			
		}
	
		Gaussian::Gaussian(int b, int d, double tLower, double tUpper, double xLower, double xHigher)
		{
			double* N = (double*)malloc(512 * sizeof(double));
			GetN(b, d, N);
			double nRows = N[0];
			double nCols = N[1];

			double** h_nodes = (double**)malloc(nRows * sizeof(double*));
			double** d_nodes;
			cudaMalloc((void **)&d_nodes, nRows * sizeof(double*));

			//double** nodes = (double**)malloc(512 * sizeof(double));
			//thrust::device_ptr<double*> nodes = thrust::device_malloc<double*>(512);
			Leicester::ThrustLib::BuildRegistry << <1, 1 >> >(b, d, tLower, tUpper,xLower, xHigher, d_nodes);
			cudaError_t e = cudaMemcpy(h_nodes, d_nodes, 512 * sizeof(double*), cudaMemcpyDeviceToHost);
			for (int i = 0; i < nRows; i++)
				cudaMemcpy(h_nodes[i], d_nodes[i], sizeof(double), cudaMemcpyDeviceToHost);

			double nNodes = *h_nodes[0];
			
			for (int i = 1; i < nNodes; i++)
			{
				int rows = N[2 + i] * N[2 + i + (int)nRows];
				thrust::device_ptr<double> d_ptr(h_nodes[i]);
				const thrust::device_vector<double> d_v(d_ptr, d_ptr + (rows * 2));//TODO: change 2 to dimensions
				nodesDetails nd;
				nd.rows = rows;
				nd.cols = 2;
				nd.nodes = d_v;
				this->nodeMap[i] = nd;
			}

			free(d_nodes);
			free(h_nodes);
			free(N);
		}

		Gaussian::Gaussian() {}

		Gaussian::~Gaussian()
		{
			this->testNodes.clear();
			this->testNodes.shrink_to_fit();
			this->centralNodes.clear();
			this->centralNodes.shrink_to_fit();
			//cudaStreamDestroy(s1);
			//cudaStreamDestroy(s2);
			//cudaStreamDestroy(s3);
		}

		vector<MatrixXd> Gaussian::Gaussian2d_2(double tLower, double tUpper, double N[], const MatrixXd & A, const MatrixXd & C)
		{
			cudaStream_t s1, s2, s3, s4;
			cudaStreamCreate(&s1);
			cudaStreamCreate(&s2);
			cudaStreamCreate(&s3);
			cudaStreamCreate(&s4);

			double length = N[3] * N[2];
			int sizeTestNodes = 2 * length * sizeof(double);
			double * h_TXYZ = (double*)malloc(sizeTestNodes);
			pinnedVector ph_testNodes(h_TXYZ, h_TXYZ + sizeTestNodes);
			device_vector<double> dv_tp(sizeTestNodes);
			cudaMemcpyAsync(thrust::raw_pointer_cast(dv_tp.data()), thrust::raw_pointer_cast(ph_testNodes.data()),
				dv_tp.size() * sizeof(double), cudaMemcpyHostToDevice, s1);
			
			int sizeN = 4 * sizeof(double);
			pinnedVector ph_N(N, N + sizeN);
			device_vector<double> dv_N(sizeN);
			cudaMemcpyAsync(thrust::raw_pointer_cast(dv_N.data()), thrust::raw_pointer_cast(ph_N.data()),
				dv_N.size() * sizeof(double), cudaMemcpyHostToDevice, s2);

			cudaDeviceSynchronize();

			double* d_TP = thrust::raw_pointer_cast(dv_tp.data());
			//cudaMalloc((void**)&d_TP, sizeTestNodes);
			GenerateTestNodes<<<1,1>>>(tLower, tUpper, thrust::raw_pointer_cast(dv_N.data()), d_TP);

			//device_ptr<double> dp_TN = device_pointer_cast<double>(d_TP);
			//device_vector<double> dv_tp1(dp_TN, dp_TN + (2 * length));

			printf("TP:\r\n");
			host_vector<double> h_tp(dv_tp.size());
			thrust::copy(dv_tp.begin(), dv_tp.end(), h_tp.begin());
			double *raw_tp = thrust::raw_pointer_cast(h_tp.data());
			Utility::printMatrix(raw_tp, dim3(2, 15));

			const double *h_a = A.data();
			const double *h_c = C.data();

			rows = length;
			cols = length;
			
			device_vector<double> d_tp0(rows);
			device_vector<double> d_cn0(rows);
			thrust::copy(thrust::cuda::par.on(s1), dv_tp.begin()+ 2, dv_tp.begin() + 2 + rows, d_tp0.begin()); //first column
			thrust::copy(thrust::cuda::par.on(s1), dv_tp.begin()+ 2, dv_tp.begin() + 2 + rows, d_cn0.begin()); //first column

			device_vector<double> d_tp1(rows);
			device_vector<double> d_cn1(rows);
			thrust::copy(thrust::cuda::par.on(s2), dv_tp.begin() + 2 + rows, dv_tp.begin() + 2 + 2 * rows, d_tp1.begin()); //second column
			thrust::copy(thrust::cuda::par.on(s2), dv_tp.begin() + 2 + cols, dv_tp.begin() + 2 + 2 * cols, d_cn1.begin()); //second column

			device_vector<double> d_a(h_a, h_a + (A.rows() * A.cols()));
			device_vector<double> d_c(h_c, h_c + (C.rows() * C.cols()));

			device_vector<double> d_PHI1(rows * cols);

			cudaStreamSynchronize(s1);
			cudaStreamSynchronize(s2);
			counting_iterator<int> first(0);
			counting_iterator<int> last(rows * cols);


			//thrust::transform(thrust::cuda::par.on(s1), d_PHI1.begin(), d_PHI1.end(), d_PHI1.begin(), 
			//	phi1_functor2(raw_pointer_cast(d_tp0.data()), A(0, 0), raw_pointer_cast(d_cn0.data()), C(0, 0), rows, cols));
			thrust::for_each(thrust::cuda::par.on(s1),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_PHI1.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_PHI1.end())),
				phi_functor3(raw_pointer_cast(d_tp0.data()), A(0, 0), raw_pointer_cast(d_cn0.data()), C(0, 0), rows, cols)
			);
			device_vector<double> d_PHI2(rows * cols);

			//thrust::transform(thrust::cuda::par.on(s2), d_PHI2.begin(), d_PHI2.end(), d_PHI2.begin(), 
			//	phi2_functor2(raw_pointer_cast(d_tp1.data()), A(0, 1), raw_pointer_cast(d_cn1.data()), C(0, 1), rows, cols));
			thrust::for_each(thrust::cuda::par.on(s2),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_PHI2.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_PHI2.end())),
				phi_functor3(raw_pointer_cast(d_tp1.data()), A(0, 1), raw_pointer_cast(d_cn1.data()), C(0, 1), rows, cols)
			);

			device_vector<double> d_D(rows * cols);
			cudaStreamSynchronize(s1);
			cudaStreamSynchronize(s2);
			thrust::transform(thrust::cuda::par.on(s1), d_PHI1.begin(), d_PHI1.end(), d_PHI2.begin(), d_D.begin(),
				d_functor2());
			cudaStreamSynchronize(s1);

			d_PHI1.clear();
			d_PHI1.shrink_to_fit();
			d_PHI2.clear();
			d_PHI2.shrink_to_fit();

			//Calculate Dt
			device_vector<double> d_Dt(rows * cols);
			double scalarDt = -2 * (A(0, 0) / C(0, 0)) * (A(0, 0) / C(0, 0));
			//thrust::transform(thrust::cuda::par.on(s1), d_Dt.begin(), d_Dt.end(), d_D.begin(), d_Dt.begin(), 
			//	dt_functor2(raw_pointer_cast(d_tp0.data()), scalarDt, raw_pointer_cast(d_cn0.data()), rows, cols));
			thrust::for_each(thrust::cuda::par.on(s1),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_D.begin(), d_Dt.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_D.end(), d_Dt.end())),
				dt_functor3(raw_pointer_cast(d_tp0.data()), scalarDt, raw_pointer_cast(d_cn0.data()), rows, cols)
			);

			//Calculate Dx
			device_vector<double> d_Dx(rows * cols);
			double scalarDx = -2 * ((A(0, 1) / C(0, 1)) * (A(0, 1) / C(0, 1)));
			/*thrust::transform(thrust::cuda::par.on(s2), d_Dx.begin(), d_Dx.end(), d_D.begin(), d_Dx.begin(),
			dx_functor2(raw_pointer_cast(d_tp1.data()), scalarDx, raw_pointer_cast(d_cn1.data()), rows, cols));*/
			thrust::for_each(thrust::cuda::par.on(s2),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_D.begin(), d_Dx.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_D.end(), d_Dx.end())),
				dx_functor3(raw_pointer_cast(d_tp1.data()), scalarDx, raw_pointer_cast(d_cn1.data()), rows, cols)
			);
			//Calculate Dxx
			device_vector<double> d_Dxx(rows * cols);
			double sA = A(0, 1) * A(0, 1);
			double qA = sA * sA;
			double sC = C(0, 1) * C(0, 1);
			double qC = sC * sC;
			double scalarDxx1 = 4 * qA / qC;
			double scalarDxx2 = -2 * sA / sC;
			/*thrust::transform(thrust::cuda::par.on(s3), d_Dxx.begin(), d_Dxx.end(), d_D.begin(), d_Dxx.begin(),
			dxx_functor2(raw_pointer_cast(d_tp1.data()), scalarDxx1, scalarDxx2, raw_pointer_cast(d_cn1.data()), rows, cols));*/
			thrust::for_each(thrust::cuda::par.on(s3),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_D.begin(), d_Dxx.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_D.end(), d_Dxx.end())),
				dxx_functor3(raw_pointer_cast(d_tp1.data()), scalarDxx1, scalarDxx2, raw_pointer_cast(d_cn1.data()), rows, cols)
			);
			cudaStreamSynchronize(s1);
			cudaStreamSynchronize(s2);
			cudaStreamSynchronize(s3);
			cudaStreamSynchronize(s4);


			//cudaDeviceSynchronize();
			//double *h_Phi1 = (double*)malloc(sizeof(double) * rows * cols);
			//double *p_Phi1 = d_PHI1.data().get();
			//cudaError_t e = cudaMemcpy(h_Phi1, p_Phi1, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			//if (e != cudaSuccess)
			//	printf("cudaMemcpy h_Phi1 returned error %s (code %d), line(%d) when copying%i\n",
			//		cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			//Eigen::Map<Eigen::MatrixXd> dataMapPhi1(h_Phi1, rows, cols);
			//MatrixXd Phi1 = dataMapPhi1.eval();

			//cudaDeviceSynchronize();
			//double *h_Phi2 = (double*)malloc(sizeof(double) * rows * cols);
			//double *p_Phi2 = d_PHI2.data().get();
			//e = cudaMemcpy(h_Phi2, p_Phi2, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			//if (e != cudaSuccess)
			//	printf("cudaMemcpy h_Phi2 returned error %s (code %d), line(%d) when copying%i\n",
			//		cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			//Eigen::Map<Eigen::MatrixXd> dataMapPhi2(h_Phi2, rows, cols);
			//MatrixXd Phi2 = dataMapPhi2.eval();

			cudaDeviceSynchronize();
			double *h_D = (double*)malloc(sizeof(double) * rows * cols);
			double *p_D = d_D.data().get();
			//cudaError_t e = cudaMemcpy(h_D, p_D, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			cudaError_t e = cudaMemcpyAsync(h_D, p_D, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost, s1);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_D returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);


			cudaDeviceSynchronize();
			double *h_Dt = (double*)malloc(sizeof(double) * rows * cols);
			double *p_Dt = d_Dt.data().get();
			//e = cudaMemcpy(h_Dt, p_Dt, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			e = cudaMemcpyAsync(h_Dt, p_Dt, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost, s2);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_Dt returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);


			//cudaDeviceSynchronize();
			double *h_Dx = (double*)malloc(sizeof(double) * rows * cols);
			double *p_Dx = d_Dx.data().get();
			//e = cudaMemcpy(h_Dx, p_Dx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			e = cudaMemcpyAsync(h_Dx, p_Dx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost, s3);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_Dx returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);


			//cudaDeviceSynchronize();
			double *h_Dxx = (double*)malloc(sizeof(double) * rows * cols);
			double *p_Dxx = d_Dxx.data().get();
			//e = cudaMemcpy(h_Dxx, p_Dxx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			e = cudaMemcpyAsync(h_Dxx, p_Dxx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost, s4);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_Dxx returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);


			cudaStreamSynchronize(s1);
			cudaStreamSynchronize(s2);
			cudaStreamSynchronize(s3);
			cudaStreamSynchronize(s4);
			Eigen::Map<Eigen::MatrixXd> dataMapDxx(h_Dxx, rows, cols);
			MatrixXd Dxx = dataMapDxx.eval();
			Eigen::Map<Eigen::MatrixXd> dataMapDx(h_Dx, rows, cols);
			MatrixXd Dx = dataMapDx.eval();
			Eigen::Map<Eigen::MatrixXd> dataMapDt(h_Dt, rows, cols);
			MatrixXd Dt = dataMapDt.eval();
			Eigen::Map<Eigen::MatrixXd> dataMapD(h_D, rows, cols);
			MatrixXd D = dataMapD.eval();

			return { D, Dt, Dx, Dxx };

		}

		vector<MatrixXd> Gaussian::Gaussian2d_1(const MatrixXd & A, const MatrixXd & C, int count)
		{
			cudaStream_t s1, s2, s3, s4;
			cudaStreamCreate(&s1);
			cudaStreamCreate(&s2);
			cudaStreamCreate(&s3);
			cudaStreamCreate(&s4);

			const double *h_a = A.data();
			const double *h_c = C.data();

			nodesDetails nd = this->nodeMap[count];
			rows = nd.rows;
			cols = nd.rows;
			device_vector<double> TN = nd.nodes;
			device_vector<double> d_tp0(nd.rows);
			device_vector<double> d_cn0(nd.rows);
			thrust::copy(thrust::cuda::par.on(s1), TN.begin(), TN.begin() + rows, d_tp0.begin()); //first column
			thrust::copy(thrust::cuda::par.on(s1), TN.begin(), TN.begin() + cols, d_cn0.begin()); //first column

			device_vector<double> d_tp1(nd.rows);
			device_vector<double> d_cn1(nd.rows);
			thrust::copy(thrust::cuda::par.on(s2), TN.begin() + rows, TN.begin() + 2 * rows, d_tp1.begin()); //second column
			thrust::copy(thrust::cuda::par.on(s2), TN.begin() + cols, TN.begin() + 2 * cols, d_cn1.begin()); //second column

			device_vector<double> d_a(h_a, h_a + (A.rows() * A.cols()));
			device_vector<double> d_c(h_c, h_c + (C.rows() * C.cols()));

			device_vector<double> d_PHI1(rows * cols);

			cudaStreamSynchronize(s1);
			cudaStreamSynchronize(s2);
			counting_iterator<int> first(0);
			counting_iterator<int> last(rows * cols);

			//thrust::transform(thrust::cuda::par.on(s1), d_PHI1.begin(), d_PHI1.end(), d_PHI1.begin(), 
			//	phi1_functor2(raw_pointer_cast(d_tp0.data()), A(0, 0), raw_pointer_cast(d_cn0.data()), C(0, 0), rows, cols));
			thrust::for_each(thrust::cuda::par.on(s1),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_PHI1.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_PHI1.end())),
				phi_functor3(raw_pointer_cast(d_tp0.data()), A(0, 0), raw_pointer_cast(d_cn0.data()), C(0, 0), rows, cols)
			);
			device_vector<double> d_PHI2(rows * cols);

			//thrust::transform(thrust::cuda::par.on(s2), d_PHI2.begin(), d_PHI2.end(), d_PHI2.begin(), 
			//	phi2_functor2(raw_pointer_cast(d_tp1.data()), A(0, 1), raw_pointer_cast(d_cn1.data()), C(0, 1), rows, cols));
			thrust::for_each(thrust::cuda::par.on(s2),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_PHI2.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_PHI2.end())),
				phi_functor3(raw_pointer_cast(d_tp1.data()), A(0, 1), raw_pointer_cast(d_cn1.data()), C(0, 1), rows, cols)
			);

			device_vector<double> d_D(rows * cols);
			cudaStreamSynchronize(s1);
			cudaStreamSynchronize(s2);
			thrust::transform(thrust::cuda::par.on(s1), d_PHI1.begin(), d_PHI1.end(), d_PHI2.begin(), d_D.begin(),
				d_functor2());
			cudaStreamSynchronize(s1);

			d_PHI1.clear();
			d_PHI1.shrink_to_fit();
			d_PHI2.clear();
			d_PHI2.shrink_to_fit();

			//Calculate Dt
			device_vector<double> d_Dt(rows * cols);
			double scalarDt = -2 * (A(0, 0) / C(0, 0)) * (A(0, 0) / C(0, 0));
			//thrust::transform(thrust::cuda::par.on(s1), d_Dt.begin(), d_Dt.end(), d_D.begin(), d_Dt.begin(), 
			//	dt_functor2(raw_pointer_cast(d_tp0.data()), scalarDt, raw_pointer_cast(d_cn0.data()), rows, cols));
			thrust::for_each(thrust::cuda::par.on(s1),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_D.begin(), d_Dt.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_D.end(), d_Dt.end())),
				dt_functor3(raw_pointer_cast(d_tp0.data()), scalarDt, raw_pointer_cast(d_cn0.data()), rows, cols)
			);

			//Calculate Dx
			device_vector<double> d_Dx(rows * cols);
			double scalarDx = -2 * ((A(0, 1) / C(0, 1)) * (A(0, 1) / C(0, 1)));
			/*thrust::transform(thrust::cuda::par.on(s2), d_Dx.begin(), d_Dx.end(), d_D.begin(), d_Dx.begin(),
			dx_functor2(raw_pointer_cast(d_tp1.data()), scalarDx, raw_pointer_cast(d_cn1.data()), rows, cols));*/
			thrust::for_each(thrust::cuda::par.on(s2),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_D.begin(), d_Dx.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_D.end(), d_Dx.end())),
				dx_functor3(raw_pointer_cast(d_tp1.data()), scalarDx, raw_pointer_cast(d_cn1.data()), rows, cols)
			);
			//Calculate Dxx
			device_vector<double> d_Dxx(rows * cols);
			double sA = A(0, 1) * A(0, 1);
			double qA = sA * sA;
			double sC = C(0, 1) * C(0, 1);
			double qC = sC * sC;
			double scalarDxx1 = 4 * qA / qC;
			double scalarDxx2 = -2 * sA / sC;
			/*thrust::transform(thrust::cuda::par.on(s3), d_Dxx.begin(), d_Dxx.end(), d_D.begin(), d_Dxx.begin(),
			dxx_functor2(raw_pointer_cast(d_tp1.data()), scalarDxx1, scalarDxx2, raw_pointer_cast(d_cn1.data()), rows, cols));*/
			thrust::for_each(thrust::cuda::par.on(s3),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_D.begin(), d_Dxx.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_D.end(), d_Dxx.end())),
				dxx_functor3(raw_pointer_cast(d_tp1.data()), scalarDxx1, scalarDxx2, raw_pointer_cast(d_cn1.data()), rows, cols)
			);
			cudaStreamSynchronize(s1);
			cudaStreamSynchronize(s2);
			cudaStreamSynchronize(s3);
			cudaStreamSynchronize(s4);


			//cudaDeviceSynchronize();
			//double *h_Phi1 = (double*)malloc(sizeof(double) * rows * cols);
			//double *p_Phi1 = d_PHI1.data().get();
			//cudaError_t e = cudaMemcpy(h_Phi1, p_Phi1, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			//if (e != cudaSuccess)
			//	printf("cudaMemcpy h_Phi1 returned error %s (code %d), line(%d) when copying%i\n",
			//		cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			//Eigen::Map<Eigen::MatrixXd> dataMapPhi1(h_Phi1, rows, cols);
			//MatrixXd Phi1 = dataMapPhi1.eval();

			//cudaDeviceSynchronize();
			//double *h_Phi2 = (double*)malloc(sizeof(double) * rows * cols);
			//double *p_Phi2 = d_PHI2.data().get();
			//e = cudaMemcpy(h_Phi2, p_Phi2, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			//if (e != cudaSuccess)
			//	printf("cudaMemcpy h_Phi2 returned error %s (code %d), line(%d) when copying%i\n",
			//		cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			//Eigen::Map<Eigen::MatrixXd> dataMapPhi2(h_Phi2, rows, cols);
			//MatrixXd Phi2 = dataMapPhi2.eval();

			cudaDeviceSynchronize();
			double *h_D = (double*)malloc(sizeof(double) * rows * cols);
			double *p_D = d_D.data().get();
			//cudaError_t e = cudaMemcpy(h_D, p_D, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			cudaError_t e = cudaMemcpyAsync(h_D, p_D, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost, s1);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_D returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);


			cudaDeviceSynchronize();
			double *h_Dt = (double*)malloc(sizeof(double) * rows * cols);
			double *p_Dt = d_Dt.data().get();
			//e = cudaMemcpy(h_Dt, p_Dt, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			e = cudaMemcpyAsync(h_Dt, p_Dt, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost, s2);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_Dt returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);


			//cudaDeviceSynchronize();
			double *h_Dx = (double*)malloc(sizeof(double) * rows * cols);
			double *p_Dx = d_Dx.data().get();
			//e = cudaMemcpy(h_Dx, p_Dx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			e = cudaMemcpyAsync(h_Dx, p_Dx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost, s3);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_Dx returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);


			//cudaDeviceSynchronize();
			double *h_Dxx = (double*)malloc(sizeof(double) * rows * cols);
			double *p_Dxx = d_Dxx.data().get();
			//e = cudaMemcpy(h_Dxx, p_Dxx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			e = cudaMemcpyAsync(h_Dxx, p_Dxx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost, s4);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_Dxx returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);


			cudaStreamSynchronize(s1);
			cudaStreamSynchronize(s2);
			cudaStreamSynchronize(s3);
			cudaStreamSynchronize(s4);
			Eigen::Map<Eigen::MatrixXd> dataMapDxx(h_Dxx, rows, cols);
			MatrixXd Dxx = dataMapDxx.eval();
			Eigen::Map<Eigen::MatrixXd> dataMapDx(h_Dx, rows, cols);
			MatrixXd Dx = dataMapDx.eval();
			Eigen::Map<Eigen::MatrixXd> dataMapDt(h_Dt, rows, cols);
			MatrixXd Dt = dataMapDt.eval();
			Eigen::Map<Eigen::MatrixXd> dataMapD(h_D, rows, cols);
			MatrixXd D = dataMapD.eval();

			return { D, Dt, Dx, Dxx };

		}
		
		vector<MatrixXd> Gaussian::Gaussian2d(const MatrixXd & CN, const MatrixXd & A, const MatrixXd & C)
		{

			
			cudaStream_t s1;
			cudaStreamCreate(&s1);
			this->cols = CN.rows();
			
			int sizeCentralNodes = sizeof(double) * CN.rows() * CN.cols();
			const double *h_centralNodes = CN.data();
			/*pinnedVector ph_centralNodes(h_centralNodes, h_centralNodes + sizeCentralNodes);
			device_vector<mytype> pd_centralNodes(sizeCentralNodes);
			cudaMemcpyAsync(thrust::raw_pointer_cast(pd_centralNodes.data()), thrust::raw_pointer_cast(ph_centralNodes.data()),
				pd_centralNodes.size() * sizeof(mytype), cudaMemcpyHostToDevice, s1)
			this->centralNodes = pd_centralNodes;*/

			double *d_centralNodes;
			cudaMalloc((void**)&d_centralNodes, sizeCentralNodes);
			device_ptr<double> dp_centralNodes = thrust::device_pointer_cast<double>(d_centralNodes);
			cudaMemcpyAsync(d_centralNodes, h_centralNodes, sizeCentralNodes, cudaMemcpyKind::cudaMemcpyHostToDevice, s1);
			
			cudaDeviceSynchronize();

			device_vector<double> dv_centralNodes(dp_centralNodes, dp_centralNodes + (CN.rows() * CN.cols()));
			this->centralNodes = dv_centralNodes;
			
			return this->Gaussian2d(A, C);
		}

		vector<MatrixXd> Gaussian::Gaussian2d(const MatrixXd & A, const MatrixXd & C)
		{
			cudaStream_t s1, s2, s3, s4;
			cudaStreamCreate(&s1);
			cudaStreamCreate(&s2);
			cudaStreamCreate(&s3);
			cudaStreamCreate(&s4);

			const double *h_a = A.data();
			const double *h_c = C.data();

			//printf("TP:\r\n");
			//host_vector<double> h_tp(testNodes.size());
			//thrust::copy(testNodes.begin(), testNodes.end(), h_tp.begin());
			//double *raw_tp = thrust::raw_pointer_cast(h_tp.data());
			//printMatrix(raw_tp, dim3(2, rows));

			device_vector<double> d_tp0(rows);
			device_vector<double> d_cn0(cols);
			thrust::copy(thrust::cuda::par.on(s1), testNodes.begin(), testNodes.begin() + rows, d_tp0.begin()); //first column
			thrust::copy(thrust::cuda::par.on(s1), centralNodes.begin(), centralNodes.begin() + cols, d_cn0.begin()); //first column

			device_vector<double> d_tp1(rows);
			device_vector<double> d_cn1(cols);
			thrust::copy(thrust::cuda::par.on(s2), testNodes.begin() + rows, testNodes.begin() + 2 * rows, d_tp1.begin()); //second column
			thrust::copy(thrust::cuda::par.on(s2), centralNodes.begin() + cols, centralNodes.begin() + 2 * cols, d_cn1.begin()); //second column

			device_vector<double> d_a(h_a, h_a + (A.rows() * A.cols()));
			device_vector<double> d_c(h_c, h_c + (C.rows() * C.cols()));

			device_vector<double> d_PHI1(rows * cols);
	
			cudaStreamSynchronize(s1);
			cudaStreamSynchronize(s2);
			counting_iterator<int> first(0);
			counting_iterator<int> last(rows * cols);

			//thrust::transform(thrust::cuda::par.on(s1), d_PHI1.begin(), d_PHI1.end(), d_PHI1.begin(), 
			//	phi1_functor2(raw_pointer_cast(d_tp0.data()), A(0, 0), raw_pointer_cast(d_cn0.data()), C(0, 0), rows, cols));
			thrust::for_each(thrust::cuda::par.on(s1),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_PHI1.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_PHI1.end())),
				phi_functor3(raw_pointer_cast(d_tp0.data()), A(0, 0), raw_pointer_cast(d_cn0.data()), C(0, 0), rows, cols)
			);
			device_vector<double> d_PHI2(rows * cols);
			
			//thrust::transform(thrust::cuda::par.on(s2), d_PHI2.begin(), d_PHI2.end(), d_PHI2.begin(), 
			//	phi2_functor2(raw_pointer_cast(d_tp1.data()), A(0, 1), raw_pointer_cast(d_cn1.data()), C(0, 1), rows, cols));
			thrust::for_each(thrust::cuda::par.on(s2),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_PHI2.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_PHI2.end())),
				phi_functor3(raw_pointer_cast(d_tp1.data()), A(0, 1), raw_pointer_cast(d_cn1.data()), C(0, 1), rows, cols)
			);

			device_vector<double> d_D(rows * cols);
			cudaStreamSynchronize(s1);
			cudaStreamSynchronize(s2);
			thrust::transform(thrust::cuda::par.on(s1), d_PHI1.begin(), d_PHI1.end(), d_PHI2.begin(), d_D.begin(),
				d_functor2());
			cudaStreamSynchronize(s1);
			
			d_PHI1.clear();
			d_PHI1.shrink_to_fit();
			d_PHI2.clear();
			d_PHI2.shrink_to_fit();

			//Calculate Dt
			device_vector<double> d_Dt(rows * cols);
			double scalarDt = -2 * (A(0, 0) / C(0, 0)) * (A(0, 0) / C(0, 0));
			//thrust::transform(thrust::cuda::par.on(s1), d_Dt.begin(), d_Dt.end(), d_D.begin(), d_Dt.begin(), 
			//	dt_functor2(raw_pointer_cast(d_tp0.data()), scalarDt, raw_pointer_cast(d_cn0.data()), rows, cols));
			thrust::for_each(thrust::cuda::par.on(s1),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_D.begin(), d_Dt.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_D.end(), d_Dt.end())),
				dt_functor3(raw_pointer_cast(d_tp0.data()), scalarDt, raw_pointer_cast(d_cn0.data()), rows, cols)
			);

			//Calculate Dx
			device_vector<double> d_Dx(rows * cols);
			double scalarDx = -2 * ((A(0, 1) / C(0, 1)) * (A(0, 1) / C(0, 1)));
			/*thrust::transform(thrust::cuda::par.on(s2), d_Dx.begin(), d_Dx.end(), d_D.begin(), d_Dx.begin(),
				dx_functor2(raw_pointer_cast(d_tp1.data()), scalarDx, raw_pointer_cast(d_cn1.data()), rows, cols));*/
			thrust::for_each(thrust::cuda::par.on(s2),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_D.begin(), d_Dx.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_D.end(), d_Dx.end())),
				dx_functor3(raw_pointer_cast(d_tp1.data()), scalarDx, raw_pointer_cast(d_cn1.data()), rows, cols)
			);
			//Calculate Dxx
			device_vector<double> d_Dxx(rows * cols);
			double sA = A(0, 1) * A(0, 1);
			double qA = sA * sA;
			double sC = C(0, 1) * C(0, 1);
			double qC = sC * sC;
			double scalarDxx1 = 4 * qA / qC;
			double scalarDxx2 = -2 * sA / sC;
			/*thrust::transform(thrust::cuda::par.on(s3), d_Dxx.begin(), d_Dxx.end(), d_D.begin(), d_Dxx.begin(),
				dxx_functor2(raw_pointer_cast(d_tp1.data()), scalarDxx1, scalarDxx2, raw_pointer_cast(d_cn1.data()), rows, cols));*/
			thrust::for_each(thrust::cuda::par.on(s3),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_D.begin(), d_Dxx.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_D.end(), d_Dxx.end())),
				dxx_functor3(raw_pointer_cast(d_tp1.data()), scalarDxx1, scalarDxx2, raw_pointer_cast(d_cn1.data()), rows, cols)
			);
			cudaStreamSynchronize(s1);
			cudaStreamSynchronize(s2);
			cudaStreamSynchronize(s3);
			cudaStreamSynchronize(s4);


			//cudaDeviceSynchronize();
			//double *h_Phi1 = (double*)malloc(sizeof(double) * rows * cols);
			//double *p_Phi1 = d_PHI1.data().get();
			//cudaError_t e = cudaMemcpy(h_Phi1, p_Phi1, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			//if (e != cudaSuccess)
			//	printf("cudaMemcpy h_Phi1 returned error %s (code %d), line(%d) when copying%i\n",
			//		cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			//Eigen::Map<Eigen::MatrixXd> dataMapPhi1(h_Phi1, rows, cols);
			//MatrixXd Phi1 = dataMapPhi1.eval();

			//cudaDeviceSynchronize();
			//double *h_Phi2 = (double*)malloc(sizeof(double) * rows * cols);
			//double *p_Phi2 = d_PHI2.data().get();
			//e = cudaMemcpy(h_Phi2, p_Phi2, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			//if (e != cudaSuccess)
			//	printf("cudaMemcpy h_Phi2 returned error %s (code %d), line(%d) when copying%i\n",
			//		cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			//Eigen::Map<Eigen::MatrixXd> dataMapPhi2(h_Phi2, rows, cols);
			//MatrixXd Phi2 = dataMapPhi2.eval();

			cudaDeviceSynchronize();
			double *h_D = (double*)malloc(sizeof(double) * rows * cols);
			double *p_D = d_D.data().get();
			//cudaError_t e = cudaMemcpy(h_D, p_D, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			cudaError_t e = cudaMemcpyAsync(h_D, p_D, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost, s1);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_D returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			

			cudaDeviceSynchronize();
			double *h_Dt = (double*)malloc(sizeof(double) * rows * cols);
			double *p_Dt = d_Dt.data().get();
			//e = cudaMemcpy(h_Dt, p_Dt, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			e = cudaMemcpyAsync(h_Dt, p_Dt, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost, s2);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_Dt returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			

			//cudaDeviceSynchronize();
			double *h_Dx = (double*)malloc(sizeof(double) * rows * cols);
			double *p_Dx = d_Dx.data().get();
			//e = cudaMemcpy(h_Dx, p_Dx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			e = cudaMemcpyAsync(h_Dx, p_Dx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost,s3);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_Dx returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			

			//cudaDeviceSynchronize();
			double *h_Dxx = (double*)malloc(sizeof(double) * rows * cols);
			double *p_Dxx = d_Dxx.data().get();
			//e = cudaMemcpy(h_Dxx, p_Dxx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			e = cudaMemcpyAsync(h_Dxx, p_Dxx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost,s4);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_Dxx returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			

			cudaStreamSynchronize(s1);
			cudaStreamSynchronize(s2);
			cudaStreamSynchronize(s3);
			cudaStreamSynchronize(s4);
			Eigen::Map<Eigen::MatrixXd> dataMapDxx(h_Dxx, rows, cols);
			MatrixXd Dxx = dataMapDxx.eval();
			Eigen::Map<Eigen::MatrixXd> dataMapDx(h_Dx, rows, cols);
			MatrixXd Dx = dataMapDx.eval();
			Eigen::Map<Eigen::MatrixXd> dataMapDt(h_Dt, rows, cols);
			MatrixXd Dt = dataMapDt.eval();
			Eigen::Map<Eigen::MatrixXd> dataMapD(h_D, rows, cols);
			MatrixXd D = dataMapD.eval();

			return { D, Dt, Dx, Dxx };

		}

		vector<MatrixXd> Gaussian::Gaussian2d(const MatrixXd & TP, const MatrixXd & CN, const MatrixXd & A, const MatrixXd & C)
		{
			int rows = TP.rows();
			const double *h_tp0 = TP.col(0).data();
			const double *h_tp1 = TP.col(1).data();
			const double *h_cn = CN.data();
			const double *h_a = A.data();
			const double *h_c = C.data();
			//wcout << printMatrix(TP.col(0)) << endl;
			//wcout << printMatrix(TP.col(1)) << endl;

			device_vector<double> d_tp0(h_tp0, h_tp0 + (TP.rows()));
			device_vector<double> d_tp1(h_tp1, h_tp1 + (TP.rows()));
			device_vector<double> d_cn(h_cn, h_cn + (CN.rows() * CN.cols()));
			device_vector<double> d_a(h_a, h_a + (A.rows() * A.cols()));
			device_vector<double> d_c(h_c, h_c + (C.rows() * C.cols()));

			device_vector<double> d_PHI1(rows * rows);
			device_vector<double> d_PHI2(rows * rows);
			device_vector<double> d_D(rows * rows);
			device_vector<double> d_Dt(rows * rows);
			device_vector<double> d_Dx(rows * rows);
			device_vector<double> d_Dxx(rows * rows);

			device_vector<double> d_test1(rows * rows);
			device_vector<double> d_test2(rows * rows);

			//thrust::transform(d_PHI1.begin(), d_PHI1.end(), d_PHI1.begin(), phi_functor2(d_tp0.data(), A(0, 0), CN.col(0).data(), C(0, 0), rows, TP.cols()));

			for (int i = 0; i < rows; i++)
			{
				//if (i == 0)
				//{
				//	d_test1.insert(d_test1.begin(), d_tp0.begin(), d_tp0.end());
				//	d_test2.insert(d_test2.begin(), d_tp1.begin(), d_tp1.end());
				//}
				//Calculate Phi1 & Phi2
				thrust::device_vector<double> phi1(rows);
				thrust::transform(d_tp0.begin(), d_tp0.end(), phi1.begin(), phi_functor(A(0, 0), CN(i, 0), C(0, 0)));
				d_PHI1.insert(d_PHI1.begin(), phi1.begin(), phi1.end());

				thrust::device_vector<double> phi2(rows);
				thrust::transform(d_tp1.begin(), d_tp1.end(), phi2.begin(), phi_functor(A(0, 1), CN(i, 1), C(0, 1)));
				d_PHI2.insert(d_PHI2.begin(), phi2.begin(), phi2.end());

				//Calculate D
				thrust::device_vector<double> d(rows);
				thrust::transform(phi1.begin(), phi1.end(), phi2.begin(), d.begin(), thrust::multiplies<double>());
				d_D.insert(d_D.begin(), d.begin(), d.end());

				//Calculate Dt
				thrust::device_vector<double> a1(rows);
				double scalarDt = -2 * (A(0, 0) / C(0, 0)) * (A(0, 0) / C(0, 0));
				thrust::transform(d_tp0.begin(), d_tp0.end(), a1.begin(), scalarVectorDifference_functor(scalarDt, CN(i, 0)));

				thrust::device_vector<double> b1(rows);
				thrust::transform(phi1.begin(), phi1.end(), a1.begin(), b1.begin(), thrust::multiplies<double>());

				thrust::device_vector<double> dt(rows);
				thrust::transform(phi2.begin(), phi2.end(), b1.begin(), dt.begin(), thrust::multiplies<double>());
				d_Dt.insert(d_Dt.begin(), dt.begin(), dt.end());
				a1.clear();
				a1.shrink_to_fit();
				b1.clear();
				b1.shrink_to_fit();

				//Calculate Dx
				double scalarDx = -2 * ((A(0, 1) / C(0, 1)) * (A(0, 1) / C(0, 1)));
				thrust::device_vector<double> a2(rows);
				thrust::transform(d_tp1.begin(), d_tp1.end(), a2.begin(), scalarVectorDifference_functor(scalarDx, CN(i, 1)));

				thrust::device_vector<double> b2(rows);
				thrust::transform(phi1.begin(), phi1.end(), a2.begin(), b2.begin(), thrust::multiplies<double>());

				thrust::device_vector<double> c2(rows);
				thrust::transform(phi2.begin(), phi2.end(), b2.begin(), c2.begin(), thrust::multiplies<double>());

				thrust::device_vector<double> dx(rows);
				thrust::transform(d_tp1.begin(), d_tp1.end(), c2.begin(), dx.begin(), thrust::multiplies<double>());
				d_Dx.insert(d_Dx.begin(), dx.begin(), dx.end());
				a2.clear();
				a2.shrink_to_fit();
				b2.clear();
				b2.shrink_to_fit();
				c2.clear();
				c2.shrink_to_fit();
				dx.clear();
				dx.shrink_to_fit();
				//Calculate Dxx
				double sA = A(0, 1) * A(0, 1);
				double qA = sA * sA;
				double sC = C(0, 1) * C(0, 1);
				double qC = sC * sC;

				thrust::device_vector<double> dTpCn(rows);
				thrust::transform(d_tp1.begin(), d_tp1.end(), dTpCn.begin(), vectorScalarDifference_functor(CN(i, 1)));

				thrust::device_vector<double> sDTpCn(rows);
				thrust::transform(dTpCn.begin(), dTpCn.end(), dTpCn.begin(), sDTpCn.begin(), thrust::multiplies<double>());

				thrust::device_vector<double> a3(rows);
				double scalarDxx1 = 4 * qA / qC;

				thrust::transform(sDTpCn.begin(), sDTpCn.end(), a3.begin(), vectorScalarMultiply_functor(scalarDxx1));

				thrust::device_vector<double> b3(rows);
				double scalarDxx2 = -2 * sA / sC;

				thrust::transform(a3.begin(), a3.end(), b3.begin(), vectorAddScalar_functor(scalarDxx2));

				thrust::device_vector<double> c3(rows);
				thrust::transform(d.begin(), d.end(), b3.begin(), c3.begin(), thrust::multiplies<double>());

				thrust::device_vector<double> sTpCol1(rows);
				thrust::transform(d_tp1.begin(), d_tp1.end(), d_tp1.begin(), sTpCol1.begin(), thrust::multiplies<double>());

				thrust::device_vector<double> dxx(rows);
				//thrust::device_vector<double> rsTpCol(rows);
				//thrust::copy(sTpCol1.rbegin(), sTpCol1.rend(), rsTpCol.begin());
				thrust::transform(c3.begin(), c3.end(), sTpCol1.begin(), dxx.begin(), thrust::multiplies<double>());
				d_Dxx.insert(d_Dxx.begin(), dxx.begin(), dxx.end());

				//d_test1.insert(d_test1.begin(), d_tp1.begin(), d_tp1.end());
				dTpCn.clear();
				dTpCn.shrink_to_fit();
				sDTpCn.clear();
				sDTpCn.shrink_to_fit();
				b3.clear();
				b3.shrink_to_fit();
				c3.clear();
				c3.shrink_to_fit();
				sTpCol1.clear();
				sTpCol1.shrink_to_fit();

				/*if (i == 1)
				{
					d_test1.insert(d_test1.begin(), d_tp1.begin(), d_tp1.end());
					d_test2.insert(d_test2.begin(), d_Dxx.begin(), d_Dxx.end());
				}*/
			}

			//printf("Phi1:\r\n");
			//host_vector<double> h_PHI1(d_PHI1.size());
			//thrust::copy(d_PHI1.begin(), d_PHI1.end(), h_PHI1.begin());
			//double *raw_PHI1 = thrust::raw_pointer_cast(h_PHI1.data());
			//printMatrix(raw_PHI1, dim3(1, rows));
			//
			//printf("Phi2:\r\n");
			//host_vector<double> h_PHI2(d_PHI2.size());
			//thrust::copy(d_PHI2.begin(), d_PHI2.end(), h_PHI2.begin());
			//double *raw_PHI2 = thrust::raw_pointer_cast(h_PHI2.data());
			//printMatrix(raw_PHI2, dim3(1, rows));
			//
			//printf("D:\r\n");
			//host_vector<double> hv_D(d_D.size());
			//thrust::copy(d_D.begin(), d_D.end(), hv_D.begin());
			//double *raw_D = thrust::raw_pointer_cast(hv_D.data());
			//printMatrix(raw_D, dim3(1, rows));

			//printf("Dt:\r\n");
			//host_vector<double> hv_Dt(d_Dt.size());
			//thrust::copy(d_Dt.begin(), d_Dt.end(), hv_Dt.begin());
			//double *raw_Dt = thrust::raw_pointer_cast(hv_Dt.data());
			//printMatrix(raw_Dt, dim3(1, rows));

			//printf("Dx:\r\n");
			//host_vector<double> hv_Dx(d_Dx.size());
			//thrust::copy(d_Dx.begin(), d_Dx.end(), hv_Dx.begin());
			//double *raw_Dx = thrust::raw_pointer_cast(hv_Dx.data());
			//printMatrix(raw_Dx, dim3(1, rows));

			//printf("Dxx:\r\n");
			//host_vector<double> hv_Dxx(d_Dxx.size());
			//thrust::copy(d_Dxx.begin(), d_Dxx.end(), hv_Dxx.begin());
			//double *raw_Dxx = thrust::raw_pointer_cast(hv_Dxx.data());
			//printMatrix(raw_Dxx, dim3(1, rows));

			//printf("test1:\r\n");
			//host_vector<double> h_test1(d_test1.size());
			//thrust::copy(d_test1.begin(), d_test1.end(), h_test1.begin());
			//double *raw_test1 = thrust::raw_pointer_cast(h_test1.data());
			//printMatrix(raw_test1, dim3(1, rows));

			//printf("test2:\r\n");
			//host_vector<double> h_test2(d_test2.size());
			//thrust::copy(d_test2.begin(), d_test2.end(), h_test2.begin());
			//double *raw_test2 = thrust::raw_pointer_cast(h_test2.data());
			//printMatrix(raw_test2, dim3(1, rows));

			cudaDeviceSynchronize();
			//host_vector<double> h_D(rows * rows);
			//thrust::copy(d_D.begin(), d_D.end(), h_D.begin());
			//cudaDeviceSynchronize();
			//MatrixXd D = MatrixXd::Zero(rows, rows +1);
			/*Eigen::Map<Eigen::MatrixXd> dataMapD(h_D.data(), rows, rows);
			Matrix<double, 15,15,StorageOptions::DontAlign> D = dataMapD.eval();*/
			//CopyToMatrix(D, h_D.data(), dim3(rows, rows));
			//wcout << printMatrix(D)<< endl;
			double *h_D = (double*)malloc(sizeof(double) * rows * rows);
			double *p_D = d_D.data().get();
			cudaError_t e = cudaMemcpy(h_D, p_D, sizeof(double) * rows * rows, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_D returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * rows);
			Eigen::Map<Eigen::MatrixXd> dataMapD(h_D, rows, rows);
			MatrixXd D = dataMapD.eval();

			cudaDeviceSynchronize();
			double *h_Dt = (double*)malloc(sizeof(double) * rows * rows);
			double *p_Dt = d_Dt.data().get();
			e = cudaMemcpy(h_Dt, p_Dt, sizeof(double) * rows * rows, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_Dt returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * rows);
			Eigen::Map<Eigen::MatrixXd> dataMapDt(h_Dt, rows, rows);
			MatrixXd Dt = dataMapDt.eval();

			cudaDeviceSynchronize();
			double *h_Dx = (double*)malloc(sizeof(double) * rows * rows);
			double *p_Dx = d_Dx.data().get();
			e = cudaMemcpy(h_Dx, p_Dx, sizeof(double) * rows * rows, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_Dx returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * rows);
			Eigen::Map<Eigen::MatrixXd> dataMapDx(h_Dx, rows, rows);
			MatrixXd Dx = dataMapDx.eval();

			cudaDeviceSynchronize();
			double *h_Dxx = (double*)malloc(sizeof(double) * rows * rows);
			double *p_Dxx = d_Dxx.data().get();
			e = cudaMemcpy(h_Dxx, p_Dxx, sizeof(double) * rows * rows, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_Dxx returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * rows);
			Eigen::Map<Eigen::MatrixXd> dataMapDxx(h_Dxx, rows, rows);
			MatrixXd Dxx = dataMapDxx.eval();

			return { D, Dt, Dx, Dxx };
		}

		void Gaussian::subnumber(int b, int d, double matrix[])
		{

			double *L = NULL;
			if (d == 1)
			{
				double * l = (double*)malloc(3 * sizeof(double));
				l[0] = 1;
				l[1] = 1;
				l[2] = b;
				L = l;
			}
			else
			{
				int nbot = 1;

				int Lrows = 0;
				int Lcols = 0;
				for (int i = 0; i < b - d + 1; i++)
				{
					double* indextemp = (double*)malloc(512 * sizeof(double));

					subnumber(b - (i + 1), d - 1, indextemp);
					//printMatrix_CUDA(indextemp, dim3(indextemp[0], indextemp[1]));

					int s = indextemp[0];
					int ntop = nbot + s - 1;

					double*l = (double*)malloc((ntop*d + 2) * sizeof(double));

					l[0] = ntop;
					l[1] = d;
					double *ones = (double*)malloc((s + 2) * sizeof(double));
					ones[0] = s;
					ones[1] = 1;

					thrust::fill(thrust::seq, ones + 2, ones + 2 + s, (i + 1));

					int start = nbot;
					int end = start + ntop - nbot;

					//fill the first column with 'ones'
					//thrust::fill(thrust::seq, l + 2 + start, l + 2 + end, (i + 1));
					//fill the rest with 'indextemp'
					//thrust::copy(thrust::seq, indextemp + 2, indextemp + 2 + (int)(l[0] * l[1]) - 1, l + start + ntop);
					int jMin = 0;
					int increment = 1;
					if (L != NULL)
					{
						int count = 0;
						for (int x = 0; x < L[1]; x++)
							for (int y = 0; y < L[0]; y++)
							{
								int diff = l[0] - L[0];
								l[count + 2 + x * diff] = L[count + 2];
								//int indx = (x * L[0]) + y + 2;

								//l[indx] = L[indx];
								count++;
							}
						jMin = L[0];
						increment = L[0];
					}

					int rows = l[0];
					int cols = l[1];
					int k = 0;
					for (int j = jMin; j < rows * cols; j += rows, k++)
					{
						int indx = j + 2;
						if (j - jMin < rows)//first col
							l[indx] = i + 1;
						else
							l[indx] = indextemp[k + 1];
					}

					nbot = ntop + 1;

					//if (Lrows > 0)
					//{
					//	thrust::copy(thrust::seq, L, L + (Lrows * Lcols) - 1, l);
					//}
					L = (double*)malloc(sizeof(double) * (l[0] * l[1] + 2));
					for (int i = 0; i < (int)(l[0] * l[1]) + 2; i++)
						L[i] = l[i];
					Lrows = ntop;
					Lcols = d;
				}
			}
			for (int i = 0; i < (int)(L[0] * L[1]) + 2; i++)
				matrix[i] = L[i];

		}

		void Gaussian::GetN(int b, int d, double N[])
		{
			double *d_L = (double*)malloc(3 * sizeof(double));
			d_L[0] = 1;
			d_L[1] = 1;

			subnumber(b, d, d_L);
			int ch = d_L[0];
			//free(N);

			//N = (double*)malloc((2 + ch * d) * sizeof(double));
			N[0] = ch;
			N[1] = d;
			int idx = 2;
			for (int i = 0; i < ch; i++)
				for (int j = 0; j < d; j++, idx++)
				{
					//int idx = 2 + j + (i * ch);
					N[idx] = pow(2, d_L[idx]) + 1;
				}

		}
	}
}



