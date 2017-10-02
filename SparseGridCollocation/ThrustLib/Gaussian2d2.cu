#include "Gaussian2d2.h"

namespace Leicester
{
	namespace ThrustLib
	{
		Gaussian2d2::Gaussian2d2()
		{
		}

		Gaussian2d2::~Gaussian2d2()
		{
			//this->testNodes.clear();
			//this->testNodes.shrink_to_fit();
			//this->centralNodes.clear();
			//this->centralNodes.shrink_to_fit();
		}

		Gaussian2d2::Gaussian2d2(double tLower, double tUpper, const double* N)
		{
			cudaStream_t s1, s2;
			cudaStreamCreate(&s1);
			cudaStreamCreate(&s2);

			double length = N[0] * N[1];
			this->rows = length;
			this->cols = length;
			
			int sizeTestNodes = (2 + 2 * length) * sizeof(double);
			double * h_TXYZ = (double*)malloc(sizeTestNodes);
			double * d_TXYZ;
			cudaMalloc((void **)&d_TXYZ, sizeTestNodes);
			//pinnedVector ph_testNodes(h_TXYZ, h_TXYZ + sizeTestNodes);
			//device_ptr<double> dv_tp;
			//double* d_tp = thrust::raw_pointer_cast(dv_tp);
			//double* h_tp = thrust::raw_pointer_cast(ph_testNodes.data());
			cudaMemcpyAsync(d_TXYZ, h_TXYZ, sizeTestNodes, cudaMemcpyHostToDevice, s1);
			cudaStreamSynchronize(s1);
			cudaError_t e = cudaPeekAtLastError();

			int sizeN = 4 * sizeof(double);
			pinnedVector ph_N(N, N + sizeN);
			device_vector<double> dv_N(sizeN);
			cudaMemcpyAsync(thrust::raw_pointer_cast(dv_N.data()), thrust::raw_pointer_cast(ph_N.data()),
				dv_N.size() * sizeof(double), cudaMemcpyHostToDevice, s2);


			cudaStreamSynchronize(s2);
			e = cudaPeekAtLastError();
			cudaDeviceSynchronize();
			
			//double* d_TP = thrust::raw_pointer_cast(dv_tp);
			
			GenerateTestNodes << <1, 1 >> > (tLower, tUpper, thrust::raw_pointer_cast(dv_N.data()), d_TXYZ);

			device_ptr<double> dv_tp = thrust::device_pointer_cast<double>(d_TXYZ);
			double * d_CN;
			cudaMalloc((void **)&d_CN, sizeTestNodes);
			
			//thrust::copy(thrust::cuda::par.on(s1), dv_tp, dv_tp + 2 + (2*length), dv_cn);
			cudaMemcpyAsync(d_CN, d_TXYZ, sizeTestNodes, cudaMemcpyHostToDevice, s1);

			cudaStreamSynchronize(s1);
			e = cudaPeekAtLastError();
			cudaDeviceSynchronize();
			
			device_ptr<double> dv_cn(d_CN);
			
			this->testNodes = dv_tp;
			this->centralNodes = dv_cn;
		}


		typedef thrust::host_vector<double, thrust::cuda::experimental::pinned_allocator<double>> pinnedVector;

		vector<MatrixXd> Gaussian2d2::Gaussian2d(const MatrixXd & CN, const MatrixXd & A, const MatrixXd & C)
		{
			//cudaStream_t s1, s2;
			//cudaStreamCreate(&s1);
			//cudaStreamCreate(&s2);
			//
			//double length = N[3] * N[2];
			//this->cols = length;

			//int sizeCentralNodes = 2 * length * sizeof(double);
			//double * h_TXYZ = (double*)malloc(sizeCentralNodes);
			//pinnedVector ph_centralNodes(h_TXYZ, h_TXYZ + sizeCentralNodes);
			//device_vector<double> dv_cn(sizeCentralNodes);
			//cudaMemcpyAsync(thrust::raw_pointer_cast(dv_cn.data()), thrust::raw_pointer_cast(ph_centralNodes.data()),
			//	dv_cn.size() * sizeof(double), cudaMemcpyHostToDevice, s1);

			//int sizeN = 4 * sizeof(double);
			//pinnedVector ph_N(N, N + sizeN);
			//device_vector<double> dv_N(sizeN);
			//cudaMemcpyAsync(thrust::raw_pointer_cast(dv_N.data()), thrust::raw_pointer_cast(ph_N.data()),
			//	dv_N.size() * sizeof(double), cudaMemcpyHostToDevice, s2);

			//cudaStreamSynchronize(s1);
			//cudaStreamSynchronize(s2);
			////cudaDeviceSynchronize();

			//double* d_CN = thrust::raw_pointer_cast(dv_cn.data());

			//GenerateTestNodes << <1, 1 >> > (tLower, tUpper, thrust::raw_pointer_cast(dv_N.data()), d_CN);

			//printf("CN:\r\n");
			//host_vector<double> h_cn(dv_cn.size());
			//thrust::copy(dv_cn.begin(), dv_cn.end(), h_cn.begin());
			//double *raw_cn = thrust::raw_pointer_cast(h_cn.data());
			//Utility::printMatrix(raw_cn, dim3(2, 15));

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

			//device_vector<double> dv_centralNodes(dp_centralNodes, dp_centralNodes + (CN.rows() * CN.cols()));
			this->centralNodes = dp_centralNodes;

			return Gaussian2d(A, C);

		}

		vector<MatrixXd> Gaussian2d2::Gaussian2d(const MatrixXd & A, const MatrixXd & C)
		{
			cudaStream_t s1, s2, s3, s4;
			cudaStreamCreate(&s1);
			cudaStreamCreate(&s2);
			cudaStreamCreate(&s3);
			cudaStreamCreate(&s4);

			const double *h_a = A.data();
			const double *h_c = C.data();
			device_vector<double> d_tp(this->testNodes, this->testNodes + 2 + (rows * 2));
			device_vector<double> d_cn(this->centralNodes, this->centralNodes + 2 + (cols * 2));

			device_vector<double> d_tp0(rows);
			device_vector<double> d_cn0(cols);
			thrust::copy(thrust::cuda::par.on(s1), d_tp.begin() + 2, d_tp.begin() + 2 + rows, d_tp0.begin()); //first column
			thrust::copy(thrust::cuda::par.on(s1), d_cn.begin() + 2, d_cn.begin() + 2 + cols, d_cn0.begin()); //first column

			device_vector<double> d_tp1(rows);
			device_vector<double> d_cn1(rows);
			thrust::copy(thrust::cuda::par.on(s2), d_tp.begin() + 2 + rows, d_tp.begin() + 2 + (2 * rows), d_tp1.begin()); //second column
			thrust::copy(thrust::cuda::par.on(s2), d_cn.begin() + 2 + cols, d_cn.begin() + 2 + (2 * cols), d_cn1.begin()); //second column

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

			return{ D, Dt, Dx, Dxx };

		}
	}
}