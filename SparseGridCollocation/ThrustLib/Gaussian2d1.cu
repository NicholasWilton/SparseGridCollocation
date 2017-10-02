#include "Gaussian2d1.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
using namespace Eigen;
using namespace thrust;

namespace Leicester
{
	namespace ThrustLib
	{
		Gaussian2d1::Gaussian2d1(MatrixXd testNodes, MatrixXd centralNodes)
		{
			this->rows = testNodes.rows();
			const double *h_testNodes = testNodes.data();
			device_vector<double> d_testNodes(h_testNodes, h_testNodes + (testNodes.rows() * testNodes.cols()));
			this->testNodes = d_testNodes;
			this->cols = centralNodes.rows();
			const double *h_centralNodes = centralNodes.data();
			device_vector<double> d_centralNodes(h_centralNodes, h_centralNodes + (centralNodes.rows() * centralNodes.cols()));
			this->centralNodes = d_centralNodes;
		}

		Gaussian2d1::Gaussian2d1(MatrixXd testNodes)
		{
			this->rows = testNodes.rows();
			const double *h_testNodes = testNodes.data();
			device_vector<double> d_testNodes(h_testNodes, h_testNodes + (testNodes.rows() * testNodes.cols()));
			this->testNodes = d_testNodes;
		}

		Gaussian2d1::~Gaussian2d1()
		{
			this->testNodes.clear();
			this->testNodes.shrink_to_fit();
		}

		vector<MatrixXd> Gaussian2d1::Gaussian2d(const MatrixXd & CN, const MatrixXd & A, const MatrixXd & C)
		{
			this->cols = CN.rows();
			const double *h_centralNodes = CN.data();
			device_vector<double> d_centralNodes(h_centralNodes, h_centralNodes + (CN.rows() * CN.cols()));
			this->centralNodes = d_centralNodes;
			return this->Gaussian2d(A, C);
		}

		vector<MatrixXd> Gaussian2d1::Gaussian2d(const MatrixXd & A, const MatrixXd & C)
		{
			cudaStream_t s1, s2, s3;
			cudaStreamCreate(&s1);
			cudaStreamCreate(&s2);
			cudaStreamCreate(&s3);

			const double *h_a = A.data();
			const double *h_c = C.data();

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
			device_vector<double> d_PHI2(rows * cols);
			device_vector<double> d_D(rows * cols);
			device_vector<double> d_Dt(rows * cols);
			device_vector<double> d_Dx(rows * cols);
			device_vector<double> d_Dxx(rows * cols);


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

			//thrust::transform(thrust::cuda::par.on(s2), d_PHI2.begin(), d_PHI2.end(), d_PHI2.begin(), 
			//	phi2_functor2(raw_pointer_cast(d_tp1.data()), A(0, 1), raw_pointer_cast(d_cn1.data()), C(0, 1), rows, cols));
			thrust::for_each(thrust::cuda::par.on(s2),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_PHI2.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_PHI2.end())),
				phi_functor3(raw_pointer_cast(d_tp1.data()), A(0, 1), raw_pointer_cast(d_cn1.data()), C(0, 1), rows, cols)
			);

			cudaStreamSynchronize(s1);
			cudaStreamSynchronize(s2);
			thrust::transform(thrust::cuda::par.on(s1), d_PHI1.begin(), d_PHI1.end(), d_PHI2.begin(), d_D.begin(),
				d_functor2());
			cudaStreamSynchronize(s1);

			//Calculate Dt
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
			cudaError_t e = cudaMemcpy(h_D, p_D, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_D returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			Eigen::Map<Eigen::MatrixXd> dataMapD(h_D, rows, cols);
			MatrixXd D = dataMapD.eval();

			cudaDeviceSynchronize();
			double *h_Dt = (double*)malloc(sizeof(double) * rows * cols);
			double *p_Dt = d_Dt.data().get();
			e = cudaMemcpy(h_Dt, p_Dt, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_Dt returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			Eigen::Map<Eigen::MatrixXd> dataMapDt(h_Dt, rows, cols);
			MatrixXd Dt = dataMapDt.eval();

			//cudaDeviceSynchronize();
			double *h_Dx = (double*)malloc(sizeof(double) * rows * cols);
			double *p_Dx = d_Dx.data().get();
			e = cudaMemcpy(h_Dx, p_Dx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_Dx returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			Eigen::Map<Eigen::MatrixXd> dataMapDx(h_Dx, rows, cols);
			MatrixXd Dx = dataMapDx.eval();

			//cudaDeviceSynchronize();
			double *h_Dxx = (double*)malloc(sizeof(double) * rows * cols);
			double *p_Dxx = d_Dxx.data().get();
			e = cudaMemcpy(h_Dxx, p_Dxx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (e != cudaSuccess)
				printf("cudaMemcpy h_Dxx returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			Eigen::Map<Eigen::MatrixXd> dataMapDxx(h_Dxx, rows, cols);
			MatrixXd Dxx = dataMapDxx.eval();

			return{ D, Dt, Dx, Dxx };

		}

		vector<MatrixXd> Gaussian2d1::Gaussian2d(const MatrixXd & TP, const MatrixXd & CN, const MatrixXd & A, const MatrixXd & C)
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

			return{ D, Dt, Dx, Dxx };
		}
	}
}