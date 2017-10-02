#include "GaussianNd1.h"
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <tchar.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
using namespace Eigen;
using namespace thrust;

namespace Leicester
{
	namespace ThrustLib
	{
		GaussianNd1::GaussianNd1(MatrixXd testNodes, MatrixXd centralNodes)
		{
			this->rows = testNodes.rows();
			const double *h_testNodes = testNodes.data();
			device_vector<double> d_testNodes(h_testNodes, h_testNodes + (testNodes.rows() * testNodes.cols()));
			this->testNodes = d_testNodes;
			this->cols = centralNodes.rows();
			const double *h_centralNodes = centralNodes.data();
			device_vector<double> d_centralNodes(h_centralNodes, h_centralNodes + (centralNodes.rows() * centralNodes.cols()));
			this->centralNodes = d_centralNodes;
			this->dimensions = testNodes.cols();
		}

		GaussianNd1::GaussianNd1(MatrixXd testNodes)
		{
			this->rows = testNodes.rows();
			const double *h_testNodes = testNodes.data();
			device_vector<double> d_testNodes(h_testNodes, h_testNodes + (testNodes.rows() * testNodes.cols()));
			this->testNodes = d_testNodes;
			this->dimensions = testNodes.cols();
		}

		GaussianNd1::~GaussianNd1()
		{
			this->testNodes.clear();
			this->testNodes.shrink_to_fit();
			this->centralNodes.clear();
			this->centralNodes.shrink_to_fit();
		}



		vector<MatrixXd> GaussianNd1::GaussianNd(const MatrixXd & CN, const MatrixXd & A, const MatrixXd & C)
		{
			this->cols = CN.rows();
			const double *h_centralNodes = CN.data();
			device_vector<double> d_centralNodes(h_centralNodes, h_centralNodes + (CN.rows() * CN.cols()));
			this->centralNodes = d_centralNodes;
			vector<MatrixXd> result = this->GaussianNd(A, C);
			return result;
		}

		void SignalHandler(int signal)
		{
			printf("Signal %d", signal);
			throw "!Access Violation!";
		}

		vector<MatrixXd> GaussianNd1::GaussianNd(const MatrixXd & A, const MatrixXd & C)
		{
			cudaStream_t s1, s2, s3;
			cudaStreamCreate(&s1);
			cudaStreamCreate(&s2);
			cudaStreamCreate(&s3);

			const double *h_a = A.data();
			const double *h_c = C.data();

			device_vector<double> d_a(h_a, h_a + (A.rows() * A.cols()));
			device_vector<double> d_c(h_c, h_c + (C.rows() * C.cols()));

			device_vector<double> d_D(rows * cols);
			device_vector<double> d_Dt(rows * cols);
			device_vector<double> d_Dx(rows * cols);
			device_vector<double> d_Dxx(rows * cols);

			cudaStreamSynchronize(s1);
			cudaStreamSynchronize(s2);
			counting_iterator<int> first(0);
			counting_iterator<int> last(rows * cols);
			
			thrust::for_each(thrust::cuda::par.on(s1),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_D.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_D.end())),
				phi_functor3Nd(raw_pointer_cast(testNodes.data()), raw_pointer_cast(d_a.data()), 
					raw_pointer_cast(centralNodes.data()), raw_pointer_cast(d_c.data()), rows, cols, dimensions)
			);

			cudaStreamSynchronize(s1);

			//Calculate Dt
			double scalarDt = -2 * (A(0, 0) / C(0, 0)) * (A(0, 0) / C(0, 0));
			thrust::for_each(thrust::cuda::par.on(s1),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_D.begin(), d_Dt.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_D.end(), d_Dt.end())),
				dt_functor3Nd(raw_pointer_cast(testNodes.data()), scalarDt, raw_pointer_cast(centralNodes.data()), rows, cols)
			);

			//Calculate Dx
			thrust::for_each(thrust::cuda::par.on(s2),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_D.begin(), d_Dx.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_D.end(), d_Dx.end())),
				dx_functor3Nd(raw_pointer_cast(testNodes.data()), raw_pointer_cast(d_a.data()), 
					raw_pointer_cast(centralNodes.data()), raw_pointer_cast(d_c.data()), rows, cols, dimensions)
			);
			//Calculate Dxx
			thrust::for_each(thrust::cuda::par.on(s3),
				thrust::make_zip_iterator(
					thrust::make_tuple(first, d_D.begin(), d_Dxx.begin())),
				thrust::make_zip_iterator(
					thrust::make_tuple(last, d_D.end(), d_Dxx.end())),
				dxx_functor3Nd(raw_pointer_cast(testNodes.data()), raw_pointer_cast(d_a.data()), raw_pointer_cast(d_c.data()),
					raw_pointer_cast(centralNodes.data()), rows, cols, dimensions)
			);
			cudaStreamSynchronize(s1);
			cudaStreamSynchronize(s2);
			cudaStreamSynchronize(s3);

			//cudaDeviceSynchronize();
			//double *h_Phi = (double*)malloc(sizeof(double) * rows * cols);
			//double *p_Phi = d_PHI.data().get();
			//cudaError_t e = cudaMemcpy(h_Phi, p_Phi, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			//if (e != cudaSuccess)
			//	printf("cudaMemcpy h_Phi returned error %s (code %d), line(%d) when copying%i\n",
			//		cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			//Eigen::Map<Eigen::MatrixXd> dataMapPhi(h_Phi, rows, cols);
			//MatrixXd Phi = dataMapPhi.eval();

			//cudaDeviceSynchronize();
			//double *h_Phi2 = (double*)malloc(sizeof(double) * rows * cols);
			//double *p_Phi2 = d_PHI2.data().get();
			//e = cudaMemcpy(h_Phi2, p_Phi2, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			//if (e != cudaSuccess)
			//	printf("cudaMemcpy h_Phi2 returned error %s (code %d), line(%d) when copying%i\n",
			//		cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
			//Eigen::Map<Eigen::MatrixXd> dataMapPhi2(h_Phi2, rows, cols);
			//MatrixXd Phi2 = dataMapPhi2.eval();

			typedef void(*SignalHandlerPointer)(int);

			SignalHandlerPointer previousHandler;
			previousHandler = signal(SIGSEGV, SignalHandler);

			cudaDeviceSynchronize();
			double *h_D = (double*)malloc(sizeof(double) * rows * cols);
			double *p_D = d_D.data().get();
			MatrixXd D(0, 0);
			cudaError_t e = cudaMemcpy(h_D, p_D, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (e != cudaSuccess)
			{
				printf("cudaMemcpy h_D returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
				//D(0, 0);
			}
			else
			{
				Eigen::Map<Eigen::MatrixXd> dataMapD(h_D, rows, cols);
				try
				{
					//D = dataMapD.eval();
					MatrixXd d = dataMapD.eval();
					D = d.replicate(1, 1);
				}
				catch (char *e)
				{
					printf("Exception Caught: %s\n", e);
				}
				catch (...)
				{
					printf("failed to copy D from h_D\r\n");
					D(0, 0);
				}
			}
			

			//cudaDeviceSynchronize();
			double *h_Dt = (double*)malloc(sizeof(double) * rows * cols);
			double *p_Dt = d_Dt.data().get();
			MatrixXd Dt(0,0);
			e = cudaMemcpy(h_Dt, p_Dt, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (e != cudaSuccess)
			{
				printf("cudaMemcpy h_Dt returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
				//Dt(0, 0);
			}
			else
			{
				Eigen::Map<Eigen::MatrixXd> dataMapDt(h_Dt, rows, cols);
				try
				{
					//Dt = dataMapDt.eval();
					MatrixXd dt = dataMapDt.eval();
					Dt = dt.replicate(1, 1);
				}
				catch (...)
				{
					printf("failed to copy Dt from h_Dt\r\n");
					Dt(0, 0);
				}
			}
			//cudaDeviceSynchronize();
			double *h_Dx = (double*)malloc(sizeof(double) * rows * cols);
			double *p_Dx = d_Dx.data().get();
			MatrixXd Dx;
			e = cudaMemcpy(h_Dx, p_Dx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (e != cudaSuccess)
			{
				printf("cudaMemcpy h_Dx returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
				//Dx(0, 0);
			}
			else
			{
				Eigen::Map<Eigen::MatrixXd> dataMapDx(h_Dx, rows, cols);
				try
				{
					//Dx = dataMapDx.eval();
					MatrixXd dx = dataMapDx.eval();
					Dx = dx.replicate(1,1);
				}
				catch (...)
				{
					printf("failed to copy Dx from h_Dx\r\n");
					Dx(0, 0);
				}
			}

			//cudaDeviceSynchronize();
			double *h_Dxx = (double*)malloc(sizeof(double) * rows * cols);
			double *p_Dxx = d_Dxx.data().get();
			MatrixXd Dxx(0, 0);
			e = cudaMemcpy(h_Dxx, p_Dxx, sizeof(double) * rows * cols, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			if (e != cudaSuccess)
			{
				printf("cudaMemcpy h_Dxx returned error %s (code %d), line(%d) when copying%i\n",
					cudaGetErrorString(e), e, __LINE__, sizeof(double) * rows * cols);
				//Dxx(0, 0);
			}
			else
			{
				Eigen::Map<Eigen::MatrixXd> dataMapDxx(h_Dxx, rows, cols);
				try
				{
					//Dxx = dataMapDxx.eval();
					MatrixXd dxx = dataMapDxx.eval();
					Dxx = dxx.replicate(1, 1);
				}
				catch (...)
				{
					printf("failed to copy Dxx from h_Dxx\r\n");
					Dxx(0, 0);
				}
			}

			delete h_D;
			delete h_Dt;
			delete h_Dx;
			delete h_Dxx;
			
			d_a.clear();
			d_a.shrink_to_fit();
			d_c.clear();
			d_c.shrink_to_fit();

			d_D.clear();
			d_D.shrink_to_fit();
			d_Dt.clear();
			d_Dt.shrink_to_fit();
			d_Dx.clear();
			d_Dx.shrink_to_fit();
			d_Dxx.clear();
			d_Dxx.shrink_to_fit();

			cudaStreamDestroy(s1);
			cudaStreamDestroy(s2);
			cudaStreamDestroy(s3);

			return{ D, Dt, Dx, Dxx };
			

		}


		
	}
}