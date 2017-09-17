#include "Gaussian2d.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
using namespace Eigen;
using namespace thrust;


namespace Leicester
{
	namespace ThrustLib
	{


		struct phi_functor
		{
			const double a;
			const double cn;
			const double c;
			phi_functor(double _a, double _cn, double _c) : a(_a), cn(_cn), c(_c) {}
			__device__
				double operator()(const double &TP)
			{
				double a1 = a * (TP - cn);
				double b1 = -(a1 * a1) / (c*c);
				double e1 = expm1(b1) + 1;
				return e1;
			}
		};



		struct vectorAddScalar_functor
		{
			const double a;

			vectorAddScalar_functor(double _a) : a(_a) {}
			__device__
				double operator()(const double &TP)
			{
				return TP + a;
			}
		};

		struct scalarVectorDifference_functor
		{
			const double a;
			const double cn;

			scalarVectorDifference_functor(double _a, double _cn) : a(_a), cn(_cn) {}
			__device__
				double operator()(const double &TP)
			{
				return a * (TP - cn);
			}
		};

		struct vectorScalarDifference_functor
		{
			const double a;

			vectorScalarDifference_functor(double _a) : a(_a) {}
			__device__
				double operator()(const double &TP)
			{
				return TP - a;
			}
		};

		struct vectorScalarAddition_functor
		{
			const double a;

			vectorScalarAddition_functor(double _a) : a(_a) {}
			__device__
				double operator()(const double &TP)
			{
				return TP + a;
			}
		};

		struct vectorScalarMultiply_functor
		{
			const double a;

			vectorScalarMultiply_functor(double _a) : a(_a) {}
			__device__
				double operator()(const double &TP)
			{
				return TP * a;
			}
		};

		void CopyToMatrix(MatrixXd &m, double* buffer, dim3 size)
		{
			int ptr = 0;
			for (int i = 0; i < size.x; i++)
				for (int j = 0; j < size.y; j++)
					m(i, j) = buffer[ptr];
		}

		struct print
		{
			__host__ __device__
				void operator()(const double &element)
			{
				int i = blockDim.y * blockIdx.y + threadIdx.y;
				int j = blockDim.x * blockIdx.x + threadIdx.x;
				printf("i=%i, j=%i, value=%f\r\n", i, j, element);
			}
		};

		struct dt_functor
		{
			const double a;
			const double cn;

			dt_functor(double _a, double _cn) : a(_a), cn(_cn) {}
			__device__
				double operator()(const double &TP, const double &D)
			{
				return a * (TP - cn) * D;
			}
		};

		struct dx_functor
		{
			const double a;
			const double cn;

			dx_functor(double _a, double _cn) : a(_a), cn(_cn) {}
			__device__
				double operator()(const double &TP, const double &D)
			{
				return a * (TP - cn) * D;
			}
		};

		struct dxx_functor
		{
			const double a;
			const double b;
			const double cn;

			dxx_functor(double _a, double _b, double _cn) : a(_a), b(_b), cn(_cn) {}
			__device__
				double operator()(const double &TP, const double &D)
			{
				double dTpCn = TP - cn;
				double sDTpCn = dTpCn * dTpCn;
				double a1 = a * sDTpCn;
				double b1 = a1 + b;
				double c1 = b1 * D;
				double sTp = TP * TP;
				return c1 * sTp;
			}
		};

		struct phi1_functor2
		{
			const double a;
			double* cn;
			const double c;
			//thrust::device_ptr<double> tp;
			double * tp;
			int rows;
			int cols;
			int iMax = 0;
			int jMax = 0;
			phi1_functor2(double* _tp, double _a, double* _cn, double _c, int _rows, int _cols)
				: a(_a), c(_c), rows(_rows), cols(_cols) {
				tp = _tp;
				cn = _cn;
			}
			__device__
				double operator()(const double &Phi)
			{
				int i = blockDim.y * blockIdx.y + threadIdx.y;
				int j = blockDim.x * blockIdx.x + threadIdx.x;
				int idx = i + (j * blockDim.y);
				int tpIdx = idx % rows;
				int cnIdx = idx / rows;

				double a1 = a * (tp[tpIdx] - cn[cnIdx]);
				double b1 = -(a1 * a1) / (c*c);
				double e1 = expm1(b1) + 1;
				return e1;
			}
		};

		struct phi2_functor2
		{
			int count = 0;
			const double a;
			double* cn;
			const double c;
			//thrust::device_ptr<double> tp;
			double * tp;
			int rows;
			int cols;
			int iMax = 0;
			int jMax = 0;
			phi2_functor2(double* _tp, double _a, double* _cn, double _c, int _rows, int _cols)
				: a(_a), c(_c), rows(_rows), cols(_cols) {
				tp = _tp;
				cn = _cn;
			}
			__device__
				double operator()(const double &Phi)
			{
				int i = blockDim.y * blockIdx.y + threadIdx.y;
				int j = blockDim.x * blockIdx.x + threadIdx.x;
				int k = blockDim.z * blockIdx.z + threadIdx.z;
				if (i > iMax)
				{
					iMax = i;
					printf("max i=%i,", iMax);
				}

				int idx = i + (j * blockDim.y);
				int tpIdx = idx % rows;
				int cnIdx = idx / rows;


				double a1 = a * (tp[tpIdx] - cn[cnIdx]);
				double b1 = -(a1 * a1) / (c*c);
				double e1 = expm1(b1) + 1;
				if (j >= 7679)
				{
					count++;
					printf("i=%i, j=%i, k=%i idx=%i, tpIdx=%i, tp[%i]=%f, cn[%i]=%f Phi=%f count=%i\r\n", i, j, k, idx, tpIdx, tpIdx, tp[tpIdx], cnIdx, cn[cnIdx], Phi, count);
				}
				return e1;
			}
		};

		struct d_functor2
		{
			__device__
				double operator()(const double &Phi1, const double &Phi2)
			{

				return  Phi1 * Phi2;
			}
		};

		struct dt_functor2
		{
			const double a;
			double* cn;
			double* tp;
			int rows;
			int cols;

			dt_functor2(double* _tp, double _a, double* _cn, int _rows, int _cols)
				: tp(_tp), a(_a), cn(_cn), rows(_rows), cols(_cols) {}
			__device__
				double operator()(const double &Dt, const double &D)
			{
				int i = blockDim.y * blockIdx.y + threadIdx.y;
				int j = blockDim.x * blockIdx.x + threadIdx.x;
				int idx = i + (j * blockDim.y);
				int tpIdx = idx % rows;
				int cnIdx = idx / rows;

				return a * (tp[tpIdx] - cn[cnIdx]) * D;
			}
		};

		struct dx_functor2
		{
			const double a;
			double* cn;
			double* tp;
			int rows;
			int cols;

			dx_functor2(double* _tp, double _a, double* _cn, int _rows, int _cols)
				:tp(_tp), a(_a), cn(_cn), rows(_rows), cols(_cols) {
			}
			__device__
				double operator()(const double &Dx, const double &D)
			{
				int i = blockDim.y * blockIdx.y + threadIdx.y;
				int j = blockDim.x * blockIdx.x + threadIdx.x;
				int idx = i + (j * blockDim.y);
				int tpIdx = idx % rows;
				int cnIdx = idx / rows;

				return tp[tpIdx] * a * (tp[tpIdx] - cn[cnIdx]) * D;
			}
		};

		struct dxx_functor2
		{
			const double a;
			const double b;
			double *cn;
			double *tp;
			int rows;
			int cols;

			dxx_functor2(double* _tp, double _a, double _b, double* _cn, int _rows, int _cols)
				: tp(_tp), a(_a), b(_b), cn(_cn), rows(_rows), cols(_cols) {}
			__device__
				double operator()(const double &Dxx, const double &D)
			{
				int i = blockDim.y * blockIdx.y + threadIdx.y;
				int j = blockDim.x * blockIdx.x + threadIdx.x;
				int idx = i + (j * blockDim.y);
				int tpIdx = idx % rows;
				int cnIdx = idx / rows;

				double dTpCn = tp[tpIdx] - cn[cnIdx];
				double sDTpCn = dTpCn * dTpCn;
				double a1 = a * sDTpCn;
				double b1 = a1 + b;
				double c1 = b1 * D;
				double sTp = tp[tpIdx] * tp[tpIdx];
				return c1 * sTp;
			}
		};


		struct phi_functor3
		{
			const double a;
			double* cn;
			const double c;
			double * tp;
			int rows;
			int cols;

			phi_functor3(double* _tp, double _a, double* _cn, double _c, int _rows, int _cols)
				: a(_a), c(_c), rows(_rows), cols(_cols) {
				tp = _tp;
				cn = _cn;
			}
			template<typename Tuple> __device__
				void operator()(Tuple t)
			{
				int idx = (int)thrust::get<0>(t);
				int tpIdx = idx % rows;
				int cnIdx = idx / rows;

				double a1 = a * (tp[tpIdx] - cn[cnIdx]);
				double b1 = -(a1 * a1) / (c*c);
				double e1 = expm1(b1) + 1;

				//printf("idx=%i, tpIdx=%i, tp[%i]=%f, cn[%i]=%f\r\n", idx, tpIdx, tpIdx, tp[tpIdx], cnIdx, cn[cnIdx]);

				thrust::get<1>(t) = e1;
			}
		};

		struct dt_functor3
		{
			const double a;
			double* cn;
			double* tp;
			int rows;
			int cols;

			dt_functor3(double* _tp, double _a, double* _cn, int _rows, int _cols)
				: tp(_tp), a(_a), cn(_cn), rows(_rows), cols(_cols) {}
			template<typename Tuple> __device__
				void operator()(Tuple t)
			{
				int idx = (int)thrust::get<0>(t);
				double D = (double)thrust::get<1>(t);
				int tpIdx = idx % rows;
				int cnIdx = idx / rows;
				double b = a * (tp[tpIdx] - cn[cnIdx]) * D;
				//if(idx < 15)
				//	printf("idx=%i, tpIdx=%i, tp[%i]=%f, cn[%i]=%f b=%f a=%f D=%f \r\n", idx, tpIdx, tpIdx, tp[tpIdx], cnIdx, cn[cnIdx], b, a, D);
				thrust::get<2>(t) = b;
			}
		};

		struct dx_functor3
		{
			const double a;
			double* cn;
			double* tp;
			int rows;
			int cols;

			dx_functor3(double* _tp, double _a, double* _cn, int _rows, int _cols)
				:tp(_tp), a(_a), cn(_cn), rows(_rows), cols(_cols) {
			}
			template<typename Tuple> __device__
				void operator()(Tuple t)
			{
				int idx = (int)thrust::get<0>(t);
				double D = (double)thrust::get<1>(t);
				int tpIdx = idx % rows;
				int cnIdx = idx / rows;

				double b = tp[tpIdx] * a * (tp[tpIdx] - cn[cnIdx]) * D;
				thrust::get<2>(t) = b;
			}
		};

		struct dxx_functor3
		{
			const double a;
			const double b;
			double *cn;
			double *tp;
			int rows;
			int cols;

			dxx_functor3(double* _tp, double _a, double _b, double* _cn, int _rows, int _cols)
				: tp(_tp), a(_a), b(_b), cn(_cn), rows(_rows), cols(_cols) {}
			template<typename Tuple> __device__
				void operator()(Tuple t)
			{
				int idx = (int)thrust::get<0>(t);
				double D = (double)thrust::get<1>(t);
				int tpIdx = idx % rows;
				int cnIdx = idx / rows;

				double dTpCn = tp[tpIdx] - cn[cnIdx];
				double sDTpCn = dTpCn * dTpCn;
				double a1 = a * sDTpCn;
				double b1 = a1 + b;
				double c1 = b1 * D;
				double sTp = tp[tpIdx] * tp[tpIdx];
				double d = c1 * sTp;
				thrust::get<2>(t) = d;
			}
		};

		typedef double mytype;
		typedef thrust::host_vector<mytype, thrust::cuda::experimental::pinned_allocator<mytype>> pinnedVector;


		Gaussian::Gaussian(MatrixXd testNodes, MatrixXd centralNodes)
		{
			cudaStream_t s1, s2;
			cudaStreamCreate(&s1);
			cudaStreamCreate(&s2);
			
			this->rows = testNodes.rows();
			
			int sizeTestNodes = sizeof(double) * testNodes.rows() * testNodes.cols();
			const double *h_testNodes = testNodes.data();
			//double *d_testNodes;
			//cudaMalloc((void**)&d_testNodes, sizeTestNodes);
			//cudaMemcpyAsync(d_testNodes, h_testNodes, sizeTestNodes, cudaMemcpyKind::cudaMemcpyHostToDevice, s1);

			pinnedVector ph_testNodes(h_testNodes, h_testNodes + sizeTestNodes);
			device_vector<mytype> pd_testNodes(sizeTestNodes);
			cudaMemcpyAsync(thrust::raw_pointer_cast(pd_testNodes.data()), thrust::raw_pointer_cast(ph_testNodes.data()), 
				pd_testNodes.size() * sizeof(mytype), cudaMemcpyHostToDevice, s1);
			this->testNodes = pd_testNodes;

			int sizeCentralNodes = sizeof(double) * centralNodes.rows() * centralNodes.cols();
			const double *h_centralNodes = centralNodes.data();
			pinnedVector ph_centralNodes(h_centralNodes, h_centralNodes + sizeCentralNodes);
			device_vector<mytype> pd_centralNodes(sizeCentralNodes);
			cudaMemcpyAsync(thrust::raw_pointer_cast(pd_centralNodes.data()), thrust::raw_pointer_cast(ph_centralNodes.data()),
				pd_centralNodes.size() * sizeof(mytype), cudaMemcpyHostToDevice, s2);
			this->centralNodes = pd_centralNodes;
			
			
			this->cols = centralNodes.rows();
			//const double *h_centralNodes = centralNodes.data();
			//double *d_centralNodes;
			//int sizeCentralNodes = sizeof(double) * centralNodes.rows() * centralNodes.cols();
			//cudaMalloc((void**)&d_centralNodes, sizeCentralNodes);
			//cudaMemcpyAsync(d_centralNodes, h_centralNodes, sizeCentralNodes, cudaMemcpyKind::cudaMemcpyHostToDevice, s2);
			
			//cudaStreamSynchronize(s1);
			//cudaStreamSynchronize(s2);
			cudaDeviceSynchronize();

			//device_vector<double> dv_testNodes(d_testNodes, d_testNodes + sizeTestNodes);
			////device_vector<double> dv_testNodes(h_testNodes, h_testNodes + (testNodes.rows() * testNodes.cols()));
			//this->testNodes = dv_testNodes;

			//device_vector<double> dv_centralNodes(d_centralNodes, d_centralNodes + sizeCentralNodes);
			//device_vector<double> dv_centralNodes(h_centralNodes, h_centralNodes + (centralNodes.rows() * centralNodes.cols()));
			//this->centralNodes = dv_centralNodes;
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
			//double *d_testNodes;
			int sizeTestNodes = sizeof(double) * testNodes.rows() * testNodes.cols();
			//cudaMemcpyAsync(d_testNodes, h_testNodes, sizeTestNodes, cudaMemcpyKind::cudaMemcpyHostToDevice, s1);
			pinnedVector ph_testNodes(h_testNodes, h_testNodes + sizeTestNodes);
			device_vector<mytype> pd_testNodes(sizeTestNodes);
			cudaMemcpyAsync(thrust::raw_pointer_cast(pd_testNodes.data()), thrust::raw_pointer_cast(ph_testNodes.data()),
				pd_testNodes.size() * sizeof(mytype), cudaMemcpyHostToDevice, s1);
			this->testNodes = pd_testNodes;
			cudaDeviceSynchronize();
		}

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

		vector<MatrixXd> Gaussian::Gaussian2d(const MatrixXd & CN, const MatrixXd & A, const MatrixXd & C)
		{

			/*this->cols = CN.rows();
			const double *h_centralNodes = CN.data();
			device_vector<double> d_centralNodes(h_centralNodes, h_centralNodes + (CN.rows() * CN.cols()));
			this->centralNodes = d_centralNodes;*/
			cudaStream_t s1;
			cudaStreamCreate(&s1);
			this->cols = CN.rows();
			//const double *h_centralNodes = CN.data();
			//double *d_centralNodes;
			//int sizeCentralNodes = sizeof(double) * CN.rows() * CN.cols();
			//cudaMemcpyAsync(d_centralNodes, h_centralNodes, sizeCentralNodes, cudaMemcpyKind::cudaMemcpyHostToDevice, s1);
			//device_vector<double> dv_centralNodes(d_centralNodes, d_centralNodes + sizeCentralNodes);
			//this->centralNodes = dv_centralNodes;
			
			int sizeCentralNodes = sizeof(double) * CN.rows() * CN.cols();
			const double *h_centralNodes = CN.data();
			pinnedVector ph_centralNodes(h_centralNodes, h_centralNodes + sizeCentralNodes);
			device_vector<mytype> pd_centralNodes(sizeCentralNodes);
			cudaMemcpyAsync(thrust::raw_pointer_cast(pd_centralNodes.data()), thrust::raw_pointer_cast(ph_centralNodes.data()),
				pd_centralNodes.size() * sizeof(mytype), cudaMemcpyHostToDevice, s1);
			this->centralNodes = pd_centralNodes;
			cudaDeviceSynchronize();
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
	}
}



