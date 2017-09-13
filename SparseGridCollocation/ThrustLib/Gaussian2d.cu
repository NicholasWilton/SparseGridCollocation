#include "Gaussian2d.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
using namespace Eigen;
using namespace thrust;



void printMatrix(const double *matrix, dim3 dimMatrix)
{
	int mSize = sizeof(matrix);

	printf("printing matrix data=");
	for (int x = 0; x < dimMatrix.x * dimMatrix.y; x++)
		printf("%f,", matrix[x]);
	printf("\r\n");
	printf("rows=%i cols=%i\r\n", dimMatrix.y, dimMatrix.x);

	for (int y = 0; y < dimMatrix.y; y++)
	{
		for (int x = 0; x < dimMatrix.x; x++)
		{
			int idx = (x * dimMatrix.y) + y;
			printf("%.16f ", matrix[idx]);
		}
		printf("\r\n");
	}
}

wstring printMatrix(MatrixXd m)
{
	int cols = m.cols();
	int rows = m.rows();

	wstringstream ss;
	ss << setprecision(25);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			double d = m(i, j);
			ss << d << "\t";

		}
		ss << "\r\n";
	}

	return ss.str();
}

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
	
	scalarVectorDifference_functor(double _a, double _cn) : a(_a), cn(_cn){}
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

vector<MatrixXd> ThrustLib::Gaussian::Gaussian2d(const MatrixXd & TP, const MatrixXd & CN, const MatrixXd & A, const MatrixXd & C)
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
	
	for (int i = 0; i < rows; i++)
	{
		//if (i == 0)
		//{
		//	d_test1.insert(d_test1.begin(), d_tp0.begin(), d_tp0.end());
		//	d_test2.insert(d_test2.begin(), d_tp1.begin(), d_tp1.end());
		//}
		//Calculate Phi1 & Phi2
		thrust::device_vector<double> phi1(rows);
		thrust::transform(d_tp0.begin(), d_tp0.end(), phi1.begin(), phi_functor(A(0,0), CN(i,0), C(0,0)));
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

		//Calculate Dx
		double scalarDx = -2 * ((A(0,1) / C(0,1)) * (A(0, 1) / C(0, 1)));
		thrust::device_vector<double> a2(rows);
		thrust::transform(d_tp1.begin(), d_tp1.end(), a2.begin(), scalarVectorDifference_functor(scalarDx, CN(i, 1)));

		thrust::device_vector<double> b2(rows);
		thrust::transform(phi1.begin(), phi1.end(), a2.begin(), b2.begin(), thrust::multiplies<double>());

		thrust::device_vector<double> c2(rows);
		thrust::transform(phi2.begin(), phi2.end(), b2.begin(), c2.begin(), thrust::multiplies<double>());

		thrust::device_vector<double> dx(rows);
		thrust::transform(d_tp1.begin(), d_tp1.end(), c2.begin(), dx.begin(), thrust::multiplies<double>());
		d_Dx.insert(d_Dx.begin(), dx.begin(), dx.end());

		//Calculate Dxx
		double sA = A(0,1) * A(0, 1);
		double qA = sA * sA;
		double sC = C(0, 1) * C(0, 1);
		double qC = sC * sC;

		thrust::device_vector<double> dTpCn(rows);
		thrust::transform(d_tp1.begin(), d_tp1.end(), dTpCn.begin(), vectorScalarDifference_functor(CN(0,1)));
		
		thrust::device_vector<double> sDTpCn(rows);
		thrust::transform(dTpCn.begin(), dTpCn.end(), dTpCn.begin(), sDTpCn.begin(), thrust::multiplies<double>());
		
		thrust::device_vector<double> a3(rows);
		double scalarDxx1 = 4 * qA / qC;
		
		thrust::transform(sDTpCn.begin(), sDTpCn.end(), a3.begin(), vectorScalarMultiply_functor(scalarDxx1));
		d_test1.insert(d_test1.begin(), a3.begin(), a3.end());
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

		if (i == 1)
		{
			//d_test1.insert(d_test1.begin(), d_tp1.begin(), d_tp1.end());
			d_test2.insert(d_test2.begin(), d_Dxx.begin(), d_Dxx.end());
		}
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
	//host_vector<double> h_D(d_D.size());
	//thrust::copy(d_D.begin(), d_D.end(), h_D.begin());
	//double *raw_D = thrust::raw_pointer_cast(h_D.data());
	//printMatrix(raw_D, dim3(1, rows));

	//printf("Dt:\r\n");
	//host_vector<double> h_Dt(d_Dt.size());
	//thrust::copy(d_Dt.begin(), d_Dt.end(), h_Dt.begin());
	//double *raw_Dt = thrust::raw_pointer_cast(h_Dt.data());
	//printMatrix(raw_Dt, dim3(1, rows));

	//printf("Dx:\r\n");
	//host_vector<double> h_Dx(d_Dx.size());
	//thrust::copy(d_Dx.begin(), d_Dx.end(), h_Dx.begin());
	//double *raw_Dx = thrust::raw_pointer_cast(h_Dx.data());
	//printMatrix(raw_Dx, dim3(1, rows));

	//printf("Dxx:\r\n");
	//host_vector<double> h_Dxx(d_Dxx.size());
	//thrust::copy(d_Dxx.begin(), d_Dxx.end(), h_Dxx.begin());
	//double *raw_Dxx = thrust::raw_pointer_cast(h_Dxx.data());
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

	return {D, Dt, Dx, Dxx};
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

MatrixXd GetTX2()
{
	MatrixXd TX1(15, 2);

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
	return TX1;
}

int main()
{
	MatrixXd TX1 = GetTX2();
	MatrixXd CN = GetTX2();
	MatrixXd C(1, 2);
	MatrixXd A(1, 2);
	C << 1.73, 600;
	//A << 2, 64;
	A << 2, 4;
	MatrixXd D(TX1.rows(), TX1.rows());
	for (int i = 0; i < 1; i++)
	{
		printf("i=%i", i);
		ThrustLib::Gaussian::Gaussian2d(TX1, CN, A, C);
	}

	return 0;
}
