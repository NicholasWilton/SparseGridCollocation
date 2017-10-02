#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <iostream>

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

struct phi_functor3Nd
{
	double* a;
	double* cn;
	double *c;
	double * tp;
	int rows;
	int cols;
	int dimensions;

	phi_functor3Nd(double* _tp, double* _a, double* _cn, double* _c, int _rows, int _cols, int _dimensions)
		: rows(_rows), cols(_cols), dimensions(_dimensions){
		tp = _tp;
		cn = _cn;
		a = _a;
		c = _c;
	}
	template<typename Tuple> __device__
		void operator()(Tuple t)
	{
		int idx = (int)thrust::get<0>(t);
		
		double product = 1.0;
		for (int dimension = 0; dimension < dimensions; dimension++)
		{
			int tpIdx = (idx % rows) + (dimension * rows);
			int cnIdx = (idx / rows) + (dimension * cols);

			double a1 = a[dimension] * (tp[tpIdx] - cn[cnIdx]);
			double b1 = -(a1 * a1) / (c[dimension] * c[dimension]);
			double e1 = expm1(b1) + 1;
			product *= e1;
			//if (idx < 27)
			//	printf("rows=%i, cols=%i, dimensions=%i, idx=%i, tpIdx=%i, tp[%i]=%f, cn[%i]=%f a[%i]=%f c[%i]=%f dimension=%i product=%f\r\n", rows, cols, dimensions, idx, tpIdx, tpIdx, tp[tpIdx], cnIdx, cn[cnIdx], dimension, a[dimension], dimension, c[dimension], dimension, product);
		}
		

		thrust::get<1>(t) = product;
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

struct dt_functor3Nd
{
	const double a;
	double* cn;
	double* tp;
	int rows;
	int cols;

	dt_functor3Nd(double* _tp, double _a, double* _cn, int _rows, int _cols)
		: tp(_tp), a(_a), cn(_cn), rows(_rows), cols(_cols) {}
	template<typename Tuple> __device__
		void operator()(Tuple t)
	{
		int idx = (int)thrust::get<0>(t);

		double D = (double)thrust::get<1>(t);
		int tpIdx = idx % rows;
		int cnIdx = idx / rows;
		double b = a * (tp[tpIdx] - cn[cnIdx]) * D;
		//if (idx == 27)
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

struct dx_functor3Nd
{
	double* a;
	double* c;
	double* cn;
	double* tp;
	int rows;
	int cols;
	int dimensions;

	dx_functor3Nd(double* _tp, double* _a, double* _cn, double* _c, int _rows, int _cols, int _dimensions)
		:tp(_tp), a(_a), cn(_cn), c(_c), rows(_rows), cols(_cols), dimensions(_dimensions) {
	}
	template<typename Tuple> __device__
		void operator()(Tuple t)
	{
		int idx = (int)thrust::get<0>(t);
		double D = (double)thrust::get<1>(t);
		double sum = 0;
		for (int dimension = 1; dimension < dimensions; dimension++)
		{
			//int tpIdx = idx % rows;
			//int cnIdx = idx / rows;
			int tpIdx = (idx % rows) + (dimension * rows);
			int cnIdx = (idx / rows) + (dimension * cols);
			double scalarDx = -2 * ((a[dimension] / c[dimension]) * ( a[dimension] / c[dimension]));
			double b = tp[tpIdx] * scalarDx * (tp[tpIdx] - cn[cnIdx]) * D;
			sum += b;
			//if (idx == 0)
				//printf("rows=%i, cols=%i, dimensions=%i, idx=%i, tpIdx=%i, tp[%i]=%f, cn[%i]=%f a[%i]=%f c[%i]=%f dimension=%i sum=%f\r\n", rows, cols, dimensions, idx, tpIdx, tpIdx, tp[tpIdx], cnIdx, cn[cnIdx], dimension, a[dimension], dimension, c[dimension], dimension, sum);
		}
		thrust::get<2>(t) = sum;
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

struct dxx_functor3Nd
{
	double *a;
	double *c;
	double *cn;
	double *tp;
	int rows;
	int cols;
	int dimensions;

	dxx_functor3Nd(double* _tp, double* _a, double* _c, double* _cn, int _rows, int _cols, int _dimensions)
		: tp(_tp), a(_a), c(_c), cn(_cn), rows(_rows), cols(_cols), dimensions(_dimensions) {}
	template<typename Tuple> __device__
		void operator()(Tuple t)
	{
		int idx = (int)thrust::get<0>(t);
		double D = (double)thrust::get<1>(t);

		double sumij = 0;
		for (int dimension = 1; dimension < dimensions; dimension++)
		{
			double sumi = 0;
			for (int i = 1; i < dimensions; i++)
			{
				//if (i == dimension)
				//{
					//int tpIdx = idx % rows;
					//int cnIdx = idx / rows;
					int tpIdxd = (idx % rows) + (dimension * rows);
					int tpIdxi = (idx % rows) + (i * rows);
					int cnIdxi = (idx / rows) + (i * cols);

					double sA = a[dimension] * a[dimension];
					double qA = sA * sA;
					double sC = c[dimension] * c[dimension];
					double qC = sC * sC;
					double scalarDxx1 = 4 * qA / qC;
					double scalarDxx2 = -2 * sA / sC;

					double dTpCn = tp[tpIdxd] - cn[cnIdxi];
					double sDTpCn = dTpCn * dTpCn;
					double a1 = scalarDxx1 * sDTpCn;
					double b1 = a1 + scalarDxx2;
					double c1 = b1 * D;
					double sTp = tp[tpIdxd] * tp[tpIdxi];
					double d = c1 * sTp;

					sumi += d;

				//	if (idx == 1)
				//		printf("rows=%i, cols=%i, dimensions=%i, idx=%i, i=%i tpIdxd=%i, tp[%i]=%f, cn[%i]=%f, tpIdxi=%i, tp[%i]=%f, a[%i]=%f c[%i]=%f dimension=%i sum=%f\r\n",
				//			rows, cols, dimensions, idx, i, tpIdxd, tpIdxd, tp[tpIdxd], cnIdxi, cn[cnIdxi], tpIdxi, tpIdxi, tp[tpIdxi], dimension, a[dimension], dimension, c[dimension], dimension, sumi);
				////}
			}
			
			sumij += sumi;
			//if (idx == 1)
			//	printf("rows=%i, cols=%i, dimensions=%i, idx=%i, dimension=%i sumij=%f\r\n",
			//		rows, cols, dimensions, idx, dimension, sumij);
		}
		thrust::get<2>(t) = sumij;
		//if (idx == 1)
		//	printf("rows=%i, cols=%i, dimensions=%i, idx=%i, sumij=%f, Dxx=%f\r\n",
		//		rows, cols, dimensions, idx, sumij, thrust::get<2>(t));
	}
};