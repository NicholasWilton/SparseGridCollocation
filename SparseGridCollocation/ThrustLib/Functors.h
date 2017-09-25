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