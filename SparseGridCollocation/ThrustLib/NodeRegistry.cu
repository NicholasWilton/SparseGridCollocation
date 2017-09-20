#include "NodeRegistry.h"

namespace Leicester
{
	namespace ThrustLib
	{

		//__global__ void Add_CUDA(int b, int d, double* N)
		//{
		//	double *d_L = (double*)malloc(3 * sizeof(double));
		//	d_L[0] = 1;
		//	d_L[1] = 1;

		//	subnumber << <1, 1 >> > (b, d, d_L);
		//	int ch = d_L[0];
		//	free(N);
		//	N = (double*)malloc(ch * d * sizeof(double));
		//	for (int i = 0; i < ch; i++)
		//		for (int j = 0; j < d; j++)
		//		{
		//			int idx = j + (i * ch);
		//			N[idx] = (d_L[idx] * d_L[idx]) + 1;
		//		}
		//}

		//MatrixXd NodeRegistry::primeNMatrix(int b, int d)
		//{
		//	MatrixXd L = subnumber(b, d);
		//	int ch = L.rows();

		//	MatrixXd N = MatrixXd::Ones(ch, d);
		//	for (int i = 0; i < ch; i++)
		//		for (int j = 0; j < d; j++)
		//			N(i, j) = pow(2, L(i, j)) + 1;

		//	return N;
		//}

		//MatrixXd NodeRegistry::subnumber(int b, int d)
		//{
		//	MatrixXd L;

		//	if (d == 1)
		//	{
		//		L = MatrixXd(1, 1);
		//		L << b;
		//	}
		//	else
		//	{
		//		int nbot = 1;

		//		for (int i = 0; i < b - d + 1; i++)
		//		{
		//			MatrixXd indextemp = subnumber(b - (i + 1), d - 1);
		//			int s = indextemp.rows();
		//			int ntop = nbot + s - 1;
		//			MatrixXd l(ntop, d);

		//			MatrixXd ones = MatrixXd::Ones(s, 1);

		//			l.block(nbot - 1, 0, ntop - nbot + 1, 1) = ones.array() * (i + 1);
		//			l.block(nbot - 1, 1, ntop - nbot + 1, d - 1) = indextemp;
		//			nbot = ntop + 1;


		//			if (L.rows() > 0)
		//			{
		//				//l.block(0, 0, l.rows() - 1, l.cols()) = L;
		//				l.block(0, 0, L.rows(), L.cols()) = L;
		//			}
		//			L = l;
		//		}

		//	}
		//	return L;
		//}

	}
}