#include "stdafx.h"
#include "Interpolation.h"
#include "Common.h"
#include "Double.h"
#include "RBF.h"
#include "PDE.h"
#include "PPP.h"
#include "Test.h"
#include <thread>


//#include "CppUnitTest.h"
//using namespace Microsoft::VisualStudio::CppUnitTestFramework;


vector<vector<MatrixXd>> Interpolation::getResult()
{
	return result;
}

MatrixXd Interpolation::getLambda(int id)
{
	return Lambda[id];
}
MatrixXd Interpolation::getTX(int id)
{
	return TX[id];
}
MatrixXd Interpolation::getC(int id)
{
	return C[id];
}
MatrixXd Interpolation::getA(int id)
{
	return A[id];
}
MatrixXd Interpolation::getU(int id)
{
	return U[id];
}

void Interpolation::interpolateGeneric(string prefix, double coef, double tsec, int b, int d, double inx1, double inx2, double r, double sigma, double T, double E, 
	vector<string> keys, const map<string, vector<vector<MatrixXd>> > *vInterpolation)
{
	Lambda.clear();
	TX.clear();
	A.clear();
	C.clear();
	U.clear();

	MatrixXd N = primeNMatrix(b, d);

	vector<thread> threads;

	for (int i = 0; i < N.rows(); i++)
	{
		threads.push_back(std::thread(&Interpolation::shapelambda2DGeneric, this, prefix, i, coef, tsec, r, sigma, T, E, inx1, inx2, N.row(i), keys, vInterpolation));
		//shapelambda2DGeneric(prefix, i, coef, tsec, r, sigma, T, E, inx1, inx2, N.row(i), keys, vInterpolation);
	}

	
	for (int i = 0; i < threads.size(); i++)
		threads.at(i).join();

	vector<MatrixXd> l;
	vector<MatrixXd> tx;
	vector<MatrixXd> a;
	vector<MatrixXd> c;
	vector<MatrixXd> u;
	
	for (int i = 0; i < N.rows(); i++)
	{
		l.push_back(Lambda[i]);
		tx.push_back(TX[i]);
		c.push_back(C[i]);
		a.push_back(A[i]);
		u.push_back(U[i]);
	}
	result = {l, tx, c, a, u };
	
}

MatrixXd Interpolation::primeNMatrix(int b, int d)
{
	MatrixXd L = subnumber(b, d);
	int ch = L.rows();

	MatrixXd N = MatrixXd::Ones(ch, d);
	for (int i = 0; i < ch; i++)
		for (int j = 0; j < d; j++)
			N(i, j) = pow(2, L(i, j)) + 1;

	return N;
}

MatrixXd Interpolation::subnumber(int b, int d)
{
	MatrixXd L;

	if (d == 1)
	{
		L = MatrixXd(1, 1);
		L << b;
	}
	else
	{
		int nbot = 1;

		for (int i = 0; i < b - d + 1; i++)
		{
			MatrixXd indextemp = subnumber(b - (i + 1), d - 1);
			int s = indextemp.rows();
			int ntop = nbot + s - 1;
			MatrixXd l(ntop, d);

			MatrixXd ones = MatrixXd::Ones(s, 1);

			l.block(nbot - 1, 0, ntop - nbot + 1, 1) = ones.array() * (i + 1);
			l.block(nbot - 1, 1, ntop - nbot + 1, d - 1) = indextemp;
			nbot = ntop + 1;


			if (L.rows() > 0)
			{
				l.block(0, 0, l.rows() - 1, l.cols()) = L;
			}
			L = l;
		}

	}
	return L;
}

typedef Eigen::SparseMatrix<double> SpMat;

void Interpolation::shapelambda2DGeneric(string prefix, int threadId, double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N,
	vector<string> keys, const map<string, vector<vector<MatrixXd>> > * state)
{
	map<string, vector<vector<MatrixXd>>> vInterpolation = * state;

	double num = N.prod();

	VectorXd t = VectorXd::LinSpaced(N(0, 0), 0, tsec);
	VectorXd x = VectorXd::LinSpaced(N(0, 1), inx1, inx2);
	double h1 = coef*tsec;
	double h2 = coef*(inx2 - inx1);

	MatrixXd c(1, 2);
	c << h1, h2;
	MatrixXd a = N.array() - 1;

	MatrixXd XXX = t.replicate(1, x.rows());
	MatrixXd YYY = x.replicate(1, t.rows());

	XXX.transposeInPlace();
	VectorXd xxx(Map<VectorXd>(XXX.data(), XXX.cols()*XXX.rows()));
	VectorXd yyy(Map<VectorXd>(YYY.data(), YYY.cols()*YYY.rows()));

	MatrixXd TX1(XXX.rows() * XXX.cols(), 2);
	TX1 << xxx, yyy;

	vector<MatrixXd> mqd = RBF::mqd2(TX1, TX1, a, c);
	MatrixXd FAI = mqd[0];
	MatrixXd FAI_t = mqd[1];
	MatrixXd FAI_x = mqd[2];
	MatrixXd FAI_xx = mqd[3];

	MatrixXd pa = (sigma * sigma * FAI_xx.array() / 2);
	MatrixXd pb = r*FAI_x.array() - r*FAI.array();
	MatrixXd P = FAI_t.array() + pa.array() + pb.array();

	VectorXd u = MatrixXd::Zero(num, 1);
	for (int s = 0; s < keys.size(); s += 2)
	{
		string k1 = keys[s];
		string k2 = keys[s + 1];
		u -= PDE::BlackScholes(TX1, r, sigma,
			vInterpolation[k1][0], vInterpolation[k1][1], vInterpolation[k1][2], vInterpolation[k1][3],
			vInterpolation[k2][0], vInterpolation[k2][1], vInterpolation[k2][2], vInterpolation[k2][3]);
	}

	for (int i = 0; i < num; i++)
	{
		if (abs(TX1(i, 1) - inx1) < DBL_EPSILON || abs(TX1(i, 1) - inx2) < DBL_EPSILON)
		{
			P.row(i) = FAI.row(i);
			double max = ( Double(TX1(i, 1)) - Double(E) * Double(exp(-r * (T - TX1(i, 0)))) ).value();
			u(i) = 0;
			double sub = 0;
			for (int s = 0; s < keys.size(); s += 2)
			{
				string k1 = keys[s];
				string k2 = keys[s + 1];
				double a = Test::inner(TX1(i, 0), TX1(i, 1), vInterpolation[k2][0], vInterpolation[k2][1], vInterpolation[k2][2], vInterpolation[k2][3]);
				double b = Test::inner(TX1(i, 0), TX1(i, 1), vInterpolation[k1][0], vInterpolation[k1][1], vInterpolation[k1][2], vInterpolation[k1][3]);
				sub += (a - b);
			}
			
			if (max > 0)
				u(i) = max - sub;
			else
				u(i) = 0 - sub;
		}

		if (abs(TX1(i, 0) - tsec) < DBL_EPSILON)
		{
			P.row(i) = FAI.row(i);
			double sub = 0;
			for (int s = 0; s < keys.size(); s += 2)
			{
				string k1 = keys[s];
				string k2 = keys[s + 1];
				double a = Test::inner(TX1(i, 0), TX1(i, 1), vInterpolation[k2][0], vInterpolation[k2][1], vInterpolation[k2][2], vInterpolation[k2][3]);
				double b = Test::inner(TX1(i, 0), TX1(i, 1), vInterpolation[k1][0], vInterpolation[k1][1], vInterpolation[k1][2], vInterpolation[k1][3]);
				sub += (a - b);
			}
			
			double d = PPP::Calculate(TX1.row(i)) - sub;
			u(i) = d;
		}
	}
	MatrixXd tx = TX1;
	
	PartialPivLU<MatrixXd> lu = PartialPivLU<MatrixXd>(P);
	
	MatrixXd J = lu.matrixLU().triangularView<UpLoType::Upper>();
	MatrixXd F = lu.matrixLU().triangularView<UpLoType::UnitLower>();
	MatrixXd Fa = lu.permutationP().transpose() * F;

	/*SpMat Fa = lu.permutationP().transpose() * F.sparseView();
	SparseLU<SparseMatrix<double, ColMajor>, COLAMDOrdering<int> > solver;
	solver.analyzePattern(Fa);
	solver.factorize(Fa);
	*/
	MatrixXd Jlamda = Fa.lu().solve(u);
	//MatrixXd Jlamda = solver.solve(u);

	//SpMat Ja = J.sparseView();
	//solver.analyzePattern(Ja);
	//solver.factorize(Ja);

	MatrixXd l = J.lu().solve(Jlamda);
	//MatrixXd l = solver.solve(Jlamda);

	Lambda[threadId] = l;
	TX[threadId] = tx;
	C[threadId] = c;
	A[threadId] = a;
	U[threadId] = u;

}
