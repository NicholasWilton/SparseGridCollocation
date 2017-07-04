#include "stdafx.h"
#include "Interpolation.h"
#include "RBF.h"
#include "PDE.h"
#include "PPP.h"
#include "Test.h"
#include <thread>
#include <mutex>


//Interpolation::Interpolation()
//{
//
//}
//
//Interpolation::Interpolation(const Interpolation & obj)
//{
//}
//
//Interpolation::~Interpolation()
//{
//}

vector<vector<MatrixXd>> Interpolation::getResult()
{
	return result;
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
		threads.push_back(std::thread(&Interpolation::shapelambda2DGeneric, this, i,
			coef, tsec, r, sigma, T, E, inx1, inx2, N.row(0), keys, vInterpolation));
		//shapelambda2DGeneric(coef, tsec, r, sigma, T, E, inx1, inx2, N.row(0), keys, vInterpolation);
	}

	
	for (int i = 0; i < threads.size(); i++)
	{
		threads.at(i).join();
	}

	vector<MatrixXd> l;
	vector<MatrixXd> tx;
	vector<MatrixXd> a;
	vector<MatrixXd> c;
	vector<MatrixXd> u;
	
	for (int i = 0; i < threads.size(); i++)
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
	//L = subnumber(b, d);
	MatrixXd L = subnumber(b, d);
	//[ch, ~] = size(L);
	int ch = L.rows();

	//N = ones(ch, d);
	MatrixXd N = MatrixXd::Ones(ch, d);
	//for i = 1:d
	for (int i = 0; i < ch; i++)
		//	N(:, i) = 2.^L(:, i) + 1;
		for (int j = 0; j < d; j++)
			N(i, j) = pow(2, L(i, j)) + 1;
	//end

	//Logger::WriteMessage(Common::printMatrix(N).c_str());
	return N;
}

MatrixXd Interpolation::subnumber(int b, int d)
{
	MatrixXd L;
	//% Find possible layouts L that satisfy | L | _1 = b
	//	%[~, N] = size(L), N = d.
	//	if d == 1
	//		L(1) = b;
	//	else
	//		nbot = 1;
	if (d == 1)
	{
		L = MatrixXd(1, 1);
		L << b;
	}
	else
	{
		int nbot = 1;

		//for i = 1:b - d + 1

		for (int i = 0; i < b - d + 1; i++)
		{
			//	indextemp = subnumber(b - i, d - 1);
			MatrixXd indextemp = subnumber(b - (i + 1), d - 1);
			//Logger::WriteMessage(Common::printMatrix(indextemp).c_str());
			//[s, ~] = size(indextemp);
			int s = indextemp.rows();
			//ntop = nbot + s - 1;
			int ntop = nbot + s - 1;
			MatrixXd l(ntop, d);

			//L(nbot:ntop, 1) = i*ones(s, 1);
			MatrixXd ones = MatrixXd::Ones(s, 1);

			l.block(nbot - 1, 0, ntop - nbot + 1, 1) = ones.array() * (i + 1);
			//Logger::WriteMessage(Common::printMatrix(l).c_str());

			//L(nbot:ntop, 2 : d) = indextemp;
			l.block(nbot - 1, 1, ntop - nbot + 1, d - 1) = indextemp;
			//Logger::WriteMessage(Common::printMatrix(l).c_str());
			//nbot = ntop + 1;
			nbot = ntop + 1;
			//end

			if (L.rows() > 0)
			{
				l.block(0, 0, l.rows() - 1, l.cols()) = L;
				//Logger::WriteMessage(Common::printMatrix(l).c_str());
			}
			L = l;
			//Logger::WriteMessage(Common::printMatrix(L).c_str());
		}
		//	end

	}
	return L;
}



void Interpolation::shapelambda2DGeneric(int threadId, double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N,
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
		//count++;
		if (abs(TX1(i, 1) - inx1) < DBL_EPSILON || abs(TX1(i, 1) - inx2) < DBL_EPSILON)
		{
			P.row(i) = FAI.row(i);
			double max = TX1(i, 1) - E*exp(-r * (T - TX1(i, 0)));
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
				/*sub += (inner_test(TX1(i, 0), TX1(i, 1), vInterpolation[k2][0], vInterpolation[k2][0], vInterpolation[k2][0], vInterpolation[k2][0])
				- inner_test(TX1(i, 0), TX1(i, 1), vInterpolation[k1][0], vInterpolation[k1][0], vInterpolation[k1][0], vInterpolation[k1][0]));*/
				sub += (a - b);
			}
			u(i) = PPP::Calculate(TX1.row(i)) - sub;
		}
	}
	MatrixXd tx = TX1;

	PartialPivLU<MatrixXd> lu = PartialPivLU<MatrixXd>(P);
	MatrixXd J = lu.matrixLU().triangularView<UpLoType::Upper>();
	MatrixXd F = lu.matrixLU().triangularView<UpLoType::UnitLower>();;
	//Hack: to get around the fact that Eigen doesn't compute the permutation matrix p correctly
	MatrixXd transform(15, 15);
	transform << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
	//MatrixXd Fa = transform * F;
	MatrixXd Fa = F;

	//Eigen also seems to solve with different rounding, maybe a double arithmetic issue:
	//Jlamda = F\U;
	MatrixXd Jlamda = Fa.lu().solve(u);
	//lamb = J\Jlamda;
	MatrixXd l = J.lu().solve(Jlamda);

	Lambda[threadId] = l;
	TX[threadId] = tx;
	C[threadId] = c;
	A[threadId] = a;
	U[threadId] = u;

}
