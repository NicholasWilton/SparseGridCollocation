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
	{
		threads.at(i).join();
	}

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
		//Logger::WriteMessage(Common::printMatrix(TX1).c_str());
		//for (auto a : vInterpolation[k1][0])
		//	Logger::WriteMessage(Common::printMatrix(a).c_str());
		//for (auto a : vInterpolation[k1][1])
		//	Logger::WriteMessage(Common::printMatrix(a).c_str());
		//for (auto a : vInterpolation[k1][2])
		//	Logger::WriteMessage(Common::printMatrix(a).c_str()); 
		//for (auto a : vInterpolation[k1][3])
		//	Logger::WriteMessage(Common::printMatrix(a).c_str());

		//for (auto a : vInterpolation[k2][0])
		//	Logger::WriteMessage(Common::printMatrix(a).c_str());
		//for (auto a : vInterpolation[k2][1])
		//	Logger::WriteMessage(Common::printMatrix(a).c_str());
		//for (auto a : vInterpolation[k2][2])
		//	Logger::WriteMessage(Common::printMatrix(a).c_str());
		//for (auto a : vInterpolation[k2][3])
		//	Logger::WriteMessage(Common::printMatrix(a).c_str());

		u -= PDE::BlackScholes(TX1, r, sigma,
			vInterpolation[k1][0], vInterpolation[k1][1], vInterpolation[k1][2], vInterpolation[k1][3],
			vInterpolation[k2][0], vInterpolation[k2][1], vInterpolation[k2][2], vInterpolation[k2][3]);
		//Logger::WriteMessage(Common::printMatrix(u).c_str());
	}
	//Logger::WriteMessage(Common::printMatrix(u).c_str());
	//wcout << Common::printMatrix(u) << endl;

	for (int i = 0; i < num; i++)
	{
		//count++;
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
				double a = Test::innerZ(TX1(i, 0), TX1(i, 1), vInterpolation[k2][0], vInterpolation[k2][1], vInterpolation[k2][2], vInterpolation[k2][3]);
				//double a = Test::inner(TX1(i, 0), TX1(i, 1), vInterpolation[k2][0], vInterpolation[k2][1], vInterpolation[k2][2], vInterpolation[k2][3]);
				//double a = Test::innerMock(prefix, threadId, i, s + 1);
				double b = Test::innerZ(TX1(i, 0), TX1(i, 1), vInterpolation[k1][0], vInterpolation[k1][1], vInterpolation[k1][2], vInterpolation[k1][3]);
				//double b = Test::innerMock(prefix, threadId, i, s + 2);
				//double b = Test::inner(TX1(i, 0), TX1(i, 1), vInterpolation[k1][0], vInterpolation[k1][1], vInterpolation[k1][2], vInterpolation[k1][3]);
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
				//double a = Test::inner(TX1(i, 0), TX1(i, 1), vInterpolation[k2][0], vInterpolation[k2][1], vInterpolation[k2][2], vInterpolation[k2][3]);
				double a = Test::innerZ(TX1(i, 0), TX1(i, 1), vInterpolation[k2][0], vInterpolation[k2][1], vInterpolation[k2][2], vInterpolation[k2][3]);
				//double a = Test::innerMock(prefix, threadId, i, keys.size() + s + 1);
				//double b = Test::inner(TX1(i, 0), TX1(i, 1), vInterpolation[k1][0], vInterpolation[k1][1], vInterpolation[k1][2], vInterpolation[k1][3]);
				double b = Test::innerZ(TX1(i, 0), TX1(i, 1), vInterpolation[k1][0], vInterpolation[k1][1], vInterpolation[k1][2], vInterpolation[k1][3]);
				//double b = Test::innerMock(prefix, threadId, i, keys.size() + s + 2);
				/*sub += (inner_test(TX1(i, 0), TX1(i, 1), vInterpolation[k2][0], vInterpolation[k2][0], vInterpolation[k2][0], vInterpolation[k2][0])
				- inner_test(TX1(i, 0), TX1(i, 1), vInterpolation[k1][0], vInterpolation[k1][0], vInterpolation[k1][0], vInterpolation[k1][0]));*/
				sub += (a - b);
			}
			
			double d = PPP::Calculate(TX1.row(i)) - sub;
			//wstringstream ss;
			//ss << d;
			//Logger::WriteMessage(ss.str().c_str());
			u(i) = d;
		}
	}
	MatrixXd tx = TX1;

	//Logger::WriteMessage(Common::printMatrix(tx).c_str());
	//Logger::WriteMessage(Common::printMatrix(u).c_str());
	//Logger::WriteMessage(Common::printMatrix(P).c_str());
	//wcout << Common::printMatrix(tx) << endl;
	//wcout << Common::printMatrix(u) << endl;
	//wcout << Common::printMatrix(P) << endl;
	
		
	PartialPivLU<MatrixXd> lu = PartialPivLU<MatrixXd>(P);
	MatrixXd J = lu.matrixLU().triangularView<UpLoType::Upper>();
	MatrixXd F = lu.matrixLU().triangularView<UpLoType::UnitLower>();
	
	//Logger::WriteMessage(Common::printMatrix(F).c_str());
	//Hack: to get around the fact that Eigen doesn't compute the permutation matrix p correctly
	//MatrixXd transform(15, 15);
	//transform << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
	MatrixXd Fa = lu.permutationP().transpose() * F;
	//wcout << Common::printMatrix(Fa) << endl;
	//MatrixXd Fa = F;
	//Logger::WriteMessage(Common::printMatrix(Fa).c_str());
	//Eigen also seems to solve with different rounding, maybe a double arithmetic issue:
	//Jlamda = F\U;
	MatrixXd Jlamda = Fa.lu().solve(u);
	//wcout << Common::printMatrix(Jlamda) << endl;
	//Logger::WriteMessage(Common::printMatrix(Jlamda).c_str());
	//lamb = J\Jlamda;
	MatrixXd l = J.lu().solve(Jlamda);
	//Logger::WriteMessage(Common::printMatrix(l).c_str());
	//wcout << Common::printMatrix(l) << endl;
	Lambda[threadId] = l;
	TX[threadId] = tx;
	C[threadId] = c;
	A[threadId] = a;
	U[threadId] = u;

}
