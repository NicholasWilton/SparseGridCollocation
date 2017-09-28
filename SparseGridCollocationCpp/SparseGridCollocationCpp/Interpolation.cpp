#include "stdafx.h"
#include "Interpolation.h"
#include ".\..\Common\Utility.h"
#include "Double.h"
#include "RBF.h"
#include "PDE.h"
#include "PPP.h"
#include "Test.h"
#include "TestNodes.h"
#include <thread>


//#include "CppUnitTest.h"
//using namespace Microsoft::VisualStudio::CppUnitTestFramework;

int Leicester::SparseGridCollocation::Interpolation::callCount;

vector<vector<MatrixXd>> Leicester::SparseGridCollocation::Interpolation::getResult()
{
	return result;
}

MatrixXd Leicester::SparseGridCollocation::Interpolation::getLambda(int id)
{
	return Lambda[id];
}
MatrixXd Leicester::SparseGridCollocation::Interpolation::getTX(int id)
{
	return TX[id];
}
MatrixXd Leicester::SparseGridCollocation::Interpolation::getC(int id)
{
	return C[id];
}
MatrixXd Leicester::SparseGridCollocation::Interpolation::getA(int id)
{
	return A[id];
}
MatrixXd Leicester::SparseGridCollocation::Interpolation::getU(int id)
{
	return U[id];
}

void Leicester::SparseGridCollocation::Interpolation::interpolateGeneric(string prefix, double coef, double tsec, int b, int d, double inx1, double inx2, double r, double sigma, double T, double E,
	vector<string> keys, const map<string, vector<vector<MatrixXd>> > *vInterpolation)
{
	Lambda.clear();
	TX.clear();
	A.clear();
	C.clear();
	U.clear();

	MatrixXd N = primeNMatrix(b, d);

	vector<thread> threads;
	//cout << "N.rows()=" << N.rows() << endl;
	

	for (int i = 0; i < N.rows(); i++)
	{
		MatrixXd TX1 = GenerateNodes(coef, tsec, inx1, inx2, N.row(i));
		threads.push_back(std::thread(&Interpolation::shapelambda2DGeneric, this, prefix, i, coef, tsec, r, sigma, T, E, inx1, inx2, N.row(i), keys, vInterpolation, TX1));
		//shapelambda2DGeneric(prefix, i, coef, tsec, r, sigma, T, E, inx1, inx2, N.row(i), keys, vInterpolation, TX1);
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
	result = { l, tx, c, a, u };

}

void Leicester::SparseGridCollocation::Interpolation::interpolateGenericND(string prefix, double coef, double tsec, int b, int d, MatrixXd inx1, MatrixXd inx2, double r, double sigma, double T, double E,
	vector<string> keys, const map<string, vector<vector<MatrixXd>> > *vInterpolation, bool useCuda)
{
	Lambda.clear();
	TX.clear();
	A.clear();
	C.clear();
	U.clear();

	MatrixXd N = primeNMatrix(b, d);
	//wcout << Common::printMatrix(N).c_str() << endl;
	vector<thread> threads;
	//cout << "N.rows()=" << N.rows() << endl;
	for (int i = 0; i < N.rows(); i++)
	{
		MatrixXd n = N.row(i);
		MatrixXd TXYZ = TestNodes::GenerateTestNodes(0, tsec, inx1.transpose(), inx2.transpose(), n, coef);
		threads.push_back(std::thread(&Interpolation::shapelambdaNDGeneric, this, prefix, i, coef, tsec, r, sigma, T, E, inx1, inx2, N.row(i), keys, vInterpolation));
		//	shapelambdaNDGeneric(prefix, i, coef, tsec, r, sigma, T, E, inx1, inx2, N.row(i), keys, vInterpolation);
		
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

MatrixXd Leicester::SparseGridCollocation::Interpolation::primeNMatrix(int b, int d)
{
	MatrixXd L = subnumber(b, d);
	int ch = L.rows();

	MatrixXd N = MatrixXd::Ones(ch, d);
	for (int i = 0; i < ch; i++)
		for (int j = 0; j < d; j++)
			N(i, j) = pow(2, L(i, j)) + 1;

	return N;
}

MatrixXd Leicester::SparseGridCollocation::Interpolation::subnumber(int b, int d)
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
				//l.block(0, 0, l.rows() - 1, l.cols()) = L;
				l.block(0, 0, L.rows(), L.cols()) = L;
			}
			L = l;
		}

	}
	return L;
}

typedef Eigen::SparseMatrix<double> SpMat;

///Copy v into result successfively until result is filled i.e. v =[1,2] totalength = 4 => result = [1,2,1,2]
VectorXd Leicester::SparseGridCollocation::Interpolation::Replicate(VectorXd v, int totalLength)
{
	VectorXd Result(totalLength);
	for (int i = 0; i < totalLength; i += v.size())
	{
		for (int j = 0; j < v.size(); j++)
		{
			Result[i + j] = v[j];
		}
	}
	return Result;
}
VectorXd Leicester::SparseGridCollocation::Interpolation::Replicate(VectorXd v, int totalLength, int dup)
{
	VectorXd Result(totalLength);
	
	for (int i = 0; i < totalLength; i += (v.size() * dup) )
	{
		for (int j = 0; j < v.size(); j++)
		{
			for (int duplicated = 0; duplicated < dup; duplicated++)
			{
				int idx = i + (j * dup) + duplicated;
				if (idx < totalLength)
				{
					Result[idx] = v[j];
					//cout << "idx="<< idx << " v[j]=" << v[j] << endl;
				}
			
			}
		}
	}
	return Result;
}
MatrixXd Leicester::SparseGridCollocation::Interpolation::GenerateNodes(double coef, double tsec, double inx1, double inx2, MatrixXd N)
{
	VectorXd t = VectorXd::LinSpaced(N(0, 0), 0, tsec);
	VectorXd x = VectorXd::LinSpaced(N(0, 1), inx1, inx2);
	
	MatrixXd XXX = t.replicate(1, x.rows());
	MatrixXd YYY = x.replicate(1, t.rows());
	//wcout << Common::printMatrix(YYY) << endl;
	//wcout << Common::printMatrix(XXX) << endl;
	XXX.transposeInPlace();
	//wcout << Common::printMatrix(XXX) << endl;
	VectorXd xxx(Map<VectorXd>(XXX.data(), XXX.cols()*XXX.rows()));
	VectorXd yyy(Map<VectorXd>(YYY.data(), YYY.cols()*YYY.rows()));
	//wcout << Common::printMatrix(xxx) << endl;
	//wcout << Common::printMatrix(yyy) << endl;

	MatrixXd TX1(XXX.rows() * XXX.cols(), 2);
	TX1 << xxx, yyy;
	return TX1;
}


void Leicester::SparseGridCollocation::Interpolation::shapelambda2DGeneric(string prefix, int threadId, double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N,
	vector<string> keys, const map<string, vector<vector<MatrixXd>> > * state, MatrixXd TP)
{
	map<string, vector<vector<MatrixXd>>> vInterpolation = *state;

	double num = N.prod();
	double h1 = coef*tsec;
	double h2 = coef*(inx2 - inx1);
	MatrixXd c(1, 2);
	c << h1, h2;
	MatrixXd a = N.array() - 1;

	vector<MatrixXd> mqd = RBF::Gaussian2D(TP, TP, a, c);

	ShapeLambda2D(mqd, sigma, r, num, keys, TP, vInterpolation, inx1, inx2, E, T, tsec, threadId, c, a);

}

void Leicester::SparseGridCollocation::Interpolation::ShapeLambda2D(vector<MatrixXd> &mqd, double sigma, double &r, double num,
	vector<string> &keys, MatrixXd &TP, map<string, vector<vector<MatrixXd>>> &vInterpolation, 
	double inx1, double inx2, double E, double T, double tsec, int &threadId, MatrixXd &c, MatrixXd &a)
{
	MatrixXd phi = mqd[0];
	MatrixXd phi_t = mqd[1];
	MatrixXd phi_x = mqd[2];
	MatrixXd phi_xx = mqd[3];

	MatrixXd pa = (sigma * sigma * phi_xx.array() / 2); //sigma^2/2 d^2V/dV^2
	MatrixXd pb = r*phi_x.array() - r*phi.array();// r dV/dx - rV
	MatrixXd P = phi_t.array() + pa.array() + pb.array(); //dV/dt + ... + ...

	VectorXd u = MatrixXd::Zero(num, 1);
	for (int s = 0; s < keys.size(); s += 2)
	{
		string k1 = keys[s];
		string k2 = keys[s + 1];
		u -= PDE::BlackScholes(TP, r, sigma,
			vInterpolation[k1][0], vInterpolation[k1][1], vInterpolation[k1][2], vInterpolation[k1][3],
			vInterpolation[k2][0], vInterpolation[k2][1], vInterpolation[k2][2], vInterpolation[k2][3]);
	}

	for (int i = 0; i < num; i++)
	{
		if (abs(TP(i, 1) - inx1) < DBL_EPSILON || abs(TP(i, 1) - inx2) < DBL_EPSILON)
		{
			P.row(i) = phi.row(i);
			double max = (Double(TP(i, 1)) - Double(E) * Double(exp(-r * (T - TP(i, 0))))).value();
			u(i) = 0;
			double sub = 0;
			for (int s = 0; s < keys.size(); s += 2)
			{
				string k1 = keys[s];
				string k2 = keys[s + 1];
				double a = Test::inner(TP(i, 0), TP(i, 1), vInterpolation[k2][0], vInterpolation[k2][1], vInterpolation[k2][2], vInterpolation[k2][3]);
				double b = Test::inner(TP(i, 0), TP(i, 1), vInterpolation[k1][0], vInterpolation[k1][1], vInterpolation[k1][2], vInterpolation[k1][3]);
				sub += (a - b);
			}

			if (max > 0)
				u(i) = max - sub;
			else
				u(i) = 0 - sub;
		}

		if (abs(TP(i, 0) - tsec) < DBL_EPSILON)
		{
			P.row(i) = phi.row(i);
			double sub = 0;
			for (int s = 0; s < keys.size(); s += 2)
			{
				string k1 = keys[s];
				string k2 = keys[s + 1];
				double a = Test::inner(TP(i, 0), TP(i, 1), vInterpolation[k2][0], vInterpolation[k2][1], vInterpolation[k2][2], vInterpolation[k2][3]);
				double b = Test::inner(TP(i, 0), TP(i, 1), vInterpolation[k1][0], vInterpolation[k1][1], vInterpolation[k1][2], vInterpolation[k1][3]);
				sub += (a - b);
			}

			double d = PPP::Calculate(TP.row(i)) - sub;
			u(i) = d;
		}
	}
	MatrixXd tx = TP;

	PartialPivLU<MatrixXd> lu = PartialPivLU<MatrixXd>(P);

	MatrixXd J = lu.matrixLU().triangularView<UpLoType::Upper>();
	MatrixXd F = lu.matrixLU().triangularView<UpLoType::UnitLower>();
	MatrixXd Fa = lu.permutationP().transpose() * F;

	MatrixXd Jlamda = Fa.lu().solve(u);
	MatrixXd l = J.lu().solve(Jlamda);

	Lambda[threadId] = l;
	TX[threadId] = tx;
	C[threadId] = c;
	A[threadId] = a;
	U[threadId] = u;
}

void Leicester::SparseGridCollocation::Interpolation::shapelambdaNDGeneric(string prefix, int threadId, double coef, double tsec, double r, double sigma, double T, double E, MatrixXd inx1, MatrixXd inx2, MatrixXd N,
	vector<string> keys, const map<string, vector<vector<MatrixXd>> > * state)
{
	map<string, vector<vector<MatrixXd>>> vInterpolation = * state;

	double num = N.prod();

	MatrixXd cha(1, 1 + inx2.cols());
	cha.block(0,1,1,inx2.cols()) = inx2 - inx1;
	cha(0, 0) = tsec;
	MatrixXd c = coef * cha;
	
	MatrixXd a = N.array() - 1;
		
	MatrixXd TXYZ = TestNodes::GenerateTestNodes(0, tsec, inx1.transpose(), inx2.transpose(), N, coef);
	//Common::Utility::saveArray(TXYZ, "TXYZ.txt");
	//Common::Utility::saveArray(a, "a.txt");
	//Common::Utility::saveArray(c, "c.txt");
	vector<MatrixXd> mqd = RBF::GaussianND(TXYZ, TXYZ, a, c);
	//Common::Utility::saveArray(mqd[0], "mqd0.txt");
	//Common::Utility::saveArray(mqd[1], "mqd1.txt");
	//Common::Utility::saveArray(mqd[2], "mqd2.txt");
	//Common::Utility::saveArray(mqd[3], "mqd3.txt");
	
	MatrixXd P = mqd[1] + (sigma * sigma) * mqd[3] / 2 + r * mqd[2] - r * mqd[0];
	//if (prefix.compare("4") == 0)
	//	Common::saveArray(P, "P.txt");

	VectorXd u = MatrixXd::Zero(num, 1);
	
	for (int s = 0; s < keys.size(); s += TXYZ.cols())
	{
		vector<string> k;
		for(int c =0; c < TXYZ.cols(); c++)
			k.push_back( keys[s+c] );
		
		u -= PDE::BlackScholesNd(TXYZ, r, sigma, k, state);
	}

	//if (prefix.compare("4") == 0)
	//	Common::saveArray(u, "u.txt");

	for (int i = 0; i < num; i++)
	{
		double lower = 0.0;
		double upper = 0.0;
		for (int j = 1; j < TXYZ.cols(); j ++ ) //if ANY node is on a spatial dimensional boundary then our product should be zero
		{
			double diff1 = abs(TXYZ(i, j) - inx1(0, j-1));
			double diff2 = abs(TXYZ(i, j) - inx2(0, j-1));
			if (j == 1)
			{
				lower = diff1;
				upper = diff2;
			}
			lower *= diff1;
			upper *= diff2;
		}
		//do the boundary update for spatial dimensions
		if (lower < DBL_EPSILON || upper < DBL_EPSILON)
		{
			//wcout << Common::printMatrix(TXYZ) << endl;
			P.row(i) = mqd[0].row(i);

			//calculate the price using the payoff function at the spatial boundary, i.e underlying asset = 0 or the max defined in inx2
			double mean = (TXYZ.row(i).sum() - TXYZ(i, 0)) / (TXYZ.cols() - 1); //ignoring time dimension
			double max = mean - E * exp(-r * (T - TXYZ(i, 0)) );
			u(i) = 0;
			if (max < 0)
				max = 0;

			for (int s = 0; s < keys.size(); s++)
			{
				string k1 = keys[s];
				double a = Test::innerND(TXYZ.row(i), vInterpolation[k1][0], vInterpolation[k1][1], vInterpolation[k1][2], vInterpolation[k1][3]);
				//double factor = Common::BinomialCoefficient(keys.size()-1, s);
				double factor = 1;
				if (s % 2 == 0)
					max -= factor * a;
				else
					max += factor * a;
			}
			u(i) = max;
			//if (max > 0)
			//	u(i) = max - sub;
			//else
			//	u(i) = 0 - sub;
		}
		//do the boundary update for time dimension
		if (abs(TXYZ(i, 0) - tsec) < DBL_EPSILON)
		{
			P.row(i) = mqd[0].row(i);
			//double sub = 0;
			double ppp = PPP::Calculate(TXYZ.row(i));
			for (int s = 0; s < keys.size(); s++)
			{
				string k1 = keys[s];
				double a = Test::innerND(TXYZ.row(i), vInterpolation[k1][0], vInterpolation[k1][1], vInterpolation[k1][2], vInterpolation[k1][3]);
				//double factor = Common::BinomialCoefficient(keys.size()-1, s);
				double factor = 1;

				if (s % 2 == 0)
					ppp -= factor * a;
				else
					ppp += factor * a;
			}

			//double d = ppp - sub;
			u(i) = ppp;
		}
	}

	//if (prefix.compare("4") == 0)
	//	Common::saveArray(u, "u.txt");

	MatrixXd tx = TXYZ;
	
	//wcout << Common::printMatrix(P) << endl;
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
	//wcout << Common::printMatrix(l) << endl;
	Lambda[threadId] = l;
	TX[threadId] = tx;
	C[threadId] = c;
	A[threadId] = a;
	U[threadId] = u;
	//Common::saveArray(l, "l.txt");

}
