#include "stdafx.h"
#include "MoL.h"
#include "Common.h"
#include "EuropeanCallOption.h"
#include "VectorUtil.h"
#include "RBF.h"
#include "BasketOption.h"
#include "MatrixUtil.h"
#include "TestNodes.h"
#include <iomanip>
#include <fstream>
#include "Sort.h";


using namespace Eigen;
using namespace std;
using namespace Leicester;

Leicester::MoL::MoL()
{
}


Leicester::MoL::~MoL()
{
}



vector<VectorXd> Leicester::MoL::MethodOfLines(Params p)
{
	stringstream ssX;
	ssX << setprecision(16) << "SmoothInitial_EuroCall_" << p.T << "_" << p.Tdone << "_" << p.Tend << "_" << p.dt << "_" << p.K << "_" << p.r << "_" << p.sigma << "_" << p.theta << "_" << p.inx1[0] << "_" << p.inx2[0] << "_X.dat";
	stringstream ssU;
	ssU << setprecision(16) << "SmoothInitial_EuroCall_" << p.T << "_" << p.Tdone << "_" << p.Tend << "_" << p.dt << "_" << p.K << "_" << p.r << "_" << p.sigma << "_" << p.theta << "_" << p.inx1[0] << "_" << p.inx2[0] << "_U.dat";
	stringstream ssT;
	ssT << setprecision(16) << "SmoothInitial_EuroCall_" << p.T << "_" << p.Tdone << "_" << p.Tend << "_" << p.dt << "_" << p.K << "_" << p.r << "_" << p.sigma << "_" << p.theta << "_" << p.inx1[0] << "_" << p.inx2[0] << "_T.dat";


	string fileX = ssX.str();
	ifstream x(fileX.c_str());
	string fileU = ssU.str();
	ifstream u(fileU.c_str());
	string fileT = ssT.str();
	ifstream t(fileU.c_str());

	if (x.good() & u.good() & t.good())
	{
		x.close();
		MatrixXd mX = Common::ReadBinary(fileX, 32769, 1);
		VectorXd X(Map<VectorXd>(mX.data(), mX.cols()*mX.rows()));
		u.close();
		MatrixXd mU = Common::ReadBinary(fileU, 32769, 1);
		VectorXd U(Map<VectorXd>(mU.data(), mU.cols()*mU.rows()));
		t.close();
		MatrixXd mT = Common::ReadBinary(fileT, 1, 1);
		VectorXd T(Map<VectorXd>(mT.data(), mT.cols()*mT.rows()));
		return { X, U, T };
	}
	else
	{
		vector<VectorXd> smoothInitial = MethodOfLines(p.T, p.Tdone, p.Tend, p.dt, p.K, p.r, p.sigma, p.theta, p.inx1[0], p.inx2[0]);
		Common::WriteToBinary(fileX, smoothInitial[0]);
		Common::WriteToBinary(fileU, smoothInitial[1]);
		Common::WriteToBinary(fileT, smoothInitial[2]);
		return smoothInitial;
	}

}

vector<VectorXd> Leicester::MoL::MethodOfLines(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, double inx1, double inx2)
{
	cout << "MethodOfLines for 1-D European Call Option" << endl;
	vector<VectorXd> price = EuroCallOption1D(T, Tdone, Tend, dt, K, r, sigma, theta, inx1, inx2);

	VectorXd x = price[0];
	VectorXd lamb = price[1];
	VectorXd c = price[2];
	VectorXd t = price[3];
	double Smin = 0;
	double Smax = 3 * K;
	double twoPower15Plus1 = pow(2, 15) + 1;

	VectorXd X_ini = VectorXd::LinSpaced(twoPower15Plus1, Smin, Smax);

	MatrixXd phi = MatrixXd::Ones(X_ini.size(), x.size());
	for (int i = 0; i < x.size(); i++)
	{
		vector<MatrixXd> rbf = RBF::MultiQuadric1D(X_ini, x(i), c(i));
		phi.col(i) = rbf[0].col(0);
	}

	VectorXd U_ini = phi * lamb;

	return { X_ini, U_ini,t };
}

vector<VectorXd> Leicester::MoL::EuroCallOption1D(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, double inx1, double inx2)
{
	Option *option = new EuropeanCallOption(K, T);
	int N_uniform = 5000;
	VectorXd x = VectorXd::LinSpaced(N_uniform, inx1, inx2);

	VectorXd u0 = option->PayOffFunction(x);
	delete option;

	VectorXd dx = VectorUtil::Diff(x);
	const double inf = numeric_limits<double>::infinity();
	VectorXd vInf(dx.rows());
	vInf.fill(inf);

	VectorXd pushed = VectorUtil::Push(dx, inf);
	VectorXd queued = VectorUtil::Queue(dx, inf);
	VectorXd c = 2 * (pushed.array() >= queued.array()).select(queued, pushed);
	//Common::saveArray(c, "1D_c.txt");

	VectorXd xx = VectorXd::LinSpaced(1000, 0, 3 * K);
	VectorXd IT = ((1.2 * K >= x.array()) && (x.array() >= 0.8 * K)).select(x, 0);
	double sumIT = IT.sum();
	VectorXd AroundE = VectorUtil::Select(IT, 0);

	//Common::saveArray(x, "1D_X.txt");
	//Common::saveArray(xx, "1D_XX.txt");

	int N = x.size();
	MatrixXd D1_mid = MatrixXd::Zero(AroundE.size(), N);
	MatrixXd D2_mid = MatrixXd::Zero(AroundE.size(), N);
	MatrixXd D3_mid = MatrixXd::Zero(AroundE.size(), N);

	MatrixXd A = MatrixXd::Zero(N, N);
	MatrixXd D1 = MatrixXd::Zero(N, N);
	MatrixXd D2 = MatrixXd::Zero(N, N);

	MatrixXd Axx = MatrixXd::Zero(xx.size(), N);

	for (int j = 0; j < N; j++)
	{
		vector<MatrixXd> vAxx = RBF::MultiQuadric1D(xx, x[j], c[j]);
		Axx.col(j) = vAxx[0].col(0);
		//Common::saveArray(Axx, "1D_Axx.txt");

		vector<MatrixXd> vAx = RBF::MultiQuadric1D(x, x[j], c[j]);
		A.col(j) = vAx[0].col(0);
		//Common::saveArray(A, "1D_A.txt");

		D1.col(j) = vAx[1].col(0);
		//Common::saveArray(D1, "1D_D1.txt");
		D2.col(j) = vAx[2].col(0);
		//Common::saveArray(D2, "1D_D2.txt");
		vector<MatrixXd> vAE = RBF::MultiQuadric1D(AroundE, x[j], c[j]);
		D1_mid.col(j) = vAE[1].col(0);
		//Common::saveArray(D1_mid, "1D_D1_mid.txt");
		D2_mid.col(j) = vAE[2].col(0);
		//Common::saveArray(D2_mid, "1D_D2_mid.txt");

		D3_mid.col(j) = vAE[3].col(0);
		//Common::saveArray(D3_mid, "1D_D3_mid.txt");
		D1_mid.col(j) = D1_mid.col(j).array() / AroundE.array();
		D2_mid.col(j) = D2_mid.col(j).array() / (AroundE.array() * AroundE.array());
	}


	VectorXd lamb = A.lu().solve(u0);

	MatrixXd uu0 = Axx*lamb;
	MatrixXd deri1 = D1_mid*lamb;
	MatrixXd deri2 = D2_mid*lamb;

	MatrixXd deri3 = D3_mid*lamb;

	MatrixXd A1 = A.row(1);
	MatrixXd Aend = A.row(A.rows() - 1);
	MatrixXd a = A.block(1, 0, A.rows() - 2, A.cols());
	MatrixXd d1 = D1.block(1, 0, D1.rows() - 2, D1.cols());
	MatrixXd d2 = D2.block(1, 0, D2.rows() - 2, D2.cols());



	MatrixXd P = a * r - 0.5 * (sigma * sigma) * d2 - r * d1;
	//Common::saveArray(P, "P.txt");
	MatrixXd H = a + dt * (1 - theta)* P;
	//Common::saveArray(H, "H.txt");
	MatrixXd G = a - dt * theta * P;
	//Common::saveArray(G, "G.txt");

	int count = 0;
	cout << "MoL Iterative solver\r\n";
	while (Tend - Tdone > 1E-8)
	{
		Tdone += dt;
		MatrixXd g = G* lamb;

		//VectorXd fff = VectorUtil::PushAndQueue(0, (VectorXd)g.col(0), inx2 - exp(-r*Tdone)*K);
		VectorXd fff = VectorUtil::PushAndQueue(0, Map<VectorXd>(g.data(), g.cols() * g.rows()), inx2 - exp(-r*Tdone)*K);

		MatrixXd HH(A1.cols(), A1.cols());
		HH.row(0) = A1;
		HH.middleRows(1, HH.rows() - 2) = H;
		HH.row(HH.rows() - 1) = Aend;

		lamb = HH.lu().solve(fff);

		MatrixXd uu0 = Axx*lamb;
		deri1 = D1_mid*lamb;
		deri2 = D2_mid*lamb;
		deri3 = D3_mid*lamb;

		double Ptop = deri3.maxCoeff();

		double Pend = deri3.minCoeff();

		MatrixXd::Index I1Row, I1Col;
		deri3.maxCoeff(&I1Row, &I1Col);

		MatrixXd::Index I2Row, I2Col;
		deri3.minCoeff(&I2Row, &I2Col);

		double part1Length = (I1Row < I2Row) ? I1Row : I2Row;
		double part2Length = (I1Row > I2Row) ? I1Row : I2Row;

		VectorXd vderi3 = deri3.col(0);
		MatrixXd part1 = VectorUtil::Diff(vderi3.block(0, 0, part1Length, 1));
		MatrixXd prediff = vderi3.block(part1Length, 0, part2Length - part1Length + 1, 1);
		MatrixXd part2 = VectorUtil::Diff(prediff);

		MatrixXd part3 = VectorUtil::Diff(vderi3.block(part2Length, 0, vderi3.rows() - part2Length, 1));

		MatrixXd zeros = MatrixXd::Zero(part1.rows(), part1.cols());
		MatrixXd ones = MatrixXd::Ones(part1.rows(), part1.cols());
		MatrixXd II1 = (part1.array() >= zeros.array()).select(ones, zeros);

		zeros = MatrixXd::Zero(part2.rows(), part2.cols());
		ones = MatrixXd::Ones(part2.rows(), part2.cols());
		MatrixXd II2 = (part2.array() >= zeros.array()).select(ones, zeros);

		zeros = MatrixXd::Zero(part3.rows(), part3.cols());
		ones = MatrixXd::Ones(part3.rows(), part3.cols());
		MatrixXd II3 = (part3.array() >= zeros.array()).select(ones, zeros);

		double min = deri2.minCoeff();
		double II1sum = II1.sum();
		double II2sum = II2.sum();
		double II3sum = II3.sum();
		double p1size = part1.size();
		double p2size = part2.size();
		double p3size = part3.size();
		cout << "iteration:" << count << setprecision(8) << " T:" << Tdone;
		if (min >= 0)
		{
			//	Approximation of Speed is monotonic in subintervals
			if (II1.sum() == 0 || II1.sum() == part1.size())
			{
				if (II2.sum() == 0 || II2.sum() == part2.size())
				{
					if (II3.sum() == 0 || II3.sum() == part3.size())
					{
						break;
					}
				}
			}
		}
		cout << "\r";
		count++;

	}
	VectorXd t(1);
	t(0) = Tdone;
	vector<VectorXd> res = { x, lamb, c, t };
	cout << "\r\n" << "Total Iterations:" << count << "\r\n";
	return res;

}

SmoothInitial Leicester::MoL::MethodOfLinesND(Params p, MatrixXd correlation)
{
	stringstream ssinx1;
	stringstream ssinx2;
	for (int i = 0; i < p.inx1.rows(); i++)
	{
		ssinx1 << p.inx1[i] << "%";
		ssinx2 << p.inx2[i] << "%";
	}

	stringstream ssX;
	ssX << setprecision(16) << "SmoothInitial_EuroCall_" << p.T << "_" << p.Tend << "_" << p.dt << "_" << p.K << "_" << p.r << "_" << p.sigma << "_" << p.theta << "_" << ssinx1.str() << "_" << ssinx2.str() << "_X.dat";
	stringstream ssU;
	ssU << setprecision(16) << "SmoothInitial_EuroCall_" << p.T << "_" << p.Tend << "_" << p.dt << "_" << p.K << "_" << p.r << "_" << p.sigma << "_" << p.theta << "_" << ssinx1.str() << "_" << ssinx2.str() << "_U.dat";
	stringstream ssT;
	ssT << setprecision(16) << "SmoothInitial_EuroCall_" << p.T << "_" << p.Tend << "_" << p.dt << "_" << p.K << "_" << p.r << "_" << p.sigma << "_" << p.theta << "_" << ssinx1.str() << "_" << ssinx2.str() << "_T.dat";

	string fileX = ssX.str();
	ifstream x(fileX.c_str());
	string fileU = ssU.str();
	ifstream u(fileU.c_str());
	string fileT = ssT.str();
	ifstream t(fileT.c_str());

	if (x.good() & u.good() & t.good())
	{
		x.close();
		MatrixXd mX = Common::ReadBinary(fileX, 32769, correlation.rows());
		VectorXd X(Map<VectorXd>(mX.data(), mX.cols()*mX.rows()));
		u.close();
		MatrixXd mU = Common::ReadBinary(fileU, 32769, correlation.rows());
		VectorXd U(Map<VectorXd>(mU.data(), mU.cols()*mU.rows()));
		t.close();
		MatrixXd mT = Common::ReadBinary(fileT, 1, 1);
		VectorXd T(Map<VectorXd>(mT.data(), mT.cols()*mT.rows()));
		SmoothInitial result;
		result.S = X;
		result.U = U;
		result.T = T[0];
		return result;
	}
	else
	{
		SmoothInitial smoothInitial = MethodOfLinesND(p.T, p.Tdone, p.Tend, p.dt, p.K, p.r, p.sigma, p.theta, p.inx1, p.inx2, correlation);
		//MatrixXd smthX(32769, smoothInitial.size());
		//MatrixXd smthU(32769, smoothInitial.size());
		//int col = 0;
		//for (auto s : smoothInitial)
		//{
		//	smthX.col(col) = s[0];
		//	smthU.col(col) = s[1];
		//	col++;
		//}
		Common::WriteToBinary(fileX, smoothInitial.S);
		Common::WriteToBinary(fileU, smoothInitial.U);
		MatrixXd T(1, 1);
		T << smoothInitial.T;
		Common::WriteToBinary(fileT, T);
		return smoothInitial;
	}

}

SmoothInitial Leicester::MoL::MethodOfLinesND(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, VectorXd inx1, VectorXd inx2, MatrixXd correlation)
{
	vector<vector<VectorXd>> result;
	cout << "MethodOfLines for N-D European Call Option" << endl;
	SmoothInitial initial = EuroCallOptionND(T, Tdone, Tend, dt, K, r, sigma, theta, correlation);

	MatrixXd a1 = MatrixXd::Ones(initial.C.rows(), initial.C.cols());
	double Smin = 0;
	double Smax = 3 * K;
	double twoPower15Plus1 = pow(2, 15) + 1;

	VectorXd X_ini = VectorXd::LinSpaced(twoPower15Plus1, Smin, Smax);
	int n = initial.TestNodes.rows();
	MatrixXd phi = MatrixXd::Ones(X_ini.size(), n);
	for (int i = 0; i < n; i++)
	{
		vector<vector<MatrixXd>> rbf = RBF::MultiQuadricND(X_ini, initial.TestNodes.row(i), a1, initial.C.row(i) );
		phi.col(i) = rbf[0][0].col(0);
	}

	VectorXd U_ini = phi * initial.Lambda;
	initial.S = X_ini;
	initial.U = U_ini;
		
	return initial;
}

SmoothInitial Leicester::MoL::EuroCallOptionND(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, MatrixXd correlation)
{
	//vector<vector<VectorXd>> result;
	int assets = correlation.rows();
	BasketOption *option = new BasketOption(K, T, correlation);
	int N_uniform = 5000*assets;
	
	MatrixXd N(1, assets);
	N.fill(N_uniform);
	
	vector<MatrixXd> setup = SetupBasket(assets, N_uniform, N_uniform/5, 0.2, K);
	MatrixXd inx1 = setup[0];
	MatrixXd inx2 = setup[1];
	MatrixXd testNodes = setup[2];
	MatrixXd centralNodes = setup[3];
	MatrixXd aroundStrikeNodes = setup[4];

	VectorXd u0 = option->PayOffFunction(testNodes);
	VectorXd dx = MatrixUtil::Diff(testNodes);
	//Common::saveArray(d, "ND_d.txt");
	//Common::saveArray(dx, "ND_dx.txt");
	const double inf = numeric_limits<double>::infinity();

	VectorXd pushed = VectorUtil::Push(dx, inf);
	VectorXd queued = VectorUtil::Queue(dx, inf);
	VectorXd c = 2 * (pushed.array() >= queued.array()).select(queued, pushed);
	//Common::saveArray(c, "ND_c.txt");
	

	int tnRows = testNodes.rows();
	int asRows = aroundStrikeNodes.rows();
	MatrixXd D1_mid = MatrixXd::Zero(asRows, tnRows);
	MatrixXd D2_mid = MatrixXd::Zero(asRows, tnRows);
	MatrixXd D3_mid = MatrixXd::Zero(asRows, tnRows);

	MatrixXd A = MatrixXd::Zero(tnRows, tnRows);
	MatrixXd D1 = MatrixXd::Zero(tnRows, tnRows);
	MatrixXd D2 = MatrixXd::Zero(tnRows, tnRows);

	MatrixXd Axx = MatrixXd::Zero(centralNodes.rows(), tnRows);
	MatrixXd a1 = MatrixXd::Ones(c.rows(), testNodes.cols());
	MatrixXd S = option->PayOffS(aroundStrikeNodes);
	Common::saveArray(S, "S.txt");
	MatrixXd Sc = option->PayOffS(centralNodes);
	delete option;
	cout << "MoL RBF Kernel interpolation\r\n";
	for (int j = 0; j < tnRows; j++)
	{
		cout << j+1 << "/" << tnRows;
		vector<vector<MatrixXd>> vAxx = RBF::MultiQuadricND(centralNodes, testNodes.row(j), a1, c.row(j));
		Axx.col(j) = vAxx[0][0].col(0);
		//Common::saveArray(vAxx[0][0], "ND_Axx.txt");

		vector<vector<MatrixXd>> vAx = RBF::MultiQuadricND(testNodes, testNodes.row(j), a1, c.row(j));
		A.col(j) = vAx[0][0].col(0);
		//Common::saveArray(A, "ND_A.txt");

		for (auto m : vAx[1])
			D1.col(j) += m.col(0);
		//Common::saveArray(D1, "ND_D1.txt");

		for (auto m : vAx[2]) //diagonal
			D2.col(j) += m.col(0);
		//
		//Common::saveArray(D2, "ND_D2.txt");
		for (auto m : vAx[3]) // upper triangle
			D2.col(j) += (2 * m.col(0));
		//Common::saveArray(D2, "ND_D2.txt");

		vector<vector<MatrixXd>> vAE = RBF::MultiQuadricND(aroundStrikeNodes, testNodes.row(j), a1, c.row(j));
		for (auto m : vAE[1])
			D1_mid.col(j) += m.col(0);
		//Common::saveArray(D1_mid, "ND_D1_mid.txt");

		for (auto m : vAE[2]) //diagonal
			D2_mid.col(j) += m.col(0);
		for (auto m : vAE[3]) //upper triangle
			D2_mid.col(j) += ( 2 * m.col(0)) ;
		//Common::saveArray(D2_mid, "ND_D2_mid.txt");

		//D3_mid.col(j) = vAE[3].col(0);
		D1_mid.col(j) = D1_mid.col(j).array() / S.array();
		//Common::saveArray(D1_mid, "ND_D1_mid.txt");
		
		D2_mid.col(j) = D2_mid.col(j).array() / (S.array() * S.array());
		if (j != tnRows -1 )
			cout << "\r";
		else
			cout << endl;
	}

	for (int i=0; i< D2_mid.cols(); i++)
		D3_mid.col(i) = -D2_mid.col(i).array() / S.col(0).array();

	//Common::saveArray(A, "A.txt");
	//Common::saveArray(u0, "u0.txt");
	//Common::saveArray(D1, "ND_D1.txt");
	//Common::saveArray(D2, "ND_D2.txt");
	//Common::saveArray(D1_mid, "ND_D1_mid.txt");
	//Common::saveArray(D2_mid, "ND_D2_mid.txt");
	VectorXd lamb = A.lu().solve(u0);

	MatrixXd uu0 = Axx*lamb;
	MatrixXd deri1 = D1_mid*lamb;
	MatrixXd deri2 = D2_mid*lamb;
	MatrixXd deri3 = D3_mid*lamb;

	MatrixXd A1 = A.row(1);
	MatrixXd Aend = A.row(A.rows() - 1);
	MatrixXd a = A.block(1, 0, A.rows() - 2, A.cols());
	MatrixXd d1 = D1.block(1, 0, D1.rows() - 2, D1.cols());
	MatrixXd d2 = D2.block(1, 0, D2.rows() - 2, D2.cols());

	MatrixXd P = a * r - 0.5 * (sigma * sigma) * d2 - r * d1;
	MatrixXd H = a + dt * (1 - theta)* P;
	MatrixXd G = a - dt * theta * P;

	int count = 0;
	cout << "MoL Iterative solver\r\n";
	while (Tend - Tdone > 1E-8)
	{
		Tdone += dt;
		MatrixXd g = G* lamb;

		//VectorXd fff = VectorUtil::PushAndQueue(0, (VectorXd)g.col(0), inx2 - exp(-r*Tdone)*K);
		MatrixXd g1 = PushAndQueueBoundaries(g, inx2.transpose(),r, Tdone, K);
		VectorXd fff = Map<VectorXd>(g1.data(), g1.cols() * g1.rows());
		//VectorXd fff = VectorUtil::PushAndQueue(0, v, inx2[n] - exp(-r*Tdone)*K);

		MatrixXd HH(A1.cols(), A1.cols());
		HH.row(0) = A1;
		HH.middleRows(1, HH.rows() - 2) = H;
		HH.row(HH.rows() - 1) = Aend;

		//Common::saveArray(HH, "HH.txt");
		//Common::saveArray(fff, "fff.txt");

		lamb = HH.lu().solve(fff);
		//Common::saveArray(lamb, "lamb.txt");
		MatrixXd uu0 = Axx*lamb;
		deri1 = D1_mid*lamb;
		//Common::saveArray(deri1, "deri1.txt");
		deri2 = D2_mid*lamb;
		//Common::saveArray(deri2, "deri2.txt");
		deri3 = D3_mid*lamb;
		MatrixXd SK = S / K;
		MatrixXd d1 = ( SK.array().log() + (r + sigma * sigma / 2) * (Tdone) ) / (sigma * sqrt(Tdone));
		deri3 = deri3.array() * (1 + ( d1 / (sigma * sqrt(Tdone)) ).array() );

		MatrixXd plot1(S.rows(), 2);
		plot1.col(0) = S.col(0);
		plot1.col(1) = deri2.col(0);
		//wcout << Common::printMatrix(plot1) << endl;
		MatrixXd sorted1 = sortMatrix(plot1);
		//wcout << Common::printMatrix(sorted1) << endl;
		plot1 = TakeMeans(sorted1);
		//wcout << Common::printMatrix(plot1) << endl;
		VectorXd vderi2 = plot1.col(1);
		

		MatrixXd plot2(S.rows(), 2);
		plot2.col(0) = S.col(0);
		plot2.col(1) = deri3.col(0);
		MatrixXd sorted2 = sortMatrix(plot2);
		plot2 = TakeMeans(sorted2);
		VectorXd vderi3 = plot2.col(1);

		wcout << Common::printMatrix(vderi3) << endl;

		double Ptop = vderi3.maxCoeff();
		double Pend = vderi3.minCoeff();

		MatrixXd::Index TopRow, TopCol;
		vderi3.maxCoeff(&TopRow, &TopCol);

		MatrixXd::Index BottomRow, BottomCol;
		vderi3.minCoeff(&BottomRow, &BottomCol);

		double part1Length = (TopRow < BottomRow) ? TopRow : BottomRow; //the region between 0 and peak speed
		double part2Length = (TopRow > BottomRow) ? TopRow : BottomRow; //the region between peak speed and lowest speed

		
		MatrixXd part1 = VectorUtil::Diff(vderi3.block(0, 0, part1Length, 1));
		MatrixXd prediff = vderi3.block(part1Length, 0, part2Length - part1Length + 1, 1);
		MatrixXd part2 = VectorUtil::Diff(prediff);

		MatrixXd part3 = VectorUtil::Diff(vderi3.block(part2Length, 0, vderi3.rows() - part2Length, 1));

		MatrixXd zeros = MatrixXd::Zero(part1.rows(), part1.cols());
		MatrixXd ones = MatrixXd::Ones(part1.rows(), part1.cols());
		MatrixXd II1 = (part1.array() >= zeros.array()).select(ones, zeros);

		zeros = MatrixXd::Zero(part2.rows(), part2.cols());
		ones = MatrixXd::Ones(part2.rows(), part2.cols());
		MatrixXd II2 = (part2.array() >= zeros.array()).select(ones, zeros);

		zeros = MatrixXd::Zero(part3.rows(), part3.cols());
		ones = MatrixXd::Ones(part3.rows(), part3.cols());
		MatrixXd II3 = (part3.array() >= zeros.array()).select(ones, zeros);

		double min = vderi2.minCoeff();
		double II1sum = II1.sum();
		double II2sum = II2.sum();
		double II3sum = II3.sum();
		double p1size = part1.size();
		double p2size = part2.size();
		double p3size = part3.size();
		wstringstream wss;
		wss << "iteration:" << count << setprecision(8) << " T:" << Tdone;
		wcout << wss.str();
		if (min >= 0)
		{
			//	Approximation of Speed is monotonic in subintervals
			if (II1.sum() == 0 || II1.sum() == part1.size())
			{
				if (II2.sum() == 0 || II2.sum() == part2.size())
				{
					if (II3.sum() == 0 || II3.sum() == part3.size())
					{
						break;
					}
				}
			}
		}
		cout << "\r";
		count++;

	}
	cout << "\r\n" << "Total Iterations:" << count << "\r\n";
	//vector<MatrixXd> res = { testNodes, lamb, c };
	//result.push_back(res);
	
	SmoothInitial result(Tdone, testNodes, lamb, c);

	return result;

}

MatrixXd Leicester::MoL::PushAndQueueBoundaries(MatrixXd A, VectorXd inx2, double r, double Tdone, double K)
{
	double push = 0;

	MatrixXd result(A.rows() + 2, A.cols());

	for (int j = 0; j < A.cols(); j++)
	{
		result(0, j) = push;
		for (int i = 1; i < A.rows(); i++)
		{
			result(i, j) = A(i - 1, j);
		}
		double queue = inx2[j] - exp(-r*Tdone)*K;
		result(A.rows(), j) = queue;
	}
	return result;

}

vector<vector<VectorXd>> Leicester::MoL::MethodOfLinesND_ODE(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, VectorXd inx1, VectorXd inx2, MatrixXd correlation)
{
	vector<vector<VectorXd>> result;
	cout << "MethodOfLines for N-D European Call Option" << endl;
	vector<vector<VectorXd>> prices = EuroCallOptionND_ODE(T, Tdone, Tend, dt, K, r, sigma, theta, inx1, inx2, correlation);

	for (auto price : prices)
	{
		VectorXd x = price[0];
		VectorXd lamb = price[1];
		VectorXd c = price[2];
		double Smin = 0;
		double Smax = 3 * K;
		double twoPower15Plus1 = pow(2, 15) + 1;

		VectorXd X_ini = VectorXd::LinSpaced(twoPower15Plus1, Smin, Smax);

		MatrixXd phi = MatrixXd::Ones(X_ini.size(), x.size());
		for (int i = 0; i < x.size(); i++)
		{
			vector<MatrixXd> rbf = RBF::MultiQuadric1D(X_ini, x(i), c(i));
			phi.col(i) = rbf[0].col(0);
		}

		VectorXd U_ini = phi * lamb;
		vector<VectorXd> r = { X_ini, U_ini };
		result.push_back(r);
	}
	return result;
}

vector<vector<VectorXd>> Leicester::MoL::EuroCallOptionND_ODE(double T, double Tdone, double Tend, double dt, double K, double r, double sigma, double theta, VectorXd inx1, VectorXd inx2, MatrixXd correlation)
{
	vector<vector<VectorXd>> result;
	int dimensions = correlation.rows();
	BasketOption *option = new BasketOption(K, T, correlation);
	int N_uniform = 5000;
	
	MatrixXd X(N_uniform, inx1.rows());
	for (int i = 0; i < inx1.rows(); i++)
		X.col(i) = VectorXd::LinSpaced(N_uniform, inx1[i], inx2[i]);
	MatrixXd N(1, dimensions);
	N.fill(N_uniform);
	
	VectorXd u0 = option->PayOffFunction(X);
	delete option;


	MatrixXd d(N_uniform, dimensions);
	for (int n = 0; n < N.cols(); n++) //N.Cols() is #dimensions
	{
		VectorXd linearDimension = VectorXd::LinSpaced(N_uniform, inx1[n], inx2[n]);
		d.col(n) = linearDimension;
	}
	MatrixXd dx = MatrixUtil::Diff(d);
	//Common::saveArray(d, "ND_d.txt");
	//Common::saveArray(dx, "ND_dx.txt");
	const double inf = numeric_limits<double>::infinity();

	MatrixXd pushed = MatrixUtil::PushRows(dx, inf);
	MatrixXd queued = MatrixUtil::QueueRows(dx, inf);
	VectorXd c = 2 * (pushed.array() >= queued.array()).select(queued, pushed);
	//Common::saveArray(c, "ND_c.txt");

	
	N.fill(1000);
	
	MatrixXd XX(1000, inx1.rows());
	for (int i = 0; i < inx1.rows(); i++)
		XX.col(i) = VectorXd::LinSpaced(1000, 0, 3*K);

	//Common::saveArray(X, "ND_X.txt");
	//Common::saveArray(XX, "ND_XX.txt");
	MatrixXd IT = ((1.2 * K >= X.array()) && (X.array() >= 0.8 * K)).select(X, 0);
	//Common::saveArray(IT, "ND_IT.txt");
	MatrixXd AroundE = MatrixUtil::Select(IT, 0);

	int n = X.size();
	MatrixXd D1_mid = MatrixXd::Zero(AroundE.rows(), n);
	MatrixXd D2_mid = MatrixXd::Zero(AroundE.rows(), n);
	MatrixXd D3_mid = MatrixXd::Zero(AroundE.rows(), n);

	MatrixXd A = MatrixXd::Zero(n, n);
	MatrixXd D1 = MatrixXd::Zero(n, n);
	MatrixXd D2 = MatrixXd::Zero(n, n);

	MatrixXd Axx = MatrixXd::Zero(XX.rows(), n);
	MatrixXd a1 = MatrixXd::Ones(c.rows(), c.cols());

	for (int j = 0; j < n; j++)
	{
		vector<MatrixXd> vAxx = RBF::MultiQuadricND_ODE(XX, X.row(j), a1, c.row(j));
		Axx.col(j) = vAxx[0].col(0);
		//Common::saveArray(Axx, "ND_Axx.txt");

		vector<MatrixXd> vAx = RBF::MultiQuadricND_ODE(X, X.row(j), a1, c.row(j));
		A.col(j) = vAx[0].col(0);
		//Common::saveArray(A, "ND_A.txt");
		D1.col(j) = vAx[1].col(0);
		//Common::saveArray(D1, "ND_D1.txt");
		D2.col(j) = vAx[2].col(0);
		//Common::saveArray(D2, "ND_D2.txt");
		vector<MatrixXd> vAE = RBF::MultiQuadricND_ODE(AroundE, X.row(j), a1, c.row(j));
		D1_mid.col(j) = vAE[1].col(0);
		//Common::saveArray(D1_mid, "ND_D1_mid.txt");
		D2_mid.col(j) = vAE[2].col(0);
		//Common::saveArray(D2_mid, "ND_D2_mid.txt");
		D3_mid.col(j) = vAE[3].col(0);
		//Common::saveArray(D3_mid, "ND_D3_mid.txt");
		D1_mid.col(j) = D1_mid.col(j).array() / AroundE.array();
		D2_mid.col(j) = D2_mid.col(j).array() / (AroundE.array() * AroundE.array());
	}

	Common::saveArray(A, "A.txt");
	Common::saveArray(u0, "u0.txt");
	VectorXd lamb = A.lu().solve(u0);

	MatrixXd uu0 = Axx*lamb;
	MatrixXd deri1 = D1_mid*lamb;
	MatrixXd deri2 = D2_mid*lamb;

	MatrixXd deri3 = D3_mid*lamb;

	MatrixXd A1 = A.row(1);
	MatrixXd Aend = A.row(A.rows() - 1);
	MatrixXd a = A.block(1, 0, A.rows() - 2, A.cols());
	MatrixXd d1 = D1.block(1, 0, D1.rows() - 2, D1.cols());
	MatrixXd d2 = D2.block(1, 0, D2.rows() - 2, D2.cols());

	MatrixXd P = a * r - 0.5 * (sigma * sigma) * d2 - r * d1;
	Common::saveArray(P, "ND_P.txt");
	MatrixXd H = a + dt * (1 - theta)* P;
	Common::saveArray(H, "ND_H.txt");
	MatrixXd G = a - dt * theta * P;
	Common::saveArray(G, "ND_G.txt");

	int count = 0;
	cout << "MoL Iterative solver\r\n";
	while (Tend - Tdone > 1E-8)
	{
		Tdone += dt;
		MatrixXd g = G* lamb;

		//VectorXd fff = VectorUtil::PushAndQueue(0, (VectorXd)g.col(0), inx2 - exp(-r*Tdone)*K);
		g = PushAndQueueBoundaries(g, inx2, r, Tdone, K);
		VectorXd fff = Map<VectorXd>(g.data(), g.cols() * g.rows());
		//VectorXd fff = VectorUtil::PushAndQueue(0, v, inx2[n] - exp(-r*Tdone)*K);

		MatrixXd HH(A1.cols(), A1.cols());
		HH.row(0) = A1;
		HH.middleRows(1, HH.rows() - 2) = H;
		HH.row(HH.rows() - 1) = Aend;

		//Common::saveArray(HH, "HH.txt");
		//Common::saveArray(fff, "fff.txt");

		lamb = HH.lu().solve(fff);

		MatrixXd uu0 = Axx*lamb;
		deri1 = D1_mid*lamb;
		deri2 = D2_mid*lamb;
		deri3 = D3_mid*lamb;

		double Ptop = deri3.maxCoeff();

		double Pend = deri3.minCoeff();

		MatrixXd::Index I1Row, I1Col;
		deri3.maxCoeff(&I1Row, &I1Col);

		MatrixXd::Index I2Row, I2Col;
		deri3.minCoeff(&I2Row, &I2Col);

		double part1Length = (I1Row < I2Row) ? I1Row : I2Row;
		double part2Length = (I1Row > I2Row) ? I1Row : I2Row;

		VectorXd vderi3 = deri3.col(0);
		MatrixXd part1 = VectorUtil::Diff(vderi3.block(0, 0, part1Length, 1));
		MatrixXd prediff = vderi3.block(part1Length, 0, part2Length - part1Length + 1, 1);
		MatrixXd part2 = VectorUtil::Diff(prediff);

		MatrixXd part3 = VectorUtil::Diff(vderi3.block(part2Length, 0, vderi3.rows() - part2Length, 1));

		MatrixXd zeros = MatrixXd::Zero(part1.rows(), part1.cols());
		MatrixXd ones = MatrixXd::Ones(part1.rows(), part1.cols());
		MatrixXd II1 = (part1.array() >= zeros.array()).select(ones, zeros);

		zeros = MatrixXd::Zero(part2.rows(), part2.cols());
		ones = MatrixXd::Ones(part2.rows(), part2.cols());
		MatrixXd II2 = (part2.array() >= zeros.array()).select(ones, zeros);

		zeros = MatrixXd::Zero(part3.rows(), part3.cols());
		ones = MatrixXd::Ones(part3.rows(), part3.cols());
		MatrixXd II3 = (part3.array() >= zeros.array()).select(ones, zeros);

		double min = deri2.minCoeff();
		double II1sum = II1.sum();
		double II2sum = II2.sum();
		double II3sum = II3.sum();
		double p1size = part1.size();
		double p2size = part2.size();
		double p3size = part3.size();
		wstringstream wss;
		wss << "iteration:" << count << setprecision(8) << " T:" << Tdone;
		Common::Logger(wss.str());
		if (min >= 0)
		{
			//	Approximation of Speed is monotonic in subintervals
			if (II1.sum() == 0 || II1.sum() == part1.size())
			{
				if (II2.sum() == 0 || II2.sum() == part2.size())
				{
					if (II3.sum() == 0 || II3.sum() == part3.size())
					{
						break;
					}
				}
			}
		}
		cout << "\r";
		count++;

	}
	cout << "Total Iterations:" << count << "\r\n";
	VectorXd t(1);
	t(0) =  Tdone;
	vector<VectorXd> res = { X.col(n), lamb, c, t };
	result.push_back(res);

	return result;

}

vector<MatrixXd> Leicester::MoL::SetupBasket(int assets, int testNodeNumber, int centralNodeNumber, double aroundStrikeRange, double strike)
{
	MatrixXd inx1 = -1 * MatrixXd::Ones(1, assets) *strike;
	MatrixXd inx2 = 6 * MatrixXd::Ones(1, assets) *strike;

	vector<MatrixXd> testNodes = TestNodes::GenerateTestNodes(testNodeNumber, inx1.row(0).transpose(), inx2.row(0).transpose(), assets);
	//Common::saveArray(testNodes, "testNodes.txt");
	vector<MatrixXd> cetralNodes = TestNodes::GenerateTestNodes(centralNodeNumber, inx1.row(0).transpose(), inx2.row(0).transpose(), assets);
	//Common::saveArray(cetralNodes, "cetralNodes.txt");
	MatrixXd aroundStrikeNodes = BasketOption::NodesAroundStrikeFromGrid(testNodes[0], strike, aroundStrikeRange);
	//Common::saveArray(aroundStrikeNodes, "aroundStrikeNodes.txt");
	return { inx1, inx2, testNodes[1], cetralNodes[1], aroundStrikeNodes };

}

MatrixXd Leicester::MoL::TakeMeans(MatrixXd M)
{
	double last =  - numeric_limits<double>::infinity();
	double sum = 0;
	int ni = 0;
	MatrixXd Na = MatrixXd::Zero(M.rows(), M.cols());
	int count = 0;
	double current = -numeric_limits<double>::infinity();
	for (int i =0 ; i<= M.rows(); i++)
	{
		if (i < M.rows())
			current = M(i,0);
		if (( i != 0 && (abs(current - last) > DBL_EPSILON)) | i == M.rows())
		{
			Na(ni, 0) = M(i-1, 0);
			Na(ni, 1) = sum / count;
			ni++;
			count = 1;
			if (i < M.rows())
				sum = M(i, 1);
		}
		else
		{
			count++;
			sum += M(i,1);
		}
		last = current;
	}

	MatrixXd N = Na.block(0, 0, ni, 2);
	return N;
}

