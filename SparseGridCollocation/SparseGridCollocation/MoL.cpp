#include "stdafx.h"
#include "MoL.h"
#include "Common.h"
#include "EuropeanCallOption.h"
#include "VectorUtil.h"
#include "RBF.h"
#include <iomanip>
#include <fstream>

using namespace Eigen;
using namespace std;

Leicester::MoL::MoL()
{
}


Leicester::MoL::~MoL()
{
}



vector<VectorXd> Leicester::MoL::MethodOfLines(Params p)
{
	stringstream ssX;
	ssX << setprecision(16) << "SmoothInitial_EuroCall_" << p.T << "_" << p.Tdone << "_" << p.Tend << "_" << p.dt << "_" << p.K << "_" << p.r << "_" << p.sigma << "_" << p.theta << "_" << p.inx1 << "_" << p.inx2 << "_X.dat";
	stringstream ssU;
	ssU << setprecision(16) << "SmoothInitial_EuroCall_" << p.T << "_" << p.Tdone << "_" << p.Tend << "_" << p.dt << "_" << p.K << "_" << p.r << "_" << p.sigma << "_" << p.theta << "_" << p.inx1 << "_" << p.inx2 << "_U.dat";

	string fileX = ssX.str();
	ifstream x(fileX.c_str());
	string fileU = ssU.str();
	ifstream u(fileU.c_str());

	if (x.good() & u.good())
	{
		x.close();
		MatrixXd mX = Common::ReadBinary(fileX, 32769, 1);
		VectorXd X(Map<VectorXd>(mX.data(), mX.cols()*mX.rows()));
		u.close();
		MatrixXd mU = Common::ReadBinary(fileU, 32769, 1);
		VectorXd U(Map<VectorXd>(mU.data(), mU.cols()*mU.rows()));
		return { X, U };
	}
	else
	{
		vector<VectorXd> smoothInitial = MethodOfLines(p.T, p.Tdone, p.Tend, p.dt, p.K, p.r, p.sigma, p.theta, p.inx1, p.inx2);
		Common::WriteToBinary(fileX, smoothInitial[0]);
		Common::WriteToBinary(fileU, smoothInitial[1]);
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
	double Smin = 0;
	double Smax = 3 * K;
	double twoPower15Plus1 = pow(2, 15) + 1;

	VectorXd X_ini = VectorXd::LinSpaced(twoPower15Plus1, Smin, Smax);

	MatrixXd phi = MatrixXd::Ones(X_ini.size(), x.size());
	for (int i = 0; i < x.size(); i++)
	{
		vector<MatrixXd> rbf = RBF::mqd1(X_ini, x(i), c(i));
		phi.col(i) = rbf[0].col(0);
	}

	VectorXd U_ini = phi * lamb;

	return { X_ini, U_ini };
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

	VectorXd xx = VectorXd::LinSpaced(1000, 0, 3 * K);
	VectorXd IT = ((1.2 * K >= x.array()) && (x.array() >= 0.8 * K)).select(x, 0);
	double sumIT = IT.sum();
	VectorXd AroundE = VectorUtil::Select(IT, 0);

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
		vector<MatrixXd> vAxx = RBF::mqd1(xx, x[j], c[j]);
		Axx.col(j) = vAxx[0].col(0);

		vector<MatrixXd> vAx = RBF::mqd1(x, x[j], c[j]);
		A.col(j) = vAx[0].col(0);

		D1.col(j) = vAx[1].col(0);
		D2.col(j) = vAx[2].col(0);
		vector<MatrixXd> vAE = RBF::mqd1(AroundE, x[j], c[j]);
		D1_mid.col(j) = vAE[1].col(0);
		D2_mid.col(j) = vAE[2].col(0);

		D3_mid.col(j) = vAE[3].col(0);
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
	MatrixXd H = a + dt * (1 - theta)* P;
	MatrixXd G = a - dt * theta * P;

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
	cout << "Total Iterations:" << count << "\r\n";
	return { x, lamb, c };


}