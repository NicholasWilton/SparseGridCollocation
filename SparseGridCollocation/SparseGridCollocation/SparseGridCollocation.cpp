// SparseGridGollocation.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "PPP.h"
#include "SmoothInitialX.h"
#include "SparseGridCollocation.h"
#include "windows.h"
#include "Common.h"

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::UpLoType;
using Eigen::Map;
using namespace Eigen;
using namespace std;

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>


using namespace Microsoft::VisualStudio::CppUnitTestFramework;


double prod(vector<double> x)
{
	double prod = 1.0;

	for (unsigned int i = 0; i < x.size(); i++)
		prod *= x[i];
	return prod;
}

template <typename T = double>
vector<T> linspace(T a, T b, size_t N) 
{
	T h = (b - a) / static_cast<T>(N - 1);
	vector<T> xs(N);
	typename vector<T>::iterator x;
	T val;
	for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
		*x = val;
	return xs;
}



vector<MatrixXd> SparseGridCollocation::mqd2(MatrixXd TP, MatrixXd CN, MatrixXd A, MatrixXd C)
{
	vector<MatrixXd> *result = new vector<MatrixXd>();
	//[Num, ~] = size(CN);
	int Num = CN.rows();
	//[N, ~] = size(TP);
	int N = TP.rows();
	//[D, Dt, Dx, Dxx] = deal(ones(N, Num));
	
	MatrixXd *D = new MatrixXd(N, Num);
	D->fill(1.0);
	MatrixXd Dt(N, Num);
	Dt.fill(1.0);
	MatrixXd Dx(N, Num);
	Dx.fill(1.0);
	MatrixXd Dxx(N, Num);
	Dxx.fill(1.0);

	//for j = 1:Num
	for (int j = 0; j < Num; j++)
	{
		
		//	FAI1 = exp(-(A(1, 1)*(TP(:, 1) - CN(j, 1))). ^ 2 / C(1, 1) ^ 2);
		VectorXd a1 = A(0, 0)*(TP.col(0).array() - CN(j, 0));
		Logger::WriteMessage(Common::printMatrix(a1).c_str());

		VectorXd b1 = -(a1.array() * a1.array()) / (C(0, 0) *C(0, 0));
		Logger::WriteMessage(Common::printMatrix(b1).c_str());

		VectorXd FAI1 = b1.array().exp();
		Logger::WriteMessage(Common::printMatrix(FAI1).c_str());

		//FAI2 = exp(-(A(1, 2)*(TP(:, 2) - CN(j, 2))). ^ 2 / C(1, 2) ^ 2);
		VectorXd a2 = A(0, 1)*(TP.col(1).array() - CN(j, 1));
		Logger::WriteMessage(Common::printMatrix(a2).c_str());
		
		VectorXd b2 = -(a2.array() * a2.array()) / (C(0, 1) *C(0, 1));
		Logger::WriteMessage(Common::printMatrix(b2).c_str());
		
		VectorXd FAI2 = b2.array().exp();
		Logger::WriteMessage(Common::printMatrix(FAI2).c_str());
		//D(:, j) = FAI1.*FAI2;
		D->col(j) = FAI1.array() * FAI2.array();
		Logger::WriteMessage(Common::printMatrix(*D).c_str());

		//TODO: this is basically how Matlab handles overloading:
		//if nargout > 1

		//	Dt(:, j) = -2 * (A(1, 1) / C(1, 1)) ^ 2 * (TP(:, 1) - CN(j, 1)).*FAI1.*FAI2;
		VectorXd a3 = -2 * (A(0, 0) / C(0, 0)) * (A(0, 0) / C(0, 0)) * (TP.col(0).array() - CN(j, 0));
		VectorXd b3 = a3.array() * FAI1.array();
		VectorXd c3 = b3.array() * FAI2.array();
		Dt.col(j) = c3;
		Logger::WriteMessage(Common::printMatrix(Dt).c_str());

		//Dx(:, j) = TP(:, 2).*(-2 * (A(1, 2) / C(1, 2)) ^ 2 * (TP(:, 2) - CN(j, 2)).*FAI1.*FAI2);
		VectorXd a4 = -2 * (A(0, 1) / C(0, 1)) * (A(0, 1) / C(0, 1)) * (TP.col(1).array() - CN(j, 1));
		VectorXd b4 = TP.col(1).array() * a4.array() * FAI1.array();
		VectorXd c4 = b4.array() * FAI2.array();
		Dx.col(j) = c4;
		Logger::WriteMessage(Common::printMatrix(Dx).c_str());
		//Dxx(:, j) = TP(:, 2).^2.*((-2 * A(1, 2) ^ 2 / C(1, 2) ^ 2 + 4 * A(1, 2) ^ 4 * (TP(:, 2) - CN(j, 2)).^2. / C(1, 2) ^ 4).*FAI2.*FAI1);
		double sA = A(0, 1) * A(0, 1);
		double qA = A(0, 1) * A(0, 1) * A(0, 1) * A(0, 1);
		double sC = C(0, 1) * C(0, 1);
		double qC = C(0, 1) * C(0, 1) * C(0, 1) * C(0, 1);
		VectorXd dTpCn = TP.col(1).array() - CN(j, 1);

		VectorXd a5 = 4 * qA * (dTpCn.array() * dTpCn.array() / qC);
		Logger::WriteMessage(Common::printMatrix(a5).c_str());
		VectorXd b5 = -2 * sA / sC + a5.array();
		Logger::WriteMessage(Common::printMatrix(b5).c_str());
		VectorXd c5 = b5.array()  * FAI2.array() * FAI1.array();
		Logger::WriteMessage(Common::printMatrix(c5).c_str());
		VectorXd d5 = (TP.col(1).array() * TP.col(1).array()).array() * c5.array();
		Logger::WriteMessage(Common::printMatrix(d5).c_str());
		//VectorXd c5 = b5.array() * FAI2.array() * FAI1.array();
		Dxx.col(j) = d5;
		Logger::WriteMessage(Common::printMatrix(Dxx).c_str());
	}
	result->push_back(*D);
	result->push_back(Dt);
	result->push_back(Dx);
	result->push_back(Dxx);
	return *result;
}

vector<MatrixXd> SparseGridCollocation::shapelambda2D(double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N )
{
	//Num=prod(N);
	double num= N.prod();

	//t = linspace(0, tsec, N(1, 1));
	VectorXd t = VectorXd::LinSpaced(N(0, 0), 0, tsec);
	Logger::WriteMessage(Common::printMatrix(t).c_str());
	//x = linspace(inx1, inx2, N(1, 2));
	VectorXd x = VectorXd::LinSpaced(N(0, 1), inx1, inx2);
	Logger::WriteMessage(Common::printMatrix(x).c_str());

	//h1 = coef*tsec;
	double h1 = coef*tsec;
	//h2 = coef*(inx2 - inx1);
	double h2 = coef*(inx2 - inx1);
	
	//C = [h1, h2];
	//possible truncation here:
	MatrixXd c(1,2);
	c << h1, h2;
	Logger::WriteMessage(Common::printMatrix(c).c_str());
	//A = N - 1;
	MatrixXd a = N.array() - 1;
	Logger::WriteMessage(Common::printMatrix(a).c_str());
	//[XXX, YYY] = meshgrid(t, x);
	/*
	XXX = RowVectorXd::LinSpaced(1, 3, 3).replicate(5, 1);
	YYY = VectorXd::LinSpaced(10, 14, 5).replicate(1, 3);
	*/

	MatrixXd XXX = t.replicate(1, x.rows());
	Logger::WriteMessage(Common::printMatrix(XXX).c_str());
	MatrixXd YYY = x.replicate(1, t.rows());
	Logger::WriteMessage(Common::printMatrix(YYY).c_str());

	XXX.transposeInPlace();

	//YYY.transposeInPlace();

	
	VectorXd xxx(Map<VectorXd>(XXX.data(), XXX.cols()*XXX.rows()));
	//MatrixXd xxx = XXX.replicate(YYY.rows(), YYY.cols());
	Logger::WriteMessage(Common::printMatrix(xxx).c_str());
	//VectorXd yyy(Map<VectorXd>(YYY.data(), YYY.cols()*YYY.rows()).replicate(XXX.rows(), XXX.cols()));
	VectorXd yyy(Map<VectorXd>(YYY.data(), YYY.cols()*YYY.rows()));
	Logger::WriteMessage(Common::printMatrix(yyy).c_str());

	//TX = [XXX(:) YYY(:)];
	MatrixXd TX1(XXX.rows() * XXX.cols(), 2);// = new MatrixXd(15, 2);
	TX1 << xxx, yyy;
	Logger::WriteMessage(Common::printMatrix(TX1).c_str());

	//U=zeros(Num,1);
	VectorXd U = MatrixXd::Zero(num, 1);

	//int Num = TX1.rows();
	//int a = TX1.rows();
	//MatrixXd Dxx(a, Num);
	//Dxx.fill(1.0);

	//[ FAI, FAI_t, FAI_x, FAI_xx ] = mq2d( TX, TX, A, C );
	//MatrixXd* FAI = new MatrixXd(0, 0), *FAI_t = new MatrixXd(0, 0), *FAI_x = new MatrixXd(0, 0), *FAI_xx = new MatrixXd(0, 0);
	Logger::WriteMessage(Common::printMatrix(TX1).c_str());
	Logger::WriteMessage(Common::printMatrix(a).c_str());
	Logger::WriteMessage(Common::printMatrix(c).c_str());

	vector<MatrixXd> mqd = mqd2(TX1, TX1, a, c);

	Logger::WriteMessage(Common::printMatrix(c).c_str());

	MatrixXd FAI = mqd[0];
	Logger::WriteMessage(Common::printMatrix(FAI).c_str());
	MatrixXd FAI_t = mqd[1];
	Logger::WriteMessage(Common::printMatrix(FAI_t).c_str());
	MatrixXd FAI_x = mqd[2];
	Logger::WriteMessage(Common::printMatrix(FAI_x).c_str());
	MatrixXd FAI_xx = mqd[3];
	Logger::WriteMessage(Common::printMatrix(FAI_xx).c_str());


	//P = FAI_t + sigma ^ 2 * FAI_xx / 2 + r*FAI_x - r*FAI;
	MatrixXd pa = (sigma * sigma * FAI_xx.array() / 2);
	Logger::WriteMessage(Common::printMatrix(pa).c_str());
	MatrixXd pb = r*FAI_x.array() - r*FAI.array();
	Logger::WriteMessage(Common::printMatrix(pb).c_str());
	MatrixXd P = FAI_t.array() + pa.array() + pb.array();
	Logger::WriteMessage(Common::printMatrix(P).c_str());

	Logger::WriteMessage(Common::printMatrix(TX1).c_str());
	//for i=1:Num
	for (int i = 0; i < num; i++)
	{
		//if TX(i,2) == inx1 || TX(i,2) == inx2       
		if (abs(TX1(i, 1) - inx1) < DBL_EPSILON || abs(TX1(i, 1) - inx2) < DBL_EPSILON)
		{
			//P(i,:) = FAI(i,:);      
			P.row(i) = FAI.row(i);
			//U(i)=max( 0, TX(i,2) - E*exp( -r * (T-TX(i,1)) ) );
			double max = TX1(i, 1) - E*exp(-r * (T - TX1(i, 0)));
			U(i) = 0;
			if (max > 0)
				U(i) = max;

		}

		//if TX(i, 1) == tsec
		if (abs(TX1(i, 0) - tsec) < DBL_EPSILON)
		{
			//P(i, :) = FAI(i, :);
			P.row(i) = FAI.row(i);

			//U(i) = PPP(TX(i, :));
			U(i) = PPP::Calculate(TX1.row(i));
		}
	}
	//TX = &TX1;
	//[F, J] = lu(P);
	MatrixXd TX = TX1;
	Logger::WriteMessage(Common::printMatrix(P).c_str());
	Logger::WriteMessage(Common::printMatrix(U).c_str());

	PartialPivLU<MatrixXd> lu = PartialPivLU<MatrixXd>(P);
	//MatrixXd p = l.permutationP();
	MatrixXd J = lu.matrixLU().triangularView<UpLoType::Upper>();
	MatrixXd F = lu.matrixLU().triangularView<UpLoType::UnitLower>();;
	//MatrixXd Fa = p * F;
	//Hack: to get around the fact that Eigen doesn't compute the permutation matrix p correctly
	MatrixXd transform(15, 15);
	transform << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
	MatrixXd Fa = transform * F;
	
	Logger::WriteMessage(Common::printMatrix(transform).c_str());
	Logger::WriteMessage(Common::printMatrix(J).c_str()); 
	Logger::WriteMessage(Common::printMatrix(F).c_str());
	Logger::WriteMessage(Common::printMatrix(Fa).c_str());
	
	//Eigen also seems to solve with different rounding, maybe a double arithmetic issue:
	//Jlamda = F\U;
	MatrixXd Jlamda = Fa.lu().solve(U);
	//lamb = J\Jlamda;
	MatrixXd l = J.lu().solve(Jlamda);

	Logger::WriteMessage(Common::printMatrix(Jlamda).c_str());
	Logger::WriteMessage(Common::printMatrix(l).c_str());
	
	vector<MatrixXd> result = { l, TX, c, a };
	return result;
}

void interplant()
{
}




