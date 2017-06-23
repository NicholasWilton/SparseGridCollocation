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

SparseGridCollocation::SparseGridCollocation()
{
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

vector<MatrixXd> SparseGridCollocation::shapelambda2D_1(double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N,
	vector<MatrixXd> lamb2, vector<MatrixXd> TX2, vector<MatrixXd> C2, vector<MatrixXd> A2, vector<MatrixXd> lamb3, vector<MatrixXd> TX3, vector<MatrixXd> C3, vector<MatrixXd> A3)
{
	//Num=prod(N);
	double num = N.prod();

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
	MatrixXd c(1, 2);
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

	
	
	vector<MatrixXd> mqd = mqd2(TX1, TX1, a, c);
	MatrixXd FAI = mqd[0];
	MatrixXd FAI_t = mqd[1];
	MatrixXd FAI_x = mqd[2];
	MatrixXd FAI_xx = mqd[3];


	//P = FAI_t + sigma ^ 2 * FAI_xx / 2 + r*FAI_x - r*FAI;
	MatrixXd pa = (sigma * sigma * FAI_xx.array() / 2);
	MatrixXd pb = r*FAI_x.array() - r*FAI.array();
	MatrixXd P = FAI_t.array() + pa.array() + pb.array();

	//U=zeros(Num,1);
	VectorXd U = MatrixXd::Zero(num, 1);
	U = U - PDE(TX1, r, sigma, lamb2, TX2, C2, A2, lamb3, TX3, C3, A3);

	//for i=1:Num
	for (int i = 0; i < num; i++)
	{
		//if TX(i,2) == inx1 || TX(i,2) == inx2       
		if (abs(TX1(i, 1) - inx1) < DBL_EPSILON || abs(TX1(i, 1) - inx2) < DBL_EPSILON)
		{
			//P(i,:) = FAI(i,:);      
			P.row(i) = FAI.row(i);
			//       re - construct the boundary conditions
			//U(i) = max(0, TX(i, 2) - E*exp(-r * (T - TX(i, 1)))) - ...
			//	(inner_test(TX(i, 1), TX(i, 2), lamb3, TX3, C3, A3) - inner_test(TX(i, 1), TX(i, 2), lamb2, TX2, C2, A2));  % boundary condition

			double max = TX1(i, 1) - E*exp(-r * (T - TX1(i, 0)));
			U(i) = 0;
			if (max > 0)
				U(i) = max -( inner_test(TX1(i,0), TX1(i,1), lamb3, TX3, C3, A3 ) - inner_test(TX1(i, 0), TX1(i, 1), lamb2, TX2, C2, A2) );
			else
				U(i) = 0 - (inner_test(TX1(i, 0), TX1(i, 1), lamb3, TX3, C3, A3) - inner_test(TX1(i, 0), TX1(i, 1), lamb2, TX2, C2, A2));
		}

		//if TX(i, 1) == tsec
		if (abs(TX1(i, 0) - tsec) < DBL_EPSILON)
		{
			//P(i, :) = FAI(i, :);
			P.row(i) = FAI.row(i);

			//U(i) = PPP(TX(i, :)) - ...
				//(inner_test(TX(i, 1), TX(i, 2), lamb3, TX3, C3, A3) - inner_test(TX(i, 1), TX(i, 2), lamb2, TX2, C2, A2));
			U(i) = PPP::Calculate(TX1.row(i)) - (inner_test(TX1(i, 0), TX1(i, 1), lamb3, TX3, C3, A3) - inner_test(TX1(i, 0), TX1(i, 1), lamb2, TX2, C2, A2));
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

vector<MatrixXd> SparseGridCollocation::shapelambda2D_2(double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N,
	vector<MatrixXd> lamb2, vector<MatrixXd> TX2, vector<MatrixXd> C2, vector<MatrixXd> A2, 
	vector<MatrixXd> lamb3, vector<MatrixXd> TX3, vector<MatrixXd> C3, vector<MatrixXd> A3,
	vector<MatrixXd> lamb_3, vector<MatrixXd> TX_3, vector<MatrixXd> C_3, vector<MatrixXd> A_3,
	vector<MatrixXd> lamb4, vector<MatrixXd> TX4, vector<MatrixXd> C4, vector<MatrixXd> A4)
{
	//Num=prod(N);
	double num = N.prod();

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
	MatrixXd c(1, 2);
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



	vector<MatrixXd> mqd = mqd2(TX1, TX1, a, c);
	MatrixXd FAI = mqd[0];
	MatrixXd FAI_t = mqd[1];
	MatrixXd FAI_x = mqd[2];
	MatrixXd FAI_xx = mqd[3];


	//P = FAI_t + sigma ^ 2 * FAI_xx / 2 + r*FAI_x - r*FAI;
	MatrixXd pa = (sigma * sigma * FAI_xx.array() / 2);
	MatrixXd pb = r*FAI_x.array() - r*FAI.array();
	MatrixXd P = FAI_t.array() + pa.array() + pb.array();

	//U=zeros(Num,1);
	VectorXd U = MatrixXd::Zero(num, 1);
	U = U - PDE(TX1, r, sigma, lamb2, TX2, C2, A2, lamb3, TX3, C3, A3) 
		- PDE(TX1, r, sigma, lamb_3, TX_3, C_3, A_3, lamb4, TX4, C4, A4);

	//for i=1:Num
	for (int i = 0; i < num; i++)
	{
		//if TX(i,2) == inx1 || TX(i,2) == inx2       
		if (abs(TX1(i, 1) - inx1) < DBL_EPSILON || abs(TX1(i, 1) - inx2) < DBL_EPSILON)
		{
			//P(i,:) = FAI(i,:);      
			P.row(i) = FAI.row(i);
			//U(i) = max(0, TX(i, 2) - E*exp(-r * (T - TX(i, 1)))) - ...
				//(inner_test(TX(i, 1), TX(i, 2), lamb3, TX3, C3, A3) - inner_test(TX(i, 1), TX(i, 2), lamb2, TX2, C2, A2)) - ...
				//(inner_test(TX(i, 1), TX(i, 2), lamb4, TX4, C4, A4) - inner_test(TX(i, 1), TX(i, 2), lamb_3, TX_3, C_3, A_3));  % boundary condition

			double max = TX1(i, 1) - E*exp(-r * (T - TX1(i, 0)));
			U(i) = 0;
			if (max > 0)
				U(i) = max 
				- (inner_test(TX1(i, 0), TX1(i, 1), lamb3, TX3, C3, A3) - inner_test(TX1(i, 0), TX1(i, 1), lamb2, TX2, C2, A2))
				- (inner_test(TX1(i, 0), TX1(i, 1), lamb4, TX4, C4, A4) - inner_test(TX1(i, 0), TX1(i, 1), lamb_3, TX_3, C_3, A_3));
			else
				U(i) = max
				- (inner_test(TX1(i, 0), TX1(i, 1), lamb3, TX3, C3, A3) - inner_test(TX1(i, 0), TX1(i, 1), lamb2, TX2, C2, A2))
				- (inner_test(TX1(i, 0), TX1(i, 1), lamb4, TX4, C4, A4) - inner_test(TX1(i, 0), TX1(i, 1), lamb_3, TX_3, C_3, A_3));
		}

		//if TX(i, 1) == tsec
		if (abs(TX1(i, 0) - tsec) < DBL_EPSILON)
		{
			//P(i, :) = FAI(i, :);
			P.row(i) = FAI.row(i);

			//U(i) = PPP(TX(i, :)) - ...
			//        U(i) = PPP( TX(i,:) ) - ...
			//(inner_test(TX(i, 1), TX(i, 2), lamb3, TX3, C3, A3) - inner_test(TX(i, 1), TX(i, 2), lamb2, TX2, C2, A2)) - ...
			//	(inner_test(TX(i, 1), TX(i, 2), lamb4, TX4, C4, A4) - inner_test(TX(i, 1), TX(i, 2), lamb_3, TX_3, C_3, A_3));
			
			U(i) = PPP::Calculate(TX1.row(i))
				- (inner_test(TX1(i, 0), TX1(i, 1), lamb3, TX3, C3, A3) - inner_test(TX1(i, 0), TX1(i, 1), lamb2, TX2, C2, A2))
				- (inner_test(TX1(i, 0), TX1(i, 1), lamb4, TX4, C4, A4) - inner_test(TX1(i, 0), TX1(i, 1), lamb_3, TX_3, C_3, A_3));
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

MatrixXd SparseGridCollocation::primeNMatrix(int b, int d)
{
	//L = subnumber(b, d);
	MatrixXd L = subnumber(b, d);
	//[ch, ~] = size(L);
	int ch = L.rows();

	//N = ones(ch, d);
	MatrixXd N = MatrixXd::Ones(ch, d);
	//for i = 1:d
	for (int i = 0; i < d; i++)
		//	N(:, i) = 2.^L(:, i) + 1;
		for (int j = 0; j < L.rows(); j++)
			N(i, j) = pow(2, L(i, j)) + 1;
	//end

	Logger::WriteMessage(Common::printMatrix(N).c_str());
	return N;
}

vector<vector<MatrixXd>> SparseGridCollocation::interpolate(double coef, double tsec, int b, int d, double inx1, double inx2, double r, double sigma, double T, double E)
{
	vector<MatrixXd> Lamb;
	vector<MatrixXd> TX;
	vector<MatrixXd> C;
	vector<MatrixXd> A;

	MatrixXd N = primeNMatrix(b, d);

	//% calculate information on every sub grid
	//%parfor i = 1 : ch
	//for i = 1 : ch
	for (int i = 0; i < N.rows(); i++)
	{
		//	%if i <= ch
		//	[lamb{ i }, TX{ i }, C{ i }, A{ i }] = ...
		//	shapelambda2D(coef, tsec, r, sigma, T, E, inx1, inx2, N(i, :));
		vector<MatrixXd> res = shapelambda2D(coef, tsec, r, sigma, T, E, inx1, inx2, N.row(i));
		Lamb.push_back(res[0]);
		TX.push_back(res[1]);
		C.push_back(res[2]);
		A.push_back(res[3]);
		//
		//end
	}

	vector<vector<MatrixXd>> result;
	result.push_back(Lamb);
	result.push_back(TX);
	result.push_back(C);
	result.push_back(A);
	return result;
}

vector<vector<MatrixXd>> SparseGridCollocation::interpolate1(double coef, double tsec, int b, int d, double inx1, double inx2, double r, double sigma, double T, double E,
	vector<MatrixXd> lamb2, vector<MatrixXd> TX2, vector<MatrixXd> C2, vector<MatrixXd> A2, vector<MatrixXd> lamb3, vector<MatrixXd> TX3, vector<MatrixXd> C3, vector<MatrixXd> A3)
{
	MatrixXd N = primeNMatrix(b, d);
	vector<MatrixXd> Lamb;
	vector<MatrixXd> TX;
	vector<MatrixXd> C;
	vector<MatrixXd> A;

	//parfor i = 1 : ch
	for (int i = 0; i < N.rows(); i++)
	{
		//	[lamb{ i }, TX{ i }, C{ i }, A{ i }, PU{ i }] = ...
		//	shapelambda2D_1(coef, tsec, r, sigma, T, E, inx1, inx2, N(i, :), lamb2, TX2, C2, A2, lamb3, TX3, C3, A3);
		vector<MatrixXd> res = shapelambda2D_1(coef, tsec, r, sigma, T, E, inx1, inx2, N.row(i), lamb2, TX2, C2, A2, lamb3, TX3, C3, A3);
		Lamb.push_back(res[0]);
		TX.push_back(res[1]);
		C.push_back(res[2]);
		A.push_back(res[3]);
		//end
	}

	vector<vector<MatrixXd>> result;
	result.push_back(Lamb);
	result.push_back(TX);
	result.push_back(C);
	result.push_back(A);

	return result;
}

vector<vector<MatrixXd>> SparseGridCollocation::interpolate2(double coef, double tsec, int b, int d, double inx1, double inx2, double r, double sigma, double T, double E,
	vector<MatrixXd> lamb2, vector<MatrixXd> TX2, vector<MatrixXd> C2, vector<MatrixXd> A2,
	vector<MatrixXd> lamb3, vector<MatrixXd> TX3, vector<MatrixXd> C3, vector<MatrixXd> A3,
	vector<MatrixXd> lamb_3, vector<MatrixXd> TX_3, vector<MatrixXd> C_3, vector<MatrixXd> A_3,
	vector<MatrixXd> lamb4, vector<MatrixXd> TX4, vector<MatrixXd> C4, vector<MatrixXd> A4)
{
	MatrixXd N = primeNMatrix(b, d);
	vector<MatrixXd> Lamb;
	vector<MatrixXd> TX;
	vector<MatrixXd> C;
	vector<MatrixXd> A;

	//parfor i = 1 : ch
	for (int i = 0; i < N.rows(); i++)
	{
		//	[lamb{ i }, TX{ i }, C{ i }, A{ i }, PU{ i }] = ...
		//	shapelambda2D_1(coef, tsec, r, sigma, T, E, inx1, inx2, N(i, :), lamb2, TX2, C2, A2, lamb3, TX3, C3, A3);
		vector<MatrixXd> res = shapelambda2D_2(coef, tsec, r, sigma, T, E, inx1, inx2, N.row(i), 
			lamb2, TX2, C2, A2, lamb3, TX3, C3, A3, 
			lamb_3, TX_3, C_3, A_3,
			lamb4, TX4, C4, A4);
		Lamb.push_back(res[0]);
		TX.push_back(res[1]);
		C.push_back(res[2]);
		A.push_back(res[3]);
		//end
	}

	vector<vector<MatrixXd>> result;
	result.push_back(Lamb);
	result.push_back(TX);
	result.push_back(C);
	result.push_back(A);

	return result;
}

MatrixXd SparseGridCollocation::subnumber(int b, int d)
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
			Logger::WriteMessage(Common::printMatrix(indextemp).c_str());
			//[s, ~] = size(indextemp);
			int s = indextemp.rows();
			//ntop = nbot + s - 1;
			int ntop = nbot + s - 1;
			MatrixXd l(ntop, d);

			//L(nbot:ntop, 1) = i*ones(s, 1);
			MatrixXd ones = MatrixXd::Ones(s, 1);

			l.block(nbot - 1, 0, ntop - nbot + 1, 1) = ones.array() * (i + 1);
			Logger::WriteMessage(Common::printMatrix(l).c_str());

			//L(nbot:ntop, 2 : d) = indextemp;
			l.block(nbot - 1, 1, ntop - nbot + 1, d - 1) = indextemp;
			Logger::WriteMessage(Common::printMatrix(l).c_str());
			//nbot = ntop + 1;
			nbot = ntop + 1;
			//end

			if (L.rows() > 0)
			{
				l.block(0, 0, l.rows()-1, l.cols()) = L;
				Logger::WriteMessage(Common::printMatrix(l).c_str());
			}
			L = l;
			Logger::WriteMessage(Common::printMatrix(L).c_str());
		}
		//	end
		
	}
	return L;
}

MatrixXd SparseGridCollocation::PDE(MatrixXd node, double r, double sigma,
	vector<MatrixXd> lambda2, vector<MatrixXd> TX2, vector<MatrixXd> C2, vector<MatrixXd> A2,
	vector<MatrixXd> lambda3, vector<MatrixXd> TX3, vector<MatrixXd> C3, vector<MatrixXd> A3)
{
	// This is used in PDE system re - construct for PDE
		//[N, ~] = size(node);
	int N = node.rows();
	//ch2 = length(TX2);
	int ch2 = TX2.size();
	//U2 = ones(N, ch2);
	MatrixXd U2 = MatrixXd::Ones(N, ch2);

	//for j = 1:ch2
	for (int j = 0; j < ch2; j++)
	{
		//[FAI2, FAI2_t, FAI2_x, FAI2_xx] = mq2d(node, TX2{ j }, A2{ j }, C2{ j });
		vector<MatrixXd> mqd = mqd2(node, TX2[j], A2[j], C2[j]);
		//   this equation is determined specially by B - S
			//U2(:, j) = FAI2_t*lamda2{ j } +sigma ^ 2 / 2 * FAI2_xx*lamda2{ j } +r*FAI2_x*lamda2{ j } -r*FAI2*lamda2{ j };
		U2.col(j) = mqd[1] * lambda2[j] + (pow(sigma, 2) / 2) * mqd[3] + r * mqd[2] * lambda2[j] - r * mqd[0] * lambda2[j];
		//end
	}
		//ch3 = length(TX3);
	int ch3 = TX3.size();
	//U3 = ones(N, ch3);
	MatrixXd U3 = MatrixXd::Ones(N, ch3);
	//for j = 1:ch3
	for (int j = 0; j < ch3; j++)
	{
		//[FAI3, FAI3_t, FAI3_x, FAI3_xx] = mq2d(node, TX3{ j }, A3{ j }, C3{ j });
		vector<MatrixXd> mqd = mqd2(node, TX3[j], A3[j], C3[j]);
		//   this equation is determined specially by B - S
			//U3(:, j) = FAI3_t*lamda3{ j } +sigma ^ 2 / 2 * FAI3_xx*lamda3{ j } +r*FAI3_x*lamda3{ j } -r*FAI3*lamda3{ j };
		//end
		U3.col(j) = mqd[1] * lambda3[j] + (pow(sigma, 2) / 2) * mqd[3] + r * mqd[2] * lambda3[j] - r * mqd[0] * lambda3[j];
	}
		//output is depending on the combination tech
		//output = (sum(U3, 2) - sum(U2, 2));
	MatrixXd output = U3.rowwise().sum() - U2.rowwise().sum();
	
	return output;
}

double SparseGridCollocation::inner_test(double t, double x, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A )
{
	// This is used in the PDE system re - construct for initial and boundary conditions
	int ch = TX.size();
	vector<MatrixXd> V;
	//for j = 1:ch
	for (int j = 0; j < ch; j++)
	{
		//   multiquadric RBF......
		//     V1 = sqrt(((t - TX{ j }(:, 1))). ^ 2 + (C{ j }(1, 1). / A{ j }(1, 1)). ^ 2);
	//     V2 = sqrt(((x - TX{ j }(:, 2))). ^ 2 + (C{ j }(1, 2). / A{ j }(1, 2)). ^ 2);
	//     VV = V1.*V2;
	//     V(j) = VV'*lamb{j};
		//   .....................
		//   Gaussian RBF  .......
		//FAI1 = exp(-(A{ j }(1, 1)*(t - TX{ j }(:, 1))). ^ 2 / C{ j }(1, 1) ^ 2);
		MatrixXd square1 = (A[j](0, 0) * (t - (TX[j].col(0).array())).array()) * (A[j](0, 0) * (t - (TX[j].col(0).array())).array());
		MatrixXd FAI1 = - square1 / (C[j](0, 0) * C[j](0, 0));

		//FAI2 = exp(-(A{ j }(1, 2)*(x - TX{ j }(:, 2))). ^ 2 / C{ j }(1, 2) ^ 2);
		MatrixXd square2 = (A[j](0, 1) * (t - (TX[j].col(1).array())).array()) * (A[j](0, 1) * (t - (TX[j].col(1).array())).array());
		MatrixXd FAI2 = -square2 / (C[j](0, 1) * C[j](0, 1));
		//D = FAI1.*FAI2;
		VectorXd D = FAI1.cwiseProduct(FAI2).eval();
		//V(j) = D'*lamb{j};
		V.push_back(D.transpose() * lamb[j]);
			//   .....................
			//end
	}

		//output = sum(V);
	double output = 0;
	int i = 0;
	for (vector<MatrixXd>::iterator it = V.begin(); it < V.end(); it++, i++)
	{
		output += V[i].sum();

	}
	
	return output;
}

void SparseGridCollocation::MuSIK()
{
	double E = 100;// strike price

		double r = 0.03; // interest rate
		double sigma = 0.15;
	double T = 1; // Maturity
		double inx1 = 0; // stock price S belongs to[inx1 inx2]
		double inx2 = 3 * E;
		//TODO: load from smooth initial:
		double Tdone = 0.135;
	double tsec = T - Tdone; // Initial time boundary for sparse grid
	int d = 2; // dimension
	double coef = 2; // coef stands for the connection constant number

	int ch = 10000;
	
	//t = linspace(0, tsec, N(1, 1));
	VectorXd x = VectorXd::LinSpaced(ch, inx1, inx2);
	VectorXd t = VectorXd::Zero(ch, 1);

	//TX = [t x']; //% testing nodes
	MatrixXd TX(ch, 2);
	TX << t, x;

		//% na, nb = level n + dimension d - 1
		int na = 3;
	int nb = 2;
	//tic


	// Level 2 ....lamb stands for \lambda the coefficients, TX stands for nodes
		// C stands for shape parater, A stands for scale parameter
	vector<vector<MatrixXd>> res3 = interpolate(coef, tsec, na, d, inx1, inx2, r, sigma, T, E);

	vector<vector<MatrixXd>> res2 = interpolate(coef, tsec, nb, d, inx1, inx2, r, sigma, T, E);

	//Level 3 ....multilevel method has to use all previous information
	vector<vector<MatrixXd>> res4 = interpolate1(coef, tsec, na + 1, d, inx1, inx2, r, sigma, T, E, res2[0], res2[1], res2[2], res2[3], res3[0], res3[0], res3[0], res3[0]);

	vector<vector<MatrixXd>> res_3 = interpolate1(coef, tsec, nb + 1, d, inx1, inx2, r, sigma, T, E, res2[0], res2[1], res2[2], res2[3], res3[0], res3[0], res3[0], res3[0]);

	//ttt(2) = toc;
	// .............
	// Level 4 ....higher level needs more information
	//[lamb5, TX5, C5, A5, PU5] = interplant_2(coef, tsec, na + 2, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2, lamb3, TX3, C3, A3, ...
	//		lamb_3, TX_3, C_3, A_3, lamb4, TX4, C4, A4);

	//[lamb_4, TX_4, C_4, A_4, PU_4] = interplant_2(coef, tsec, nb + 2, d, inx1, inx2, r, sigma, T, E, lamb2, TX2, C2, A2, lamb3, TX3, C3, A3, ...
	//	lamb_3, TX_3, C_3, A_3, lamb4, TX4, C4, A4);
	
	//ttt(3) = toc;
}


void SparseGridCollocation::MuSIKGeneric()
{
	double E = 100;// strike price

	double r = 0.03; // interest rate
	double sigma = 0.15;
	double T = 1; // Maturity
	double inx1 = 0; // stock price S belongs to[inx1 inx2]
	double inx2 = 3 * E;
	//TODO: load from smooth initial:
	double Tdone = 0.135;
	double tsec = T - Tdone; // Initial time boundary for sparse grid
	int d = 2; // dimension
	double coef = 2; // coef stands for the connection constant number

	int ch = 10000;

	//t = linspace(0, tsec, N(1, 1));
	VectorXd x = VectorXd::LinSpaced(ch, inx1, inx2);
	VectorXd t = VectorXd::Zero(ch, 1);

	//TX = [t x']; //% testing nodes
	MatrixXd TX(ch, 2);
	TX << t, x;

	//% na, nb = level n + dimension d - 1
	int na = 3;
	int nb = 2;
	//tic


	// Level 2 ....lamb stands for \lambda the coefficients, TX stands for nodes
	// C stands for shape parater, A stands for scale parameter
	vector<vector<MatrixXd>> res3 = interpolate(coef, tsec, na, d, inx1, inx2, r, sigma, T, E);

	vector<vector<MatrixXd>> res2 = interpolate(coef, tsec, nb, d, inx1, inx2, r, sigma, T, E);

	//Level 3 ....multilevel method has to use all previous information
	vector<string> level3 = {"2","3"};
	interpolateGeneric("4", coef, tsec, na + 1, d, inx1, inx2, r, sigma, T, E, level3);
	interpolateGeneric("_3", coef, tsec, nb + 1, d, inx1, inx2, r, sigma, T, E, level3);

	//ttt(2) = toc;
	//Level 4 ....higher level needs more information
	vector<string> level4 = { "2","3","_3","4" };
	interpolateGeneric("5", coef, tsec, na + 1, d, inx1, inx2, r, sigma, T, E, level4);
	interpolateGeneric("_4", coef, tsec, nb + 1, d, inx1, inx2, r, sigma, T, E, level4);

	//ttt(3) = toc;

	//Level 5
	vector<string> level5 = { "2","3","_3","4","_4","5" };
	interpolateGeneric("6", coef, tsec, na + 1, d, inx1, inx2, r, sigma, T, E, level5);
	interpolateGeneric("_5", coef, tsec, nb + 1, d, inx1, inx2, r, sigma, T, E, level5);

	//ttt(4) = toc;

	//Level 6
	vector<string> level6 = { "2","3","_3","4","_4","5" };
	interpolateGeneric("7", coef, tsec, na + 1, d, inx1, inx2, r, sigma, T, E, level6);
	interpolateGeneric("_6", coef, tsec, nb + 1, d, inx1, inx2, r, sigma, T, E, level6);

	//ttt(5) = toc;

	//Level7
	vector<string> level7 = { "2","3","_3","4","_4","5","_5","6" };
	interpolateGeneric("8", coef, tsec, na + 1, d, inx1, inx2, r, sigma, T, E, level7);
	interpolateGeneric("_7", coef, tsec, nb + 1, d, inx1, inx2, r, sigma, T, E, level7);

}
void SparseGridCollocation::interpolateGeneric(string prefix, double coef, double tsec, int b, int d, double inx1, double inx2, double r, double sigma, double T, double E, vector<string> keys)
{
	vector<MatrixXd> Lamb;
	vector<MatrixXd> TX;
	vector<MatrixXd> C;
	vector<MatrixXd> A;
	vector<MatrixXd> U;

	MatrixXd N = primeNMatrix(b, d);

	for (int i = 0; i < N.rows(); i++)
	{
		vector<MatrixXd> res = shapelambda2DGeneric(coef, tsec, r, sigma, T, E, inx1, inx2, N.row(i), keys);

		Lamb.push_back(res[0]);
		TX.push_back(res[1]);
		C.push_back(res[2]);
		A.push_back(res[3]);
		U.push_back(res[4]);
	}

	vector<vector<MatrixXd>> result;
	result.push_back(Lamb);
	result.push_back(TX);
	result.push_back(C);
	result.push_back(A);
	result.push_back(U);
	vInterpolation[prefix] = result;

}


vector<MatrixXd> SparseGridCollocation::shapelambda2DGeneric(double coef, double tsec, double r, double sigma, double T, double E, double inx1, double inx2, MatrixXd N, vector<string> keys)
{

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

	vector<MatrixXd> mqd = mqd2(TX1, TX1, a, c);
	MatrixXd FAI = mqd[0];
	MatrixXd FAI_t = mqd[1];
	MatrixXd FAI_x = mqd[2];
	MatrixXd FAI_xx = mqd[3];

	MatrixXd pa = (sigma * sigma * FAI_xx.array() / 2);
	MatrixXd pb = r*FAI_x.array() - r*FAI.array();
	MatrixXd P = FAI_t.array() + pa.array() + pb.array();

	VectorXd U = MatrixXd::Zero(num, 1);
	for (int s =0; s < keys.size(); s+=2)
	{
		string k1 = keys[s];
		string k2 = keys[s+1];
		U -= PDE(TX1, r, sigma, 
			vInterpolation[k1][0], vInterpolation[k1][1], vInterpolation[k1][2], vInterpolation[k1][3],
			vInterpolation[k2][0], vInterpolation[k2][1], vInterpolation[k2][2], vInterpolation[k2][3]);
	}

	for (int i = 0; i < num; i++)
	{
		if (abs(TX1(i, 1) - inx1) < DBL_EPSILON || abs(TX1(i, 1) - inx2) < DBL_EPSILON)
		{
			P.row(i) = FAI.row(i);
			double max = TX1(i, 1) - E*exp(-r * (T - TX1(i, 0)));
			U(i) = 0;
			double sub = 0;
			for (int s = 0; s < keys.size(); s += 2)
			{
				string k1 = keys[s];
				string k2 = keys[s + 1];
				sub += (inner_test(TX1(i, 0), TX1(i, 1), vInterpolation[k2][0], vInterpolation[k2][0], vInterpolation[k2][0], vInterpolation[k2][0])
					- inner_test(TX1(i, 0), TX1(i, 1), vInterpolation[k1][0], vInterpolation[k1][0], vInterpolation[k1][0], vInterpolation[k1][0]));
			}

			if (max > 0)
				U(i) = max - sub;
			else
				U(i) = 0 - sub;
		}

		if (abs(TX1(i, 0) - tsec) < DBL_EPSILON)
		{
			P.row(i) = FAI.row(i);
			double sub = 0;
			for (int s = 0; s < keys.size(); s += 2)
			{
				string k1 = keys[s];
				string k2 = keys[s + 1];
				sub += (inner_test(TX1(i, 0), TX1(i, 1), vInterpolation[k2][0], vInterpolation[k2][0], vInterpolation[k2][0], vInterpolation[k2][0])
					- inner_test(TX1(i, 0), TX1(i, 1), vInterpolation[k1][0], vInterpolation[k1][0], vInterpolation[k1][0], vInterpolation[k1][0]));
			}
			U(i) = PPP::Calculate(TX1.row(i)) - sub;
		}
	}
	MatrixXd TX = TX1;
	
	PartialPivLU<MatrixXd> lu = PartialPivLU<MatrixXd>(P);
	MatrixXd J = lu.matrixLU().triangularView<UpLoType::Upper>();
	MatrixXd F = lu.matrixLU().triangularView<UpLoType::UnitLower>();;
	//Hack: to get around the fact that Eigen doesn't compute the permutation matrix p correctly
	MatrixXd transform(15, 15);
	transform << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
	MatrixXd Fa = transform * F;

	//Eigen also seems to solve with different rounding, maybe a double arithmetic issue:
	//Jlamda = F\U;
	MatrixXd Jlamda = Fa.lu().solve(U);
	//lamb = J\Jlamda;
	MatrixXd l = J.lu().solve(Jlamda);

	vector<MatrixXd> result = { l, TX, c, a, U };
	return result;
}

