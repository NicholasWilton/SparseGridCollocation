#include "stdafx.h"
#include "RBF.h"
#include <math.h>


RBF::RBF()
{
}


RBF::~RBF()
{
}

VectorXd RBF::exp(const VectorXd &v)
{
	VectorXd result(v.rows());
	for (int i = 0; i < v.size(); i++)
	{
		result[i] = std::exp(v[i]);
	}
	return result;
}

vector<MatrixXd> RBF::mqd2(const MatrixXd &TP, const MatrixXd &CN, const MatrixXd &A, const MatrixXd &C)
{
	vector<MatrixXd> result;
	int Num = CN.rows();
	int N = TP.rows();

	MatrixXd D(N, Num);
	D.fill(1.0);
	MatrixXd Dt(N, Num);
	Dt.fill(1.0);
	MatrixXd Dx(N, Num);
	Dx.fill(1.0);
	MatrixXd Dxx(N, Num);
	Dxx.fill(1.0);

	for (int j = 0; j < Num; j++)
	{
		VectorXd a1 = A(0, 0)*(TP.col(0).array() - CN(j, 0));
		VectorXd b1 = -(a1.array() * a1.array()) / (C(0, 0) *C(0, 0));
		VectorXd FAI1 = b1.array().exp();
		//VectorXd FAI1 = RBF::exp(b1);

		VectorXd a2 = A(0, 1)*(TP.col(1).array() - CN(j, 1));
		VectorXd b2 = -(a2.array() * a2.array()) / (C(0, 1) *C(0, 1));

		VectorXd FAI2 = b2.array().exp();
		//VectorXd FAI2 = RBF::exp(b2);
		D.col(j) = FAI1.array() * FAI2.array();

		VectorXd a3 = -2 * (A(0, 0) / C(0, 0)) * (A(0, 0) / C(0, 0)) * (TP.col(0).array() - CN(j, 0));
		VectorXd b3 = a3.array() * FAI1.array();
		VectorXd c3 = b3.array() * FAI2.array();
		Dt.col(j) = c3;

		VectorXd a4 = -2 * (A(0, 1) / C(0, 1)) * (A(0, 1) / C(0, 1)) * (TP.col(1).array() - CN(j, 1));
		VectorXd b4 = TP.col(1).array() * a4.array() * FAI1.array();
		VectorXd c4 = b4.array() * FAI2.array();
		Dx.col(j) = c4;

		double sA = A(0, 1) * A(0, 1);
		double qA = A(0, 1) * A(0, 1) * A(0, 1) * A(0, 1);
		double sC = C(0, 1) * C(0, 1);
		double qC = C(0, 1) * C(0, 1) * C(0, 1) * C(0, 1);
		VectorXd dTpCn = TP.col(1).array() - CN(j, 1);

		VectorXd a5 = 4 * qA * (dTpCn.array() * dTpCn.array() / qC);
		VectorXd b5 = -2 * sA / sC + a5.array();
		VectorXd c5 = b5.array()  * FAI2.array() * FAI1.array();
		VectorXd d5 = (TP.col(1).array() * TP.col(1).array()).array() * c5.array();
		Dxx.col(j) = d5;
	}
	result.push_back(D);
	result.push_back(Dt);
	result.push_back(Dx);
	result.push_back(Dxx);
	return result;
}