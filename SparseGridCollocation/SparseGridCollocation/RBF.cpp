#include "stdafx.h"
#include "RBF.h"


RBF::RBF()
{
}


RBF::~RBF()
{
}

vector<MatrixXd> RBF::mqd2(MatrixXd TP, MatrixXd CN, MatrixXd A, MatrixXd C)
{
	vector<MatrixXd> result;
	//[Num, ~] = size(CN);
	int Num = CN.rows();
	//[N, ~] = size(TP);
	int N = TP.rows();
	//[D, Dt, Dx, Dxx] = deal(ones(N, Num));

	MatrixXd D(N, Num);
	D.fill(1.0);
	MatrixXd Dt(N, Num);
	Dt.fill(1.0);
	MatrixXd Dx(N, Num);
	Dx.fill(1.0);
	MatrixXd Dxx(N, Num);
	Dxx.fill(1.0);

	//MatrixXd cn = *CN;
	//MatrixXd a = *A;
	//MatrixXd c = *C;
	//MatrixXd tp = *TP;
	//for j = 1:Num
	for (int j = 0; j < Num; j++)
	{

		//	FAI1 = exp(-(A(1, 1)*(TP(:, 1) - CN(j, 1))). ^ 2 / C(1, 1) ^ 2);
		VectorXd a1 = A(0, 0)*(TP.col(0).array() - CN(j, 0));
		//Logger::WriteMessage(Common::printMatrix(a1).c_str());

		VectorXd b1 = -(a1.array() * a1.array()) / (C(0, 0) *C(0, 0));
		//Logger::WriteMessage(Common::printMatrix(b1).c_str());

		VectorXd FAI1 = b1.array().exp();
		//Logger::WriteMessage(Common::printMatrix(FAI1).c_str());

		//FAI2 = exp(-(A(1, 2)*(TP(:, 2) - CN(j, 2))). ^ 2 / C(1, 2) ^ 2);
		VectorXd a2 = A(0, 1)*(TP.col(1).array() - CN(j, 1));
		//Logger::WriteMessage(Common::printMatrix(a2).c_str());

		VectorXd b2 = -(a2.array() * a2.array()) / (C(0, 1) *C(0, 1));
		//Logger::WriteMessage(Common::printMatrix(b2).c_str());

		VectorXd FAI2 = b2.array().exp();
		//Logger::WriteMessage(Common::printMatrix(FAI2).c_str());
		//D(:, j) = FAI1.*FAI2;
		D.col(j) = FAI1.array() * FAI2.array();
		//Logger::WriteMessage(Common::printMatrix(*D).c_str());

		//TODO: this is basically how Matlab handles overloading:
		//if nargout > 1

		//	Dt(:, j) = -2 * (A(1, 1) / C(1, 1)) ^ 2 * (TP(:, 1) - CN(j, 1)).*FAI1.*FAI2;
		VectorXd a3 = -2 * (A(0, 0) / C(0, 0)) * (A(0, 0) / C(0, 0)) * (TP.col(0).array() - CN(j, 0));
		VectorXd b3 = a3.array() * FAI1.array();
		VectorXd c3 = b3.array() * FAI2.array();
		Dt.col(j) = c3;
		//Logger::WriteMessage(Common::printMatrix(Dt).c_str());

		//Dx(:, j) = TP(:, 2).*(-2 * (A(1, 2) / C(1, 2)) ^ 2 * (TP(:, 2) - CN(j, 2)).*FAI1.*FAI2);
		VectorXd a4 = -2 * (A(0, 1) / C(0, 1)) * (A(0, 1) / C(0, 1)) * (TP.col(1).array() - CN(j, 1));
		VectorXd b4 = TP.col(1).array() * a4.array() * FAI1.array();
		VectorXd c4 = b4.array() * FAI2.array();
		Dx.col(j) = c4;
		//Logger::WriteMessage(Common::printMatrix(Dx).c_str());
		//Dxx(:, j) = TP(:, 2).^2.*((-2 * A(1, 2) ^ 2 / C(1, 2) ^ 2 + 4 * A(1, 2) ^ 4 * (TP(:, 2) - CN(j, 2)).^2. / C(1, 2) ^ 4).*FAI2.*FAI1);
		double sA = A(0, 1) * A(0, 1);
		double qA = A(0, 1) * A(0, 1) * A(0, 1) * A(0, 1);
		double sC = C(0, 1) * C(0, 1);
		double qC = C(0, 1) * C(0, 1) * C(0, 1) * C(0, 1);
		VectorXd dTpCn = TP.col(1).array() - CN(j, 1);

		VectorXd a5 = 4 * qA * (dTpCn.array() * dTpCn.array() / qC);
		//Logger::WriteMessage(Common::printMatrix(a5).c_str());
		VectorXd b5 = -2 * sA / sC + a5.array();
		//Logger::WriteMessage(Common::printMatrix(b5).c_str());
		VectorXd c5 = b5.array()  * FAI2.array() * FAI1.array();
		//Logger::WriteMessage(Common::printMatrix(c5).c_str());
		VectorXd d5 = (TP.col(1).array() * TP.col(1).array()).array() * c5.array();
		//Logger::WriteMessage(Common::printMatrix(d5).c_str());
		//VectorXd c5 = b5.array() * FAI2.array() * FAI1.array();
		Dxx.col(j) = d5;
		//Logger::WriteMessage(Common::printMatrix(Dxx).c_str());
	}
	result.push_back(D);
	result.push_back(Dt);
	result.push_back(Dx);
	result.push_back(Dxx);
	return result;
}