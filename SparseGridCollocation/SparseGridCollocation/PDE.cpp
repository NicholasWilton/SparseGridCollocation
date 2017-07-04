#include "stdafx.h"
#include "PDE.h"
#include "RBF.h"


PDE::PDE()
{
}


PDE::~PDE()
{
}

MatrixXd PDE::BlackScholes(MatrixXd node, double r, double sigma,
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
		vector<MatrixXd> mqd = RBF::mqd2(node, TX2[j], A2[j], C2[j]);
		//   this equation is determined specially by B - S
		//U2(:, j) = FAI2_t*lamda2{ j } +sigma ^ 2 / 2 * FAI2_xx*lamda2{ j } +r*FAI2_x*lamda2{ j } -r*FAI2*lamda2{ j };
		MatrixXd a = mqd[1] * lambda2[j];
		MatrixXd b = (pow(sigma, 2) / 2) * mqd[3] * lambda2[j];
		MatrixXd c = r * mqd[2] * lambda2[j];
		MatrixXd d = r * mqd[0] * lambda2[j];
		U2.col(j) = a + b + c - d;
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
		vector<MatrixXd> mqd = RBF::mqd2(node, TX3[j], A3[j], C3[j]);
		//   this equation is determined specially by B - S
		//U3(:, j) = FAI3_t*lamda3{ j } +sigma ^ 2 / 2 * FAI3_xx*lamda3{ j } +r*FAI3_x*lamda3{ j } -r*FAI3*lamda3{ j };
		//end
		MatrixXd a = mqd[1] * lambda3[j];
		MatrixXd b = (pow(sigma, 2) / 2) * mqd[3] * lambda3[j];
		MatrixXd c = r * mqd[2] * lambda3[j];
		MatrixXd d = r * mqd[0] * lambda3[j];
		U3.col(j) = a + b + c - d;
	}
	//output is depending on the combination tech
	//output = (sum(U3, 2) - sum(U2, 2));
	MatrixXd output = U3.rowwise().sum() - U2.rowwise().sum();

	return output;
}
