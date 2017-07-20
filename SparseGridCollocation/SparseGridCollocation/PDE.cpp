#include "stdafx.h"
#include "PDE.h"
#include "RBF.h"
#include "Common.h"
#include "MatrixXdm.h"

//#include "CppUnitTest.h"
//using namespace Microsoft::VisualStudio::CppUnitTestFramework;


PDE::PDE()
{
}


PDE::~PDE()
{
}

MatrixXd PDE::BlackScholes(const MatrixXd &node, double r, double sigma,
	const vector<MatrixXd> &lambda2, const vector<MatrixXd> &TX2, const vector<MatrixXd> &C2, const vector<MatrixXd> &A2,
	const vector<MatrixXd> &lambda3, const vector<MatrixXd> &TX3, const vector<MatrixXd> &C3, const vector<MatrixXd> A3)
{
	int N = node.rows();
	int ch2 = TX2.size();
	MatrixXd U2 = MatrixXd::Ones(N, ch2);

	for (int j = 0; j < ch2; j++)
	{
		vector<MatrixXd> mqd = RBF::mqd2(node, TX2[j], A2[j], C2[j]);
		MatrixXd a = mqd[1] * lambda2[j];
		MatrixXd b = (pow(sigma, 2) / 2) * mqd[3] * lambda2[j];
		MatrixXd c = r * mqd[2] * lambda2[j];
		MatrixXd d = r * mqd[0] * lambda2[j];
		U2.col(j) = a + b + c - d;
	}
	int ch3 = TX3.size();
	MatrixXd U3 = MatrixXd::Ones(N, ch3);
	for (int j = 0; j < ch3; j++)
	{
		vector<MatrixXd> mqd = RBF::mqd2(node, TX3[j], A3[j], C3[j]);
		MatrixXd a = mqd[1] * lambda3[j];
		MatrixXd b = (pow(sigma, 2) / 2) * mqd[3] * lambda3[j];
		MatrixXd c = r * mqd[2] * lambda3[j];
		MatrixXd d = r * mqd[0] * lambda3[j];
		U3.col(j) = a + b + c - d;
	}
	MatrixXd s1 = U3.rowwise().sum();
	MatrixXd s2 = U2.rowwise().sum();
	MatrixXd output = s1 - s2;

	return output;
}

