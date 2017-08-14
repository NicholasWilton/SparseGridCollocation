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

MatrixXd PDE::BlackScholesNd(const MatrixXd &node, double r, double sigma, vector<string> keys, const map<string, vector<vector<MatrixXd>> > * state)
{
	int N = node.rows();
	vector<MatrixXd> Us;
	for (auto key : keys)
	{
		vector<vector<MatrixXd>> item = state->at(key); //0-lambda, 1-TX, 2-C, 3-A, 4-U
		int ch2 = item[1].size();
		MatrixXd U = MatrixXd::Ones(N, ch2);

		for (int j = 0; j < ch2; j++)
		{
			vector<MatrixXd> mqd = RBF::mqNd(node, item[1][j], item[3][j], item[2][j]);
			MatrixXd a = mqd[1] * item[0][j]; // lambda * dV/dt
			MatrixXd b = (pow(sigma, 2) / 2) * mqd[3] * item[0][j]; // 1/2 sum-i sum-j sigma^2 rho-ij Si Sj d2V/dSi dSj
			MatrixXd c = r * mqd[2] * item[0][j]; // sum-i (r - q-i) Si dV/dSi
			MatrixXd d = r * mqd[0] * item[0][j]; //rV
			U.col(j) = a + b + c - d;
		}
		Us.push_back(U);
	}

	int n = Us.size();
	MatrixXd output(Us[0].rows(), 1);;
	for (int i = 0; i < Us.size(); i++)
	{
		int coeff = Common::BinomialCoefficient(n-1, i);
		MatrixXd U = Us[i];
		VectorXd sum = U.rowwise().sum();
		if (i % 2 == 0)
			output.col(0).array() -= (coeff * sum).array();
		else
			output.col(0).array() += (coeff * sum).array();
	}

	return output;
}

