#include "stdafx.h"
#include "Test.h"
#include "RBF.h"
#include "Math.h"
#include "Common.h"


#include <iostream>
#include <sstream>
#include <string>
#include <iomanip> 
#include <fstream>

//#include "CppUnitTest.h"
//using namespace Microsoft::VisualStudio::CppUnitTestFramework;



Leicester::SparseGridCollocation::Test::Test()
{
}


Leicester::SparseGridCollocation::Test::~Test()
{
}

double Leicester::SparseGridCollocation::Test::inner(double t, double x, const vector<MatrixXd> &lamb, const vector<MatrixXd> &TX, const vector<MatrixXd> &C, const vector<MatrixXd> &A)
{

	int ch = TX.size();
	vector<double> V;

	for (int j = 0; j < ch; j++)
	{

		MatrixXd square1 = (A[j](0, 0) * (t - (TX[j].col(0).array())).array()) * (A[j](0, 0) * (t - (TX[j].col(0).array())).array());
		MatrixXd a = -square1 / (C[j](0, 0) * C[j](0, 0));
		MatrixXd FAI1 = a.array().exp();

		MatrixXd square2 = (A[j](0, 1) * (x - (TX[j].col(1).array())).array()) * (A[j](0, 1) * (x - (TX[j].col(1).array())).array());
		MatrixXd b = -square2 / (C[j](0, 1) * C[j](0, 1));
		MatrixXd FAI2 = b.array().exp();

		VectorXd D = FAI1.cwiseProduct(FAI2).eval();
		MatrixXd d = D.transpose();


		MatrixXd res = d * lamb[j];


		V.push_back(res(0, 0));

	}


	double output = 0;
	int i = 0;
	for (vector<double>::iterator it = V.begin(); it < V.end(); it++, i++)
	{
		output += V[i];// .sum();

	}

	return output;
}

//TXYZ should be a row vector for the N dimension matrix
double Leicester::SparseGridCollocation::Test::innerND(MatrixXd TXYZ, const vector<MatrixXd> &lamb, const vector<MatrixXd> &TX, const vector<MatrixXd> &C, const vector<MatrixXd> &A)
{

	int ch = TX.size();
	vector<double> V;

	for (int j = 0; j < ch; j++)
	{
		MatrixXd VV = MatrixXd::Ones(TX[j].rows(),1);
		for (int i = 0; i < TX[j].cols(); i++)
		{
			MatrixXd square = (A[j](0, i) * (TXYZ(0, i) - (TX[j].col(i).array())).array()) * (A[j](0, i) * (TXYZ(0, i) - (TX[j].col(i).array())).array());
			MatrixXd a = -square / (C[j](0, i) * C[j](0, i));
			MatrixXd FAI = a.array().exp();
			VV.array() = VV.array() * FAI.array();
		}
		
		VV.transposeInPlace();
		MatrixXd res = VV * lamb[j];
		V.push_back(res(0, 0));
	}


	double output = 0;
	int i = 0;
	for (vector<double>::iterator it = V.begin(); it < V.end(); it++, i++)
	{
		output += V[i];// .sum();

	}

	return output;
}


VectorXd Leicester::SparseGridCollocation::Test::inter(MatrixXd X, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A)
{
	// This is used to calculate values on final testing points
	//ch = length(TX);
	int ch = TX.size();

	//[N, ~] = size(X);
	int N = X.rows();
	//V = ones(N, ch);
	MatrixXd V = MatrixXd::Ones(N, ch);
	//for j = 1:ch
	vector<MatrixXd> res;
	for (int j = 0; j < ch; j++)
	{
		//[D] = mq2d(X, TX{ j }, A{ j }, C{ j });
		vector<MatrixXd> D = RBF::Gaussian2D(X, TX[j], A[j], C[j]);

		//V(:, j) = D*lamb{ j };
		VectorXd v = D[0] * lamb[j];
		V.col(j) = v;
		//end
	}
	//output = sum(V, 2);
	VectorXd output = V.colwise().sum();
	return output;
}