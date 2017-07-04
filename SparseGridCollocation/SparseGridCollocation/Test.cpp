#include "stdafx.h"
#include "Test.h"
#include "RBF.h"


Test::Test()
{
}


Test::~Test()
{
}

double Test::inner(double t, double x, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A)
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
		MatrixXd FAI1 = (-square1 / (C[j](0, 0) * C[j](0, 0))).array().exp();

		//FAI2 = exp(-(A{ j }(1, 2)*(x - TX{ j }(:, 2))). ^ 2 / C{ j }(1, 2) ^ 2);
		MatrixXd square2 = (A[j](0, 1) * (t - (TX[j].col(1).array())).array()) * (A[j](0, 1) * (t - (TX[j].col(1).array())).array());
		MatrixXd FAI2 = (-square2 / (C[j](0, 1) * C[j](0, 1))).array().exp();
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

VectorXd Test::inter(MatrixXd X, vector<MatrixXd> lamb, vector<MatrixXd> TX, vector<MatrixXd> C, vector<MatrixXd> A)
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
		vector<MatrixXd> D = RBF::mqd2(X, TX[j], A[j], C[j]);

		//V(:, j) = D*lamb{ j };
		VectorXd v = D[0] * lamb[j];
		V.col(j) = v;
		//end
	}
	//output = sum(V, 2);
	VectorXd output = V.colwise().sum();
	return output;
}