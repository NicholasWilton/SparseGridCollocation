#pragma once
#include "stdafx.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using namespace std;

class PDE
{
public:
	PDE();
	~PDE();
	static MatrixXd BlackScholes(MatrixXd node, double r, double sigma,
		vector<MatrixXd> lambda2, vector<MatrixXd> TX2, vector<MatrixXd> C2, vector<MatrixXd> A2,
		vector<MatrixXd> lambda3, vector<MatrixXd> TX3, vector<MatrixXd> C3, vector<MatrixXd> A3);
};

